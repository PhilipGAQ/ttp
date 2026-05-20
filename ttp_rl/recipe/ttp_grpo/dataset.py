# Copyright 2024 verl-gap authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GAP-GRPO Dataset

Supports data format from gap-r1 framework:
- query: [instruction, query] or str
- query_gen: str (ground truth for query rewriting)
- hist_list: [] (user history sequence)
- pos: [] (positive passages)
- neg: [] (negative passages)

The model processes queries with:
1. Generate rewritten query based on history: <think>rewrite query</think><emb>
2. Extract embedding at <emb> token position

For InfoNCE embedding computation, we provide:
- query_prompt: Full prompt with history (for generation/policy)
- query_prompt_no_hist: Prompt without history (for embedding if mask_hist=True)
"""

import copy
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask


class GapGRPODataset(Dataset):
    """
    Dataset for GAP-GRPO training.
    
    Loads JSONL files with the following schema:
    - query: [instruction, query_text] or str
    - query_gen: str (target rewritten query, optional)
    - hist_list: list of history items (optional)
    - pos: list of positive passages
    - neg: list of negative passages
    
    The dataset constructs prompts for query-side generation + embedding task.
    
    For InfoNCE, we provide two versions of query prompt:
    - query_prompt: instruction + history + query (for generation/policy)
    - query_prompt_no_hist: instruction + query only (for embedding when mask_hist=True)
    """
    
    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]
        
        self.data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        
        # Configuration
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/gap_grpo"))
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.max_example_num_per_dataset = config.get("max_example_num_per_dataset", 100000000)
        self.train_group_size = config.get("train_group_size", 8)
        
        # Chat template tokens
        self.base_bos = config.get("base_bos", "")
        self.user_bos = config.get("user_bos", "<|im_start|>user\n")
        self.user_eos = config.get("user_eos", "<|im_end|>\n")
        self.assistant_bos = config.get("assistant_bos", "<|im_start|>assistant\n")
        self.assistant_eos = config.get("assistant_eos", "<|im_end|>")
        
        # Special tokens
        self.think_open = "<think>"
        self.think_close = "</think>"
        self.emb_tok = "<emb>"
        
        self._download()
        self._read_files()
    
    def _download(self):
        """Download data files to local cache if needed."""
        from verl.utils.fs import copy_to_local
        
        for i, data_file in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(src=data_file, cache_dir=self.cache_dir)
    
    def _read_files(self):
        """Read and concatenate all data files."""
        dataframes = []
        for data_file in self.data_files:
            # Load JSONL files
            df = datasets.load_dataset("json", data_files=data_file, split="train")
            if len(df) > self.max_example_num_per_dataset:
                df = df.select(random.sample(range(len(df)), self.max_example_num_per_dataset))
            dataframes.append(df)
        
        self.dataframe = datasets.concatenate_datasets(dataframes) if len(dataframes) > 1 else dataframes[0]
        print(f"[GapGRPODataset] Loaded {len(self.dataframe)} examples from {len(self.data_files)} files")
    
    def __len__(self):
        return len(self.dataframe)
    
    def _build_query_prompt(
        self,
        query: Union[str, List[str]],
        hist_list: Optional[List[str]] = None,
        include_history: bool = True,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Build query prompt for generation task.
        
        Format: base_bos + user_bos + [instruction] + [history] + query + user_eos + assistant_bos
        
        Truncation strategy (when max_tokens is provided):
        1. First, truncate hist_list (remove oldest items first)
        2. If still too long, truncate history content
        3. Query is NEVER truncated (most important)
        
        Args:
            query: Query text or [instruction, query_text]
            hist_list: Optional list of history items
            include_history: If False, omit history from prompt (for embedding)
            max_tokens: Maximum tokens allowed (if provided, applies smart truncation)
        
        The model will generate: <think>rewritten query</think><emb> + assistant_eos
        """
        # Parse query
        if isinstance(query, (list, tuple)) and len(query) >= 2:
            instruction = str(query[0]).strip()
            query_text = str(query[1]).strip()
        else:
            instruction = ""
            query_text = str(query).strip()
        
        # Build base parts (instruction + query) - these are NOT truncated
        base_content_parts = []
        if instruction:
            base_content_parts.append(instruction)
        base_content_parts.append(f"User Query: {query_text}")
        base_content = " ".join(base_content_parts)
        
        # Build base prompt without history to measure base length
        base_prompt = f"{self.base_bos}{self.user_bos}{base_content}{self.user_eos}{self.assistant_bos}"
        
        if not include_history or not hist_list or len(hist_list) == 0:
            return base_prompt
        
        # Calculate available tokens for history
        if max_tokens is not None:
            base_tokens = len(self.tokenizer.encode(base_prompt, add_special_tokens=False))
            available_for_hist = max_tokens - base_tokens - 10  # 10 tokens buffer
            
            if available_for_hist <= 0:
                # No room for history, return base prompt
                return base_prompt
            
            # Try to fit history, truncating oldest items first
            hist_list_copy = list(hist_list)  # Make a copy
            hist_prefix = "User History Sequence: "
            
            while hist_list_copy:
                hist_str = " ".join([str(h).strip() for h in hist_list_copy])
                hist_full = hist_prefix + hist_str
                hist_tokens = len(self.tokenizer.encode(hist_full, add_special_tokens=False))
                
                if hist_tokens <= available_for_hist:
                    # History fits
                    content_parts = []
                    if instruction:
                        content_parts.append(instruction)
                    content_parts.append(hist_full)
                    content_parts.append(f"User Query: {query_text}")
                    content = " ".join(content_parts)
                    return f"{self.base_bos}{self.user_bos}{content}{self.user_eos}{self.assistant_bos}"
                
                # Remove oldest history item (first item)
                hist_list_copy = hist_list_copy[1:]
            
            # No history fits, return base prompt
            return base_prompt
        
        # No truncation needed, include all history
        hist_str = " ".join([str(h).strip() for h in hist_list])
        content_parts = []
        if instruction:
            content_parts.append(instruction)
        content_parts.append(f"User History Sequence: {hist_str}")
        content_parts.append(f"User Query: {query_text}")
        content = " ".join(content_parts)
        
        return f"{self.base_bos}{self.user_bos}{content}{self.user_eos}{self.assistant_bos}"
    
    def _build_passage_text(self, passage: Union[str, List[str]]) -> str:
        """
        Build passage text for embedding.
        
        Format: base_bos + user_bos + [instruction] + passage + user_eos + assistant_bos + <emb> + assistant_eos
        """
        if isinstance(passage, (list, tuple)) and len(passage) >= 2:
            instruction = str(passage[0]).strip()
            passage_text = str(passage[1]).strip()
        else:
            instruction = ""
            passage_text = str(passage).strip()
        
        content = f"{instruction}{passage_text}" if instruction else passage_text
        
        text = f"{self.base_bos}{self.user_bos}{content}{self.user_eos}{self.assistant_bos}{self.emb_tok}{self.assistant_eos}"
        
        return text
    
    def _build_query_prompt_ids(
        self,
        query: Union[str, List[str]],
        hist_list: Optional[List[str]] = None,
        include_history: bool = True,
        max_tokens: Optional[int] = None,
    ) -> Tuple[List[int], str]:
        """
        Build query prompt by manually concatenating token IDs.
        This avoids 'apply_chat_template' adding unwanted system prompts (e.g. "You are a helpful assistant"),
        while providing precise control over history truncation.
        """
        # 1. Encode Headers/Footers
        # We encode these separately to respect the config strings
        user_bos_ids = self.tokenizer.encode(self.base_bos + self.user_bos, add_special_tokens=False)
        user_eos_ids = self.tokenizer.encode(self.user_eos, add_special_tokens=False)
        assistant_bos_ids = self.tokenizer.encode(self.assistant_bos, add_special_tokens=False)
        
        # 2. Encode Content Parts
        if isinstance(query, (list, tuple)) and len(query) >= 2:
            instruction = str(query[0]).strip()
            query_text = str(query[1]).strip()
        else:
            instruction = ""
            query_text = str(query).strip()

        # Space ID for joining parts (equivalent to " ".join)
        space_ids = self.tokenizer.encode(" ", add_special_tokens=False)

        # Instruction
        inst_ids = []
        if instruction:
            inst_ids = self.tokenizer.encode(instruction, add_special_tokens=False)
        
        # Query
        query_prefix_ids = self.tokenizer.encode("User Query: ", add_special_tokens=False)
        query_content_ids = self.tokenizer.encode(query_text, add_special_tokens=False)
        full_query_ids = query_prefix_ids + query_content_ids
        
        # 3. Handle History
        hist_section_ids = []
        if include_history and hist_list:
            hist_prefix_ids = self.tokenizer.encode("User History Sequence: ", add_special_tokens=False)
            
            # Encode all items first
            encoded_hist_items = []
            for h in hist_list:
                encoded_hist_items.append(self.tokenizer.encode(str(h).strip(), add_special_tokens=False))
            
            # Select items based on available length
            selected_items_ids = []
            
            if max_tokens is None:
                selected_items_ids = encoded_hist_items
            else:
                # Calculate base length (user_bos + [inst] + [query] + user_eos + assistant_bos)
                base_len = len(user_bos_ids) + len(user_eos_ids) + len(assistant_bos_ids)
                if inst_ids:
                    base_len += len(inst_ids) + len(space_ids)
                base_len += len(full_query_ids)
                # Add space for hist-query separator
                base_len += len(space_ids) 
                # Add hist prefix
                base_len += len(hist_prefix_ids)
                
                available = max_tokens - base_len - 10 # buffer
                
                if available > 0:
                    current_hist_len = 0
                    # Try to fit from newest (end of list) to oldest
                    # NOTE: Dataset logic was "truncate oldest items first", so we keep newest.
                    for item_ids in reversed(encoded_hist_items):
                        # item + space separator
                        item_cost = len(item_ids) + len(space_ids)
                        if current_hist_len + item_cost <= available:
                            selected_items_ids.insert(0, item_ids)
                            current_hist_len += item_cost
                        else:
                            break
            
            # Assemble History Section if we have items
            if selected_items_ids:
                hist_section_ids.extend(hist_prefix_ids)
                for i, item_ids in enumerate(selected_items_ids):
                    # Join items with space: "item1 item2"
                    if i > 0: # Add space before items except the first (after prefix? prefix has space? "Sequence: ")
                        # "User History Sequence: " already ends with space usually, but let's follow string logic
                        # String logic: " ".join([h.strip() ...])
                        # Prefix: "User History Sequence: " + joined_hist
                        # So no extra space needed after prefix, but needed between items.
                        hist_section_ids.extend(space_ids)
                    hist_section_ids.extend(item_ids)

        # 4. Final Assembly
        # Structure: [user_bos] [inst] [space] [hist] [space] [query] [user_eos] [assistant_bos]
        
        content_parts_ids = []
        
        # Part 1: Instruction
        if inst_ids:
            content_parts_ids.append(inst_ids)
            
        # Part 2: History
        if hist_section_ids:
            content_parts_ids.append(hist_section_ids)
            
        # Part 3: Query
        content_parts_ids.append(full_query_ids)
        
        # Join parts with space
        final_content_ids = []
        for i, part in enumerate(content_parts_ids):
            if i > 0:
                final_content_ids.extend(space_ids)
            final_content_ids.extend(part)
            
        # Full Prompt
        prompt_ids = user_bos_ids + final_content_ids + user_eos_ids + assistant_bos_ids
        
        # Decode for debugging/consistency
        prompt_str = self.tokenizer.decode(prompt_ids)
        
        # CRITICAL: Ensure prompt_str is not empty
        if not prompt_str or prompt_str.strip() == "":
            # Fallback: rebuild prompt string directly
            # This should ideally not be hit if tokenization is correct and query_text is not empty
            user_bos_str = self.base_bos + self.user_bos
            user_eos_str = self.user_eos
            assistant_bos_str = self.assistant_bos
            
            # Reconstruct content parts based on whether instruction and history are present
            content_parts_str = []
            if instruction:
                content_parts_str.append(instruction)
            # History is explicitly excluded when include_history is False
            content_parts_str.append(f"User Query: {query_text}")
            
            content_str = " ".join(content_parts_str)
            prompt_str = f"{user_bos_str}{content_str}{user_eos_str}{assistant_bos_str}"
            
            # Re-encode to get prompt_ids for consistency, though it might not be strictly necessary
            # as this fallback is for the string representation.
            prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
            
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"[GapGRPODataset] _build_query_prompt_ids returned empty string for query: {repr(query_text)}. "
                f"Instruction: {repr(instruction)}, include_history: {include_history}. "
                f"Rebuilt prompt_str: {repr(prompt_str[:100])}..."
                f"Original prompt_ids length: {len(prompt_ids)}. "
                f"user_bos_ids len: {len(user_bos_ids)}, final_content_ids len: {len(final_content_ids)}, "
                f"user_eos_ids len: {len(user_eos_ids)}, assistant_bos_ids len: {len(assistant_bos_ids)}"
            )
        
        return prompt_ids, prompt_str

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        """
        row_dict = dict(self.dataframe[idx])
        
        # Extract fields
        query = row_dict.get("query", "")
        query_gen = row_dict.get("query_gen", "")
        hist_list = row_dict.get("hist_list", [])
        pos_list = row_dict.get("pos", [])
        neg_list = row_dict.get("neg", [])
        
        # Sample one positive and (train_group_size - 1) negatives
        pos_idx = random.randrange(len(pos_list)) if pos_list else 0
        pos = pos_list[pos_idx] if pos_list else ""
        
        num_negs = max(0, self.train_group_size - 1)
        if len(neg_list) >= num_negs:
            chosen_neg_indices = random.sample(range(len(neg_list)), num_negs)
            negs = [neg_list[i] for i in chosen_neg_indices]
        else:
            # Sample with replacement if not enough negatives
            negs = [neg_list[random.randrange(len(neg_list))] for _ in range(num_negs)] if neg_list else []
        
        # Build query prompt WITH history (for generation/policy)
        # Use ID-based construction to ensure special tokens are correct
        input_ids_list, query_prompt = self._build_query_prompt_ids(
            query, hist_list, include_history=True, max_tokens=self.max_prompt_length
        )
        
        # Build query prompt WITHOUT history (for embedding when mask_hist=True)
        _, query_prompt_no_hist = self._build_query_prompt_ids(query, hist_list, include_history=False)
        
        # CRITICAL: Ensure query_prompt_no_hist is not empty
        if not query_prompt_no_hist or query_prompt_no_hist.strip() == "":
            # Fallback: rebuild with minimal prompt
            if isinstance(query, (list, tuple)) and len(query) >= 2:
                instruction = str(query[0]).strip()
                query_text = str(query[1]).strip()
            else:
                instruction = ""
                query_text = str(query).strip()
            
            if not query_text:
                print(f"[GapGRPODataset] Sample {idx}: query_text is empty")
                query_text = ""  # Should not happen if data is correct
            
            # Build minimal prompt: base_bos + user_bos + instruction + query + user_eos + assistant_bos
            content_parts = []
            if instruction:
                content_parts.append(instruction)
            content_parts.append(f"User Query: {query_text}")
            content = " ".join(content_parts)
            query_prompt_no_hist = f"{self.base_bos}{self.user_bos}{content}{self.user_eos}{self.assistant_bos}"
            
            # Log warning if we had to rebuild
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"[GapGRPODataset] Sample {idx}: query_prompt_no_hist was empty, rebuilt: {query_prompt_no_hist[:100]}...")
        
        # Convert to tensor for input_ids (which NEEDS padding for batching in Actor)
        input_ids = torch.tensor(input_ids_list, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        
        # Safety check: if still too long
        if input_ids.shape[1] > self.max_prompt_length:
            input_ids = input_ids[:, -self.max_prompt_length:]
            attention_mask = attention_mask[:, -self.max_prompt_length:]
        
        # Pad input_ids to max_prompt_length for consistent batch sizes (Required for Actor forward pass)
        seq_len = input_ids.shape[1]
        if seq_len < self.max_prompt_length:
            pad_len = self.max_prompt_length - seq_len
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids = torch.cat([
                torch.full((1, pad_len), pad_token_id, dtype=input_ids.dtype),
                input_ids
            ], dim=1)
            attention_mask = torch.cat([
                torch.zeros((1, pad_len), dtype=attention_mask.dtype),
                attention_mask
            ], dim=1)
        
        # Compute position ids
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # Build passage texts for later embedding computation
        pos_text = self._build_passage_text(pos) if pos else ""
        neg_texts = [self._build_passage_text(n) for n in negs] if negs else []
        
        # Handle raw_prompt_ids - THIS IS FOR vLLM ROLLOUT
        # CRITICAL FIX: raw_prompt_ids must NOT have padding! 
        # vLLM expects pure prompt tokens. Padding causes position ID shift and garbage output.
        raw_prompt_ids = list(input_ids_list)
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
        
        # DO NOT PAD raw_prompt_ids!
        
        # Prepare result dict
        result = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            # 🔑 CRITICAL: Put these at top level so Actor can select them for InfoNCE
            "pos_text": pos_text,
            "neg_texts": neg_texts,
            "query_prompt_no_hist": query_prompt_no_hist,
            "data_source": "gap_grpo",
            "reward_model": {
                "ground_truth": {
                    "query": query,
                    "query_gen": query_gen,
                    "pos": pos,
                    "neg": negs,
                    "pos_text": pos_text,
                    "neg_texts": neg_texts,
                    "query_prompt_no_hist": query_prompt_no_hist,
                }
            },
            "extra_info": {
                "query_prompt_no_hist": query_prompt_no_hist,
                "pos_text": pos_text,
            }
        }
        
        return result


def collate_fn(data_list: List[Dict]) -> Dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.
    
    Extended from verl's default collate_fn to handle GAP-specific fields.
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    
    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)
    
    # Stack tensors
    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)
    
    # Convert non-tensors to numpy arrays
    for key, val in non_tensors.items():
        # DIAGNOSTIC: Check extra_info before converting to numpy array
        if key == "extra_info":
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    qpnh = item.get("query_prompt_no_hist", "")
                    if not qpnh or (isinstance(qpnh, str) and qpnh.strip() == ""):
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error(
                            f"[GapGRPODataset.collate_fn] ERROR: Sample {i} has empty query_prompt_no_hist! "
                            f"extra_info keys: {list(item.keys())}, "
                            f"query: {repr(item.get('query', 'MISSING'))}, "
                            f"query_prompt: {repr(item.get('query_prompt', 'MISSING')[:100]) if item.get('query_prompt') else 'MISSING'}"
                        )
        # Use np.array instead of np.fromiter for better handling of dict/list objects
        non_tensors[key] = np.array(val, dtype=object)
    
    # 🔑 CRITICAL: Pre-create placeholder for retrieval_scores so that it survives union_numpy_dict
    # If we don't create this key here, the key added by rollout worker will be dropped
    # when merging non_tensor_batch across ranks, and RewardManager will see None.
    # Ensure retrieval_scores is always at least 1D array to prevent IndexError in chunk()
    if "retrieval_scores" not in non_tensors:
        batch_size = len(data_list)
        if batch_size > 0:
            non_tensors["retrieval_scores"] = np.array([None] * batch_size, dtype=object)
        else:
            # Create empty 1D array instead of 0D scalar
            non_tensors["retrieval_scores"] = np.array([], dtype=object)
    
    return {**tensors, **non_tensors}
