from dataclasses import dataclass
import logging
import random
from typing import Iterator, List, Tuple, Union, Dict, Any

import datasets
import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from .arguments import DataArguments

logger = logging.getLogger(__name__)


class MultiFileDataset(torch.utils.data.Dataset):
    """
    Dataset that loads multiple jsonl files.
    
    Expected jsonl schema:
    - `query`: either string, or `[embed_instruct, text]`
    - `query_gen`: generative supervision for query (string or [instruct, target])
    - `hist_list`: optional list of user history items (e.g., previously ordered products)
    - `pos`: list of strings, or list of `[embed_instruct, text]`
    - `neg`: list of strings, or list of `[embed_instruct, text]`
    - `batch_size`: optional per-file batch size
    - `train_group_size`: optional per-file train group size
    """
    def __init__(self, files: List[str], args: DataArguments, tokenizer: PreTrainedTokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.files = files
        # Defaults from args (used when file-level metadata is missing)
        self.default_train_group_size: int = args.train_group_size

        self.datasets: List[datasets.Dataset] = []
        self.file_batch_sizes: List[int] = []
        self.file_train_group_sizes: List[int] = []
        self.file_lengths: List[int] = []
        self.file_offsets: List[int] = []

        offset = 0
        for file_path in files:
            hf_dataset = datasets.load_dataset('json', data_files=file_path, split='train')
            if len(hf_dataset) > args.max_example_num_per_dataset:
                hf_dataset = hf_dataset.select(random.sample(list(range(len(hf_dataset))), args.max_example_num_per_dataset))
            self.datasets.append(hf_dataset)
            self.file_lengths.append(len(hf_dataset))
            self.file_offsets.append(offset)
            offset += len(hf_dataset)
            file_batch_size_value = hf_dataset['batch_size'][0] if 'batch_size' in hf_dataset.column_names else None
            # Leave -1 to signal sampler to use global effective batch size when missing
            self.file_batch_sizes.append(int(file_batch_size_value) if isinstance(file_batch_size_value, int) else -1)
            file_train_group_size_value = hf_dataset['train_group_size'][0] if 'train_group_size' in hf_dataset.column_names else None
            self.file_train_group_sizes.append(int(file_train_group_size_value) if isinstance(file_train_group_size_value, int) else int(self.default_train_group_size))

        self.concat = datasets.concatenate_datasets(self.datasets) if len(self.datasets) > 1 else self.datasets[0]
        self.total_len = len(self.concat)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
        data = self.concat[item]
        meta = {"file_index": self._file_index_from_global(item)}
        query = data['query']
        # query_gen: generative target for query side
        query_gen = data.get('query_gen', '')
        # hist_list: user history sequence (e.g., previously ordered products)
        hist_list = data.get('hist_list', None)

        passages = []
        pos_ids = random.randrange(len(data['pos']))
        passages.append(data['pos'][pos_ids])
        # determine negatives count by train_group_size (default 2 -> 1 negative)
        fidx = meta["file_index"]
        effective_train_group_size = self.file_train_group_sizes[fidx] if (fidx is not None and fidx < len(self.file_train_group_sizes)) else int(self.default_train_group_size)
        if not isinstance(effective_train_group_size, int) or effective_train_group_size <= 0:
            effective_train_group_size = int(self.default_train_group_size)
        num_negs = max(0, effective_train_group_size - 1)
        if num_negs > 0:
            negs = data.get('neg', [])
            if len(negs) >= num_negs:
                chosen_indices = random.sample(range(len(negs)), num_negs)
                passages.extend([negs[i] for i in chosen_indices])
            elif len(negs) > 0:
                # sample with replacement when not enough negatives
                for _ in range(num_negs):
                    neg_ids = random.randrange(len(negs))
                    passages.append(negs[neg_ids])

        return query, passages, query_gen, hist_list, meta

    def _file_index_from_global(self, global_idx: int) -> int:
        for i in range(len(self.file_offsets)):
            start = self.file_offsets[i]
            end = start + self.file_lengths[i]
            if start <= global_idx < end:
                return i
        return len(self.file_offsets) - 1


@dataclass
class QueryGenCollator(DataCollatorWithPadding):
    """
    Collator for query-side generation task.
    
    Query format: base_bos + user_bos + [instruct + content] + user_eos + assistant_bos + <think> gen_text </think> <emb> + assistant_eos
    Passage format: base_bos + user_bos + [instruct + content] + user_eos + assistant_bos + <emb> + assistant_eos
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    generative_max_len: int = 128
    base_bos: str = ""
    user_bos: str = ""
    user_eos: str = ""
    assistant_bos: str = ""
    assistant_eos: str = ""
    sub_batch_size: int = -1
    _first_batch_printed: bool = False  # Class variable to track if first batch was printed

    def __call__(self, features):
        queries = []
        passages = []
        # Store parts separately for manual label computation and position tracking (query side)
        # Each entry: (prefix_part, hist_part, query_part, gen_part)
        query_parts_list = []
        think_open = "<think>"
        think_close = "</think>"
        emb_tok = "<emb>"
        
        for f in features:
            q, p_list, q_gen, hist_list = f[0], f[1], f[2], f[3]
            
            # === Query: user_bos + instruction + [User History Sequence: hist...] [User Query:] content + user_eos + assistant_bos + [<think> gen_text </think>] <emb> + assistant_eos ===
            if q is not None:
                if isinstance(q, (tuple, list)) and len(q) >= 2:
                    instr = str(q[0]).strip()
                    content = str(q[1]).strip()
                else:
                    instr = ""
                    content = str(q).strip()
                
                # Build parts separately for position tracking
                has_history = hist_list and isinstance(hist_list, list) and len(hist_list) > 0
                
                if has_history:
                    # With history: prefix + hist_with_markers + query_suffix
                    # hist_with_markers includes "User History Sequence: " prefix for proper masking
                    hist_str = " ".join([str(h).strip() for h in hist_list])
                    prefix_part = self.base_bos + self.user_bos + instr
                    hist_part = "User History Sequence: " + hist_str  # Include prefix for complete masking
                    query_suffix_part = " User Query: " + content + self.user_eos + self.assistant_bos
                else:
                    # No history: prefix + query
                    prefix_part = self.base_bos + self.user_bos + instr
                    hist_part = ""
                    query_suffix_part = content + self.user_eos + self.assistant_bos
                
                # Handle query generation
                if q_gen:
                    gen_text = str(q_gen).strip() if isinstance(q_gen, str) else str(q_gen[1]).strip() if isinstance(q_gen, (tuple, list)) and len(q_gen) >= 2 else str(q_gen)
                    gen_part = think_open + gen_text + think_close + emb_tok
                else:
                    gen_part = emb_tok
                
                query_parts_list.append((prefix_part, hist_part, query_suffix_part, gen_part))
                q_text = prefix_part + hist_part + query_suffix_part + gen_part + self.assistant_eos
                queries.append(q_text)

            # === Passage: user_bos + instruction content + user_eos + assistant_bos + <emb> + assistant_eos ===
            if p_list is not None:
                for p in p_list:
                    if isinstance(p, (tuple, list)) and len(p) >= 2:
                        dinstr = str(p[0]).strip()
                        doc = str(p[1]).strip()
                    else:
                        dinstr = ""
                        doc = str(p)
                    
                    p_text = self.base_bos + self.user_bos + (dinstr + doc if dinstr else doc) + self.user_eos + self.assistant_bos + emb_tok + self.assistant_eos
                    passages.append(p_text)

        batch: Dict[str, Any] = {}
        
        # Helper function to process queries (with generation task)
        def _process_queries(queries_list, query_parts_list_subset):
            if not queries_list:
                return None
            
            # Tokenize each part separately
            prefix_parts = [p[0] for p in query_parts_list_subset]
            hist_parts = [p[1] for p in query_parts_list_subset]
            query_suffix_parts = [p[2] for p in query_parts_list_subset]
            gen_parts = [p[3] for p in query_parts_list_subset]
            
            prefix_toks = self.tokenizer(prefix_parts, padding=False, truncation=False, add_special_tokens=False)
            # For empty hist_parts, tokenizer returns empty list
            hist_toks = self.tokenizer(hist_parts, padding=False, truncation=False, add_special_tokens=False)
            query_suffix_toks = self.tokenizer(query_suffix_parts, padding=False, truncation=False, add_special_tokens=False)
            gen_toks = self.tokenizer(gen_parts, padding=False, truncation=False, add_special_tokens=False)
            eos_toks = self.tokenizer([self.assistant_eos] * len(queries_list), padding=False, truncation=False, add_special_tokens=False)
            
            # Get token IDs for position finding
            emb_id = self.tokenizer.convert_tokens_to_ids(emb_tok)
            t_open_id = self.tokenizer.convert_tokens_to_ids(think_open)
            t_close_id = self.tokenizer.convert_tokens_to_ids(think_close)
            
            # Build combined sequences and labels
            all_input_ids = []
            all_labels = []
            embed_pos = []
            gen_start = []
            gen_end = []
            hist_start = []
            hist_end = []
            
            for i in range(len(queries_list)):
                prefix_ids = prefix_toks["input_ids"][i]
                hist_ids = hist_toks["input_ids"][i]
                query_suffix_ids = query_suffix_toks["input_ids"][i]
                gen_ids = gen_toks["input_ids"][i]
                eos_ids = eos_toks["input_ids"][i]
                
                # Concatenate: prefix + hist + query_suffix + gen + eos
                combined_ids = prefix_ids + hist_ids + query_suffix_ids + gen_ids + eos_ids
                
                # Calculate lengths for position tracking
                prefix_len = len(prefix_ids)
                hist_len = len(hist_ids)
                query_suffix_len = len(query_suffix_ids)
                gen_len = len(gen_ids)
                input_len = prefix_len + hist_len + query_suffix_len  # Total input before gen
                
                # Truncate if needed (prioritize keeping gen part, then query, then hist)
                max_len = self.query_max_len
                if len(combined_ids) > max_len:
                    # Calculate minimum required length (query_suffix + gen + some eos)
                    min_required = query_suffix_len + gen_len
                    if min_required > max_len:
                        # Truncate gen
                        gen_len = max(0, max_len - query_suffix_len)
                        combined_ids = query_suffix_ids[:max_len - gen_len] + gen_ids[:gen_len]
                        prefix_len = 0
                        hist_len = 0
                        query_suffix_len = len(combined_ids) - gen_len
                        input_len = query_suffix_len
                    else:
                        # Try to keep as much as possible
                        available_for_prefix_hist = max_len - min_required - len(eos_ids)
                        if available_for_prefix_hist <= 0:
                            # Only keep query_suffix + gen + maybe some eos
                            combined_ids = query_suffix_ids + gen_ids + eos_ids[:max(0, max_len - min_required)]
                            prefix_len = 0
                            hist_len = 0
                            input_len = query_suffix_len
                        elif available_for_prefix_hist >= prefix_len + hist_len:
                            # Keep all prefix and hist
                            combined_ids = combined_ids[:max_len]
                        elif available_for_prefix_hist >= prefix_len:
                            # Keep all prefix, truncate hist
                            new_hist_len = available_for_prefix_hist - prefix_len
                            combined_ids = prefix_ids + hist_ids[:new_hist_len] + query_suffix_ids + gen_ids + eos_ids[:max(0, max_len - prefix_len - new_hist_len - query_suffix_len - gen_len)]
                            hist_len = new_hist_len
                            input_len = prefix_len + hist_len + query_suffix_len
                        else:
                            # Truncate prefix too
                            new_prefix_len = available_for_prefix_hist
                            combined_ids = prefix_ids[:new_prefix_len] + query_suffix_ids + gen_ids + eos_ids[:max(0, max_len - new_prefix_len - query_suffix_len - gen_len)]
                            prefix_len = new_prefix_len
                            hist_len = 0
                            input_len = prefix_len + query_suffix_len
                
                # Record hist position (if no hist, set to -1)
                if hist_len > 0:
                    h_start = prefix_len
                    h_end = prefix_len + hist_len - 1
                else:
                    h_start = -1
                    h_end = -1
                hist_start.append(h_start)
                hist_end.append(h_end)
                
                # Check if there's actual generation content (not just <emb>)
                has_think_content = t_open_id in gen_ids
                
                # Build labels: input (-100) + gen (trainable only if has think content) + eos (trainable)
                # 注意：eos_ids 也需要训练，让模型学会在 <emb> 之后输出 <|im_end|> 并停止
                if has_think_content:
                    # 计算截断后实际保留的 eos 长度
                    # combined_ids 的结构是: input + gen + eos (可能被截断)
                    eos_len = len(combined_ids) - input_len - gen_len
                    # input 部分 mask，gen 部分可训练，eos 部分也可训练
                    combined_labels = [-100] * input_len + list(gen_ids[:gen_len]) + list(eos_ids[:eos_len])
                    # 确保长度匹配（理论上应该已经匹配，但保险起见）
                    if len(combined_labels) < len(combined_ids):
                        combined_labels += [-100] * (len(combined_ids) - len(combined_labels))
                    elif len(combined_labels) > len(combined_ids):
                        combined_labels = combined_labels[:len(combined_ids)]
                else:
                    # No generation task, all labels are -100
                    combined_labels = [-100] * len(combined_ids)
                
                all_input_ids.append(combined_ids)
                all_labels.append(combined_labels)
                
                # Find positions in combined sequence
                ids_list = combined_ids
                epos = max([j for j, tid in enumerate(ids_list) if tid == emb_id], default=-1)
                if epos == -1:
                    # force last non-pad token to <emb>
                    last_idx = len(ids_list) - 1
                    if last_idx >= 0:
                        ids_list[last_idx] = emb_id
                        combined_labels[last_idx] = -100
                    epos = last_idx
                embed_pos.append(epos)
                
                try:
                    s = ids_list.index(t_open_id)
                except ValueError:
                    s = -1
                try:
                    e = ids_list.index(t_close_id, s+1)
                except ValueError:
                    e = -1

                if e == -1 and s != -1:
                    e = epos - 1

                gen_start.append(s)
                gen_end.append(e)
            
            # Batch pad efficiently
            max_len = max(len(ids) for ids in all_input_ids) if all_input_ids else 0
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            padded_input_ids = torch.tensor([
                ids + [pad_id] * (max_len - len(ids)) for ids in all_input_ids
            ], dtype=torch.long)
            padded_labels = torch.tensor([
                labels + [-100] * (max_len - len(labels)) for labels in all_labels
            ], dtype=torch.long)
            padded_attention_mask = torch.tensor([
                [1] * len(ids) + [0] * (max_len - len(ids)) for ids in all_input_ids
            ], dtype=torch.long)
            
            # Debug: Print first batch info
            if not QueryGenCollator._first_batch_printed and len(all_input_ids) > 0:
                logger.info("=" * 80)
                logger.info("[DEBUG] First batch - Sample 0:")
                logger.info("=" * 80)
                
                # Print original input_ids (before padding)
                first_input_ids = all_input_ids[0]
                first_labels = all_labels[0]
                logger.info(f"Input IDs (length={len(first_input_ids)}): {first_input_ids}")
                logger.info(f"Label IDs (length={len(first_labels)}): {first_labels}")
                
                # Decode to show text
                try:
                    decoded_input = self.tokenizer.decode(first_input_ids, skip_special_tokens=False)
                    logger.info(f"Decoded Input Text:\n{decoded_input}")
                except Exception as e:
                    logger.warning(f"Failed to decode input: {e}")
                
                # Show which positions are trainable (not -100)
                trainable_positions = [i for i, label in enumerate(first_labels) if label != -100]
                logger.info(f"Trainable positions (label != -100): {trainable_positions}")
                if trainable_positions:
                    trainable_ids = [first_input_ids[i] for i in trainable_positions]
                    trainable_labels = [first_labels[i] for i in trainable_positions]
                    logger.info(f"Trainable Input IDs: {trainable_ids}")
                    logger.info(f"Trainable Label IDs: {trainable_labels}")
                    try:
                        decoded_trainable = self.tokenizer.decode(trainable_ids, skip_special_tokens=False)
                        logger.info(f"Decoded Trainable Text:\n{decoded_trainable}")
                    except Exception as e:
                        logger.warning(f"Failed to decode trainable text: {e}")
                else:
                    logger.warning("No trainable positions found (all labels are -100)!")
                
                # Show special token positions
                think_open_id = self.tokenizer.convert_tokens_to_ids("<think>")
                think_close_id = self.tokenizer.convert_tokens_to_ids("</think>")
                emb_id = self.tokenizer.convert_tokens_to_ids("<emb>")
                
                think_open_pos = [i for i, tid in enumerate(first_input_ids) if tid == think_open_id]
                think_close_pos = [i for i, tid in enumerate(first_input_ids) if tid == think_close_id]
                emb_pos_list = [i for i, tid in enumerate(first_input_ids) if tid == emb_id]
                
                logger.info(f"<think> positions: {think_open_pos}")
                logger.info(f"</think> positions: {think_close_pos}")
                logger.info(f"<emb> positions: {emb_pos_list}")
                
                logger.info("=" * 80)
                QueryGenCollator._first_batch_printed = True
            
            return {
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_mask,
                "labels": padded_labels,
                "embed_pos": torch.tensor(embed_pos, dtype=torch.long),
                "gen_start": torch.tensor(gen_start, dtype=torch.long),
                "gen_end": torch.tensor(gen_end, dtype=torch.long),
                "hist_start": torch.tensor(hist_start, dtype=torch.long),
                "hist_end": torch.tensor(hist_end, dtype=torch.long),
            }
        
        # Helper function to process passages (embedding only)
        def _process_passages(passages_list):
            if not passages_list:
                return None
            passage_tok = self.tokenizer(passages_list, padding=True, truncation=True, max_length=self.passage_max_len, return_tensors="pt", add_special_tokens=False)
            # find <emb> positions per passage
            emb_id = self.tokenizer.convert_tokens_to_ids(emb_tok)
            p_embed_pos = []
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            for batch_idx, ids in enumerate(passage_tok["input_ids"]):
                ids_list = ids.tolist()
                ppos = max([i for i, tid in enumerate(ids_list) if tid == emb_id], default=-1)
                if ppos == -1:
                    # find last non-pad position and force it to <emb>
                    last_idx = len(ids_list) - 1
                    while last_idx > 0 and ids_list[last_idx] == pad_id:
                        last_idx -= 1
                    if last_idx >= 0:
                        passage_tok["input_ids"][batch_idx, last_idx] = emb_id
                    ppos = last_idx
                p_embed_pos.append(ppos)
            passage_tok["embed_pos"] = torch.tensor(p_embed_pos, dtype=torch.long)
            return passage_tok
        
        # Handle sub-batch processing if sub_batch_size > 0
        if self.sub_batch_size is not None and self.sub_batch_size > 0:
            # Process queries in sub-batches
            if queries:
                query_batches = []
                for i in range(0, len(queries), self.sub_batch_size):
                    end_idx = min(i + self.sub_batch_size, len(queries))
                    query_batch = _process_queries(
                        queries[i:end_idx],
                        query_parts_list[i:end_idx]
                    )
                    if query_batch is not None:
                        query_batches.append(query_batch)
                if query_batches:
                    batch["query"] = query_batches if len(query_batches) > 1 else query_batches[0]
            else:
                batch["query"] = None
            
            # Process passages in sub-batches
            if passages:
                passage_batches = []
                for i in range(0, len(passages), self.sub_batch_size):
                    end_idx = min(i + self.sub_batch_size, len(passages))
                    passage_batch = _process_passages(passages[i:end_idx])
                    if passage_batch is not None:
                        passage_batches.append(passage_batch)
                if passage_batches:
                    batch["passage"] = passage_batches if len(passage_batches) > 1 else passage_batches[0]
            else:
                batch["passage"] = None
        else:
            # Process entire batch at once (original behavior)
            if queries:
                query_tok = _process_queries(queries, query_parts_list)
                if query_tok is not None:
                    batch["query"] = query_tok
            if passages:
                passage_tok = _process_passages(passages)
                if passage_tok is not None:
                    batch["passage"] = passage_tok

        return batch


@dataclass
class SameFileBatchSampler(torch.utils.data.sampler.Sampler[int]):
    dataset: MultiFileDataset
    global_batch_size: int
    seed: int = 42

    def __iter__(self) -> Iterator[int]:
        random_generator = torch.Generator()
        random_generator.manual_seed(int(self.seed))
        per_file_indices: List[List[int]] = []
        for file_index, file_length in enumerate(self.dataset.file_lengths):
            start_offset = self.dataset.file_offsets[file_index]
            indices = list(range(start_offset, start_offset + file_length))
            permuted = torch.randperm(len(indices), generator=random_generator).tolist()
            indices = [indices[i] for i in permuted]
            per_file_indices.append(indices)
        batch_blocks: List[List[int]] = []
        for file_index, indices in enumerate(per_file_indices):
            effective_batch_size = self.dataset.file_batch_sizes[file_index]
            # prefer dataset-level default if provided, else fall back to global
            if not (isinstance(effective_batch_size, int) and effective_batch_size > 0):
                dataset_default = getattr(self.dataset, 'default_file_batch_size', -1)
                effective_batch_size = dataset_default if (isinstance(dataset_default, int) and dataset_default > 0) else self.global_batch_size
            chunks = [indices[i:i+effective_batch_size] for i in range(0, len(indices), effective_batch_size)]
            chunks = [c for c in chunks if len(c) == effective_batch_size]
            batch_blocks.extend(chunks)
        shuffled_block_order = torch.randperm(len(batch_blocks), generator=random_generator).tolist()
        flattened_indices = [i for block in (batch_blocks[i] for i in shuffled_block_order) for i in block]
        for i in flattened_indices:
            yield i

    def __len__(self) -> int:
        total_count = 0
        for file_index, file_length in enumerate(self.dataset.file_lengths):
            effective_batch_size = self.dataset.file_batch_sizes[file_index]
            if not (isinstance(effective_batch_size, int) and effective_batch_size > 0):
                dataset_default = getattr(self.dataset, 'default_file_batch_size', -1)
                effective_batch_size = dataset_default if (isinstance(dataset_default, int) and dataset_default > 0) else self.global_batch_size
            total_count += (file_length // effective_batch_size) * effective_batch_size
        return total_count

