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
GAP-GRPO Actor

Extended actor that:
1. Extracts embeddings at <emb> token position from hidden states
2. Computes InfoNCE loss for contrastive learning (optional)
3. Combines policy loss with InfoNCE loss
4. Supports mask_hist: use query_prompt_no_hist + response for embedding (no history info)

Reference: gap-r1/train/training_qwen3_query/model.py
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

logger = logging.getLogger(__name__)

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.logger import print_rank_0, print_with_rank
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["GapGRPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DistributedContrastiveLoss:
    """
    InfoNCE loss for contrastive learning.
    Reference: gap-r1/train/training_qwen3_query/model.py
    """
    
    def __init__(self, temperature: float = 0.02, negatives_cross_device: bool = True):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device
        print_rank_0(f"negatives_cross_device: {self.negatives_cross_device}")
        if self.negatives_cross_device and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.negatives_cross_device = False
            self.rank = 0
            self.world_size = 1
    
    def __call__(self, q_reps: torch.Tensor, p_reps: torch.Tensor, group_size: int = 1):
        """
        Compute InfoNCE loss.
        
        Args:
            q_reps: Query embeddings [batch_size, hidden_dim]
            p_reps: Passage embeddings [batch_size * group_size, hidden_dim]
            group_size: Number of passages per query (1 pos + (group_size-1) negs)
        
        Returns:
            InfoNCE loss
        """
        if self.negatives_cross_device:
            # Gather both queries and passages across devices
            cross_q_reps = self._gather(q_reps)
            cross_p_reps = self._gather(p_reps)
            
            # Compute scores using global tensors
            scores = torch.matmul(cross_q_reps, cross_p_reps.transpose(0, 1)) / self.temperature
            
            # Target: each query i's positive is at column i * group_size
            cross_idxs = torch.arange(cross_q_reps.size(0), device=scores.device, dtype=torch.long)
            targets = cross_idxs * group_size
            
            return self.cross_entropy(scores, targets)
        else:
            # Local in-batch negatives only
            batch_size = q_reps.size(0)
            scores = torch.matmul(q_reps, p_reps.transpose(0, 1)) / self.temperature
            targets = torch.arange(batch_size, device=scores.device, dtype=torch.long) * group_size
            return self.cross_entropy(scores, targets)
    
    def _gather(self, t: torch.Tensor) -> torch.Tensor:
        """Gather tensors across all devices."""
        if t is None:
            return None
        t = t.contiguous()
        
        # Check tensor sizes before gather to ensure consistency
        local_size = t.size(0)
        sizes = [torch.zeros(1, dtype=torch.long, device=t.device) for _ in range(self.world_size)]
        dist.all_gather(sizes, torch.tensor([local_size], dtype=torch.long, device=t.device))
        sizes_list = [s.item() for s in sizes]
        
        # Check if sizes are consistent
        if len(set(sizes_list)) > 1:
            print_rank_0(f"[GapActor] ⚠️ ERROR: Different ranks have different tensor sizes in _gather!")
            print_rank_0(f"  Rank {self.rank}: size={local_size}, All sizes: {sizes_list}")
            print_rank_0(f"  Tensor shape: {t.shape}")
            raise RuntimeError(
                f"Tensor size mismatch in _gather: rank {self.rank} has size {local_size}, "
                f"but other ranks have sizes {sizes_list}. This will cause dist.all_gather to timeout."
            )
        
        bufs = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(bufs, t)
        bufs[self.rank] = t
        return torch.cat(bufs, dim=0)


class GapGRPOActor(DataParallelPPOActor):
    """
    Extended PPO Actor for GAP-GRPO training.
    
    Adds:
    1. Embedding extraction at <emb> token position
    2. InfoNCE loss for contrastive learning (optional)
    3. mask_hist: When True, use query_prompt_no_hist + response for query embedding
       This prevents history information from leaking into the query embedding.
    
    Config (gap_config):
    - embedder_loss_weight: Weight for InfoNCE loss (0 to disable, >0 to enable)
    - infonce_temperature: Temperature for InfoNCE
    - emb_token: Token to extract embedding at (default: "<emb>")
    - normalized_embeddings: Whether to L2 normalize embeddings
    - mask_hist: Whether to exclude history from query embedding computation
    """
    
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        tokenizer = None,
        gap_config = None,  # Pass gap_config separately (from top-level config)
    ):
        # Debug: Log that GapGRPOActor is being initialized
        try:
            rank = dist.get_rank() if dist.is_initialized() else 0
        except:
            rank = 0
        
        
        super().__init__(config, actor_module, actor_optimizer)
        
        self.tokenizer = tokenizer
        
        # GAP-specific config - passed separately or extracted from config
        if gap_config is None:
            # Fallback: try to get from config (for backwards compatibility)
            if hasattr(config, 'gap_config'):
                gap_config = config.gap_config
            elif isinstance(config, dict):
                gap_config = config.get("gap_config", {})
            else:
                gap_config = {}
        
        # Helper to extract config values (handle dict, dataclass, or OmegaConf)
        def get_config_value(cfg, key, default):
            if cfg is None:
                return default
            if hasattr(cfg, key):
                return getattr(cfg, key)
            elif isinstance(cfg, dict):
                return cfg.get(key, default)
            elif hasattr(cfg, '__getitem__'):
                try:
                    return cfg[key]
                except (KeyError, TypeError):
                    return default
            return default
        
        # Loss weights for combined loss
        self.gap_config = gap_config
        self.embedder_loss_weight = get_config_value(gap_config, "embedder_loss_weight", 0.0)
        self.rl_loss_weight = get_config_value(gap_config, "rl_loss_weight", 1.0)
        self.infonce_temperature = get_config_value(gap_config, "infonce_temperature", 0.02)
        self.emb_token = get_config_value(gap_config, "emb_token", "<emb>")
        self.think_open = get_config_value(gap_config, "think_open", "<think>")
        self.think_close = get_config_value(gap_config, "think_close", "</think>")
        self.normalized_embeddings = get_config_value(gap_config, "normalized_embeddings", True)
        self.mask_hist = get_config_value(gap_config, "mask_hist", False)
        
        # New: o1-embedder best rewrite selection params
        self.use_best_rewrite_selection = get_config_value(gap_config, "use_best_rewrite_selection", True)
        # Try to get num_generations from gap_config first, then from rollout config
        self.num_generations = get_config_value(gap_config, "num_generations", None)
        if self.num_generations is None:
            # Fallback: try to get from config.actor_rollout_ref.rollout.n
            try:
                if hasattr(config, 'actor_rollout_ref') and hasattr(config.actor_rollout_ref, 'rollout'):
                    if hasattr(config.actor_rollout_ref.rollout, 'n'):
                        self.num_generations = config.actor_rollout_ref.rollout.n
                    elif isinstance(config.actor_rollout_ref.rollout, dict) and 'n' in config.actor_rollout_ref.rollout:
                        self.num_generations = config.actor_rollout_ref.rollout['n']
            except:
                pass
        if self.num_generations is None:
            self.num_generations = 4  # Default fallback
        print_rank_0(f"[GapGRPOActor] num_generations: {self.num_generations}")
        
        # Get emb token id
        self.emb_token_id = None
        if tokenizer is not None:
            try:
                self.emb_token_id = tokenizer.convert_tokens_to_ids(self.emb_token)
                if self.emb_token_id == tokenizer.unk_token_id:
                    print_rank_0(f"[GapGRPOActor] Warning: emb_token '{self.emb_token}' not in vocabulary")
                    self.emb_token_id = None
            except Exception as e:
                self.emb_token_id = None
        
        # Initialize InfoNCE loss
        if self.embedder_loss_weight > 0:
            self.infonce_loss_fn = DistributedContrastiveLoss(
                temperature=self.infonce_temperature,
                negatives_cross_device=True,
            )
            print_rank_0(f"[GapGRPOActor] InfoNCE enabled: weight={self.embedder_loss_weight}, "
                         f"temp={self.infonce_temperature}, mask_hist={self.mask_hist}")
        else:
            self.infonce_loss_fn = None
            print_rank_0("[GapGRPOActor] InfoNCE disabled (weight=0)")

        self.stage_logging_enabled = os.environ.get("GAP_STAGE_LOG", "1") not in {"0", "false", "False"}
        
        # Debug: Log initialization summary
        print_rank_0("=" * 80)
        print_rank_0(f"[GapGRPOActor] ===== INITIALIZATION COMPLETE =====")
        print_rank_0(f"[GapGRPOActor] embedder_loss_weight: {self.embedder_loss_weight}")
        print_rank_0(f"[GapGRPOActor] rl_loss_weight: {self.rl_loss_weight}")
        print_rank_0(f"[GapGRPOActor] emb_token: {self.emb_token}, emb_token_id: {self.emb_token_id}")
        print_rank_0(f"[GapGRPOActor] infonce_loss_fn is None: {self.infonce_loss_fn is None}")
        print_rank_0(f"[GapGRPOActor] mask_hist: {self.mask_hist}")
        print_rank_0(f"[GapGRPOActor] normalized_embeddings: {self.normalized_embeddings}")
        print_rank_0("=" * 80)

    def _log_stage(self, message: str):
        """Log stage transitions on rank 0 to avoid noisy logs."""
        # Disabled for cleaner logs
        pass
    
    def _find_emb_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Find the positions of <emb> token in input sequences.
        
        Args:
            input_ids: [batch_size, seq_len]
        
        Returns:
            positions: [batch_size] - position of last <emb> token in each sequence.
                       If not found, returns -1.
        """
        batch_size = input_ids.size(0)
        positions = torch.full((batch_size,), -1, dtype=torch.long, device=input_ids.device)
        
        # Vectorized: find all <emb> token positions
        # Shape: [batch_size, seq_len] - True where emb_token_id matches
        emb_mask = (input_ids == self.emb_token_id)
        
        # For each sequence, find the last True position
        for i in range(batch_size):
            if emb_mask[i].any():
                # Find all positions where emb_token_id appears
                pos_indices = emb_mask[i].nonzero(as_tuple=True)[0]
                if len(pos_indices) > 0:
                    positions[i] = pos_indices[-1]
        
        return positions
    
    def _extract_embeddings(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract embeddings at <emb> token positions.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            input_ids: [batch_size, seq_len]
        
        Returns:
            embeddings: [batch_size, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.size()
        
        emb_positions = self._find_emb_positions(input_ids)
        
        # Handle cases where <emb> is not found (pos == -1)
        # We fallback to the last token, but log a warning if needed
        # Create a mask for valid positions
        valid_mask = (emb_positions != -1)
        
        # For gathering, replace -1 with seq_len - 1 to avoid index error
        safe_positions = emb_positions.clone()
        safe_positions[~valid_mask] = seq_len - 1
        
        idx = safe_positions.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_dim)
        embeddings = torch.gather(hidden_states, dim=1, index=idx).squeeze(1)
        
        # Zero out invalid embeddings (optional, but safer)
        # or keep last token embedding as fallback
        # embeddings = embeddings * valid_mask.unsqueeze(-1)
        
        if self.normalized_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        
        return embeddings, valid_mask
    
    def _extract_response_text(self, input_ids: torch.Tensor, prompt_length: int) -> List[str]:
        """
        Extract the generated response text from input_ids.
        
        Args:
            input_ids: [batch_size, seq_len] - full sequence (prompt + response)
            prompt_length: Length of prompt tokens
        
        Returns:
            List of response strings
        """
        if self.tokenizer is None:
            return []
        
        batch_size = input_ids.size(0)
        responses = []
        for i in range(batch_size):
            # Get response tokens (after prompt)
            response_ids = input_ids[i, prompt_length:]
            # Decode
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            responses.append(response_text)
        
        return responses
    
    def _encode_for_embedding(
        self,
        texts: List[str],
        allow_grad: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Encode texts to get embeddings.
        
        For query texts without <emb> token:
        - If there's space, append <emb> token id at the end
        - Otherwise, replace the last token with <emb> token id
        
        Args:
            texts: List of texts (may or may not contain <emb>)
            allow_grad: If True, allows gradient flow (for InfoNCE training)
        
        Returns:
            embeddings: [num_texts, hidden_dim] or None if no texts
        """
        if not texts or self.tokenizer is None or self.emb_token_id is None:
            return None
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            add_special_tokens=False,
        )
        
        input_ids = encoded["input_ids"].to(get_device_id())
        attention_mask = encoded["attention_mask"].to(get_device_id())
        
        # Vectorized check and add <emb> token if missing
        batch_size = input_ids.size(0)
        max_seq_len = input_ids.size(1)
        
        # Find last valid position for each sequence
        # attention_mask is [batch_size, seq_len], find last 1 for each row
        last_valid_positions = (attention_mask * torch.arange(max_seq_len, device=input_ids.device).unsqueeze(0)).max(dim=1)[0]
        
        # Check if <emb> token exists in valid tokens for each sequence (vectorized)
        # Create a mask for valid tokens only
        pos_indices = torch.arange(max_seq_len, device=input_ids.device).unsqueeze(0)  # [1, max_seq_len]
        valid_mask = pos_indices <= last_valid_positions.unsqueeze(1)  # [batch_size, max_seq_len]
        
        # Check for <emb> token in valid positions
        emb_in_valid = ((input_ids == self.emb_token_id) & valid_mask).any(dim=1)  # [batch_size]
        
        # Find sequences that need <emb> token added
        needs_emb = ~emb_in_valid & (last_valid_positions >= 0)
        
        if needs_emb.any():
            # Process sequences that need <emb> token
            for i in needs_emb.nonzero(as_tuple=True)[0]:
                i = i.item()
                last_pos = last_valid_positions[i].item()
                
                if last_pos + 1 < max_seq_len:
                    # Append <emb> token id after last valid token
                    input_ids[i, last_pos + 1] = self.emb_token_id
                    attention_mask[i, last_pos + 1] = 1
                else:
                    # At max_length, replace last token with <emb>
                    input_ids[i, last_pos] = self.emb_token_id
        
        # Forward pass - WITH gradient for InfoNCE
        if allow_grad:
            with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = output.hidden_states[-1]
        else:
            with torch.no_grad():
                with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    last_hidden = output.hidden_states[-1]
        
        # Extract embeddings at <emb> positions
        embeddings, valid_mask = self._extract_embeddings(last_hidden, input_ids)
        
        # Now all sequences should have <emb> token (we added it if missing)
        # But still check valid_mask in case of edge cases
        if not valid_mask.all():
            print_rank_0(f"[GapGRPOActor] Warning: Failed to extract <emb> embeddings. "
                      f"Valid count: {valid_mask.sum().item()}/{valid_mask.size(0)}. Falling back to policy hidden states.")
            return None
            
        return embeddings
    
    def _forward_micro_batch_with_hidden(
        self,
        micro_batch: Dict,
        temperature: float,
        calculate_entropy: bool = False,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns hidden states for embedding extraction.
        
        Returns:
            entropy: [batch_size, response_len] or None
            log_probs: [batch_size, response_len]
            last_hidden_state: [batch_size, seq_len, hidden_dim]
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )
        
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)
            
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            
            logits = output.logits
            last_hidden_state = output.hidden_states[-1]
            
            logits = logits / temperature
            logits_for_logprob = logits[:, -response_length - 1 : -1, :]
            log_probs = logprobs_from_logits(logits_for_logprob, micro_batch["responses"])
            
            if calculate_entropy:
                if not self.config.entropy_checkpointing:
                    entropy = verl_F.entropy_from_logits(logits_for_logprob)
                else:
                    entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits_for_logprob)
            
            return entropy, log_probs, last_hidden_state
    
    def _compute_query_embeddings_for_infonce(
        self,
        model_inputs: Dict,
        last_hidden_state: torch.Tensor,
        retrieval_rewards: Optional[torch.Tensor] = None,
        force_use_original_query: bool = False,
    ) -> torch.Tensor:
        """
        Compute query embeddings for InfoNCE.
        
        Strategy:
        1. If force_use_original_query=True: always use original query (for deduplication mode)
        2. Elif retrieval_rewards provided and reward < 0: use original query (avoid training on bad rewrites)
        3. Otherwise: use query_prompt_no_hist + response (rewritten query)
        
        This ensures we only train InfoNCE on successful rewrites or deduplicated original queries.
        
        Args:
            model_inputs: Dict containing input data
            last_hidden_state: [batch_size, seq_len, hidden_dim]
            retrieval_rewards: [batch_size] optional tensor of retrieval rewards
            force_use_original_query: If True, always use original query (for best_selection=False mode)
        """
        input_ids = model_inputs["input_ids"]
        batch_size = input_ids.size(0)
        
        # 🔑 Priority 1: Use top-level query_prompt_no_hist if available (passed from dataset)
        query_prompts_no_hist = model_inputs.get("query_prompt_no_hist", None)
        
        # Priority 2: Try to get from extra_info
        extra_info = model_inputs.get("extra_info", None)
        if query_prompts_no_hist is None:
            if extra_info is not None:
                if isinstance(extra_info, (list, np.ndarray, torch.Tensor)):
                    if hasattr(extra_info, 'tolist'):
                        extra_info = extra_info.tolist()
                    query_prompts_no_hist = [
                        e.get("query_prompt_no_hist", "") if isinstance(e, dict) else ""
                        for e in extra_info
                    ]
        
        # Convert to list if it's a single string or numpy array
        if query_prompts_no_hist is not None:
            if isinstance(query_prompts_no_hist, (np.ndarray, torch.Tensor)):
                query_prompts_no_hist = query_prompts_no_hist.tolist()
            elif isinstance(query_prompts_no_hist, str):
                query_prompts_no_hist = [query_prompts_no_hist] * batch_size

        if not query_prompts_no_hist or not any(query_prompts_no_hist):
            # Fallback: extract from hidden states (less accurate but safe)
            emb, _ = self._extract_embeddings(last_hidden_state, input_ids)
            return emb
        
        # Get response texts
        responses = model_inputs.get("responses", None)
        if responses is None:
            emb, _ = self._extract_embeddings(last_hidden_state, input_ids)
            return emb
        
        # Calculate prompt length to extract response
        prompt_length = input_ids.size(1) - responses.size(1)
        response_texts = self._extract_response_text(input_ids, prompt_length)
        
        # 🔑 NEW: Conditional query construction based on retrieval_rewards or force flag
        use_original_query_mask = None
        query_texts_for_emb = None  # Initialize to avoid UnboundLocalError
        
        if force_use_original_query:
            # Force mode: all samples use original query (for deduplication when best_selection=False)
            # Build original query texts directly, don't depend on retrieval_rewards
            if extra_info is not None:
                # Normalize extra_info to list
                if isinstance(extra_info, np.ndarray):
                    extra_info = extra_info.tolist()
                elif not isinstance(extra_info, list):
                    extra_info = None
                
                if extra_info and len(extra_info) == batch_size:
                    # Verify query_prompts_no_hist and response_texts lengths match
                    if len(query_prompts_no_hist) == batch_size and len(response_texts) == batch_size:
                        # Build query texts with all original queries
                        query_texts_for_emb = []
                        for i in range(batch_size):
                            if isinstance(extra_info[i], dict):
                                original_query = extra_info[i].get("query", "")
                                # Extract original query text (handle list/tuple format)
                                if isinstance(original_query, (list, tuple)) and len(original_query) >= 2:
                                    original_query_text = str(original_query[1]).strip()
                                else:
                                    original_query_text = str(original_query).strip()
                                
                                if original_query_text:
                                    # Build: query_prompt_no_hist + <think> + original_query + </think> + <emb>
                                    query_text = (
                                        str(query_prompts_no_hist[i]) + 
                                        self.think_open + 
                                        original_query_text + 
                                        self.think_close + 
                                        " " + 
                                        self.emb_token
                                    )
                                    query_texts_for_emb.append(query_text)
                                else:
                                    # Fallback to rewritten query if original_query is empty
                                    query_texts_for_emb.append(str(query_prompts_no_hist[i]) + response_texts[i])
                            else:
                                # Fallback to rewritten query if extra_info format is wrong
                                query_texts_for_emb.append(str(query_prompts_no_hist[i]) + response_texts[i])
                    else:
                        # Length mismatch, fallback to rewritten queries
                        query_texts_for_emb = [
                            str(qp) + resp
                            for qp, resp in zip(query_prompts_no_hist, response_texts)
                        ]
                else:
                    # extra_info format wrong or missing, fallback to rewritten queries
                    query_texts_for_emb = [
                        str(qp) + resp
                        for qp, resp in zip(query_prompts_no_hist, response_texts)
                    ]
            else:
                # No extra_info, fallback to rewritten queries
                query_texts_for_emb = [
                    str(qp) + resp
                    for qp, resp in zip(query_prompts_no_hist, response_texts)
                ]
        elif retrieval_rewards is not None and len(retrieval_rewards) == batch_size:
            # Conditional mode: samples with negative retrieval rewards use original query
            use_original_query_mask = retrieval_rewards < 0
            
            # Only proceed if we have extra_info (needed for original query)
            if use_original_query_mask.any() and extra_info is not None:
                # Normalize extra_info to list
                if isinstance(extra_info, np.ndarray):
                    extra_info = extra_info.tolist()
                elif not isinstance(extra_info, list):
                    extra_info = None
                
                if extra_info and len(extra_info) == batch_size:
                    # Verify query_prompts_no_hist and response_texts lengths match
                    if len(query_prompts_no_hist) == batch_size and len(response_texts) == batch_size:
                        # Build query texts with conditional logic
                        query_texts_for_emb = []
                        for i in range(batch_size):
                            if use_original_query_mask[i]:
                                # Use original query for negative rewards
                                if isinstance(extra_info[i], dict):
                                    original_query = extra_info[i].get("query", "")
                                    # Extract original query text (handle list/tuple format)
                                    if isinstance(original_query, (list, tuple)) and len(original_query) >= 2:
                                        original_query_text = str(original_query[1]).strip()
                                    else:
                                        original_query_text = str(original_query).strip()
                                    
                                    if original_query_text:
                                        # Build: query_prompt_no_hist + <think> + original_query + </think> + <emb>
                                        query_text = (
                                            str(query_prompts_no_hist[i]) + 
                                            self.think_open + 
                                            original_query_text + 
                                            self.think_close + 
                                            " " + 
                                            self.emb_token
                                        )
                                        query_texts_for_emb.append(query_text)
                                    else:
                                        # Fallback to rewritten query if original_query is empty
                                        query_texts_for_emb.append(str(query_prompts_no_hist[i]) + response_texts[i])
                                else:
                                    # Fallback to rewritten query if extra_info format is wrong
                                    query_texts_for_emb.append(str(query_prompts_no_hist[i]) + response_texts[i])
                            else:
                                # Use rewritten query for positive/zero rewards
                                query_texts_for_emb.append(str(query_prompts_no_hist[i]) + response_texts[i])
                    else:
                        # Length mismatch, fallback to rewritten queries
                        query_texts_for_emb = [
                            str(qp) + resp
                            for qp, resp in zip(query_prompts_no_hist, response_texts)
                        ]
                else:
                    # extra_info format wrong, use all rewritten queries
                    query_texts_for_emb = [
                        str(qp) + resp
                        for qp, resp in zip(query_prompts_no_hist, response_texts)
                    ]
            else:
                # No negative rewards or no extra_info, use all rewritten queries
                query_texts_for_emb = [
                    str(qp) + resp
                    for qp, resp in zip(query_prompts_no_hist, response_texts)
                ]
        else:
            # No retrieval_rewards provided, use all rewritten queries (original behavior)
            query_texts_for_emb = [
                str(qp) + resp
                for qp, resp in zip(query_prompts_no_hist, response_texts)
            ]
        
        # Ensure query_texts_for_emb is defined (safety check)
        if query_texts_for_emb is None:
            # Final fallback: use all rewritten queries
            query_texts_for_emb = [
                str(qp) + resp
                for qp, resp in zip(query_prompts_no_hist, response_texts)
            ]
        
        # Forward through model to get embeddings
        query_embeddings = self._encode_for_embedding(query_texts_for_emb, allow_grad=True)
        
        if query_embeddings is None:
            emb, _ = self._extract_embeddings(last_hidden_state, input_ids)
            return emb
        
        return query_embeddings
    
    @GPUMemoryLogger(role="gap actor", logger=logger)
    def update_policy(self, data: DataProto):
        """
        Update policy with optional InfoNCE loss.
        
        Extended from base class to:
        1. Extract embeddings from hidden states (with optional mask_hist)
        2. Compute InfoNCE loss if enabled (both query and passage allow gradient)
        3. Combine policy loss with InfoNCE loss
        """
        self.actor_module.train()
        
        temperature = data.meta_info["temperature"]
        
        # Select keys
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        tis_imp_ratio_cap = getattr(self.config, 'tis_imp_ratio_cap', 0)
        if tis_imp_ratio_cap > 0:
            select_keys.append("rollout_log_probs")
        
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        
        # Add GAP-specific keys for InfoNCE
        if self.embedder_loss_weight > 0:
            for key in ["pos_text", "neg_texts", "extra_info", "query_prompt_no_hist"]:
                if key in data.non_tensor_batch.keys():
                    non_tensor_select_keys.append(key)
        
        # Save reward_extra_info from non_tensor_batch before data.select (if available)
        # Note: In verl's trainer, reward_extra_infos_dict is stored in batch.non_tensor_batch,
        # not in meta_info. Each key maps to a list of values for the batch.
        reward_extra_info_global = {}
        if hasattr(data, "non_tensor_batch"):
            # Extract reward-related keys from non_tensor_batch
            for key in ["format_reward", "retrieval_reward", "ranking_reward", "reranker_reward", "rewritten_query", "format_correct"]:
                if key in data.non_tensor_batch:
                    reward_extra_info_global[key] = data.non_tensor_batch[key]
        
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        
        metrics = {}
        
        for epoch_idx in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                
                self.actor_optimizer.zero_grad()
                
                # Track InfoNCE computation stats for this mini-batch
                infonce_stats = {"success": 0, "failed": 0, "fail_reasons": {}}
                
                for micro_idx, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    rollout_log_probs = model_inputs.get("rollout_log_probs") if tis_imp_ratio_cap > 0 else None
                    advantages = model_inputs["advantages"]
                    
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    
                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                    
                    # Forward with hidden states
                    calculate_entropy = entropy_coeff != 0
                    entropy, log_prob, last_hidden_state = self._forward_micro_batch_with_hidden(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    
                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]
                    
                    # Compute policy loss
                    policy_loss_cfg = getattr(self.config, 'policy_loss', {})
                    if hasattr(policy_loss_cfg, 'get'):
                        loss_mode = policy_loss_cfg.get("loss_mode", "vanilla")
                    elif hasattr(policy_loss_cfg, 'loss_mode'):
                        loss_mode = getattr(policy_loss_cfg, 'loss_mode', "vanilla")
                    else:
                        loss_mode = "vanilla"
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_log_probs=rollout_log_probs,
                    )
                    
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss
                    
                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                    
                    # InfoNCE loss (with best-rewrite selection mechanism from o1-embedder)
                    infonce_loss_value = 0.0
                    infonce_loss_tensor = None  # Store tensor for combination
                    infonce_computed = False
                    if self.embedder_loss_weight > 0 and self.emb_token_id is not None and self.infonce_loss_fn is not None:
                        # 🔑 Step 1: Get retrieval rewards to select best rewrites (o1-embedder style)
                        extra_info = model_inputs.get("extra_info", None)
                        retrieval_rewards_all = None
                        use_best_selection = self.use_best_rewrite_selection
                        num_generations = self.num_generations
                        
                        if extra_info is not None and use_best_selection:
                            # Normalize extra_info to Python list
                            if hasattr(extra_info, 'tolist'):
                                extra_info = list(extra_info.tolist())
                            elif isinstance(extra_info, np.ndarray):
                                extra_info = list(extra_info)
                            
                            # Try to get total_rewards_all from extra_info (for best rewrite selection)
                            # Use total_reward which includes format + reranker + retrieval + ranking
                            if isinstance(extra_info, list) and len(extra_info) > 0:
                                if isinstance(extra_info[0], dict):
                                    # Check if first item has total_reward
                                    total_rewards_list = []
                                    for e in extra_info:
                                        reward = e.get("total_reward", None)
                                        if reward is None:
                                            # Fallback to retrieval_reward for backward compatibility
                                            reward = e.get("retrieval_reward", None)
                                        if reward is not None:
                                            total_rewards_list.append(float(reward))
                                    
                                    if len(total_rewards_list) == len(extra_info):
                                        retrieval_rewards_all = torch.tensor(
                                            total_rewards_list, 
                                            device=last_hidden_state.device
                                        )
                        
                        # 🔑 Step 2: Select indices for InfoNCE (best selection or deduplication)
                        selected_indices = None
                        use_original_query_for_infonce = False  # Flag to control query type
                        
                        batch_size = last_hidden_state.size(0)
                        
                        if not use_best_selection:
                            # 🔑 Best selection disabled: Deduplicate and use original query
                            # Sample every num_generations-th sample (first sample from each group)
                            if batch_size % num_generations == 0:
                                num_queries = batch_size // num_generations
                                selected_indices = list(range(0, batch_size, num_generations))
                                use_original_query_for_infonce = True  # Use original query
                                
                                if batch_idx == 0 and epoch_idx == 0 and micro_idx == 0:
                                    print_rank_0(f"[GapActor] Best selection disabled: using deduplicated samples with original query")
                                    print_rank_0(f"[GapActor] Selected {len(selected_indices)} unique queries from {batch_size} samples (every {num_generations}th sample)")
                            else:
                                # Batch size not compatible, use all samples with rewritten query
                                if batch_idx == 0 and epoch_idx == 0:
                                    print_rank_0(f"[GapActor] Warning: Batch size ({batch_size}) is not a multiple of num_generations ({num_generations}). "
                                                f"Using all samples with rewritten query as fallback.")
                                selected_indices = None
                                use_original_query_for_infonce = False
                        elif retrieval_rewards_all is not None and len(retrieval_rewards_all) > 0:
                            # 🔑 Best selection enabled: Select best rewrite for each query
                            try:
                                # Validate length before reshape
                                if len(retrieval_rewards_all) == batch_size and batch_size % num_generations == 0:
                                    num_queries = batch_size // num_generations
                                    rewards_reshaped = retrieval_rewards_all.view(num_queries, num_generations)
                                    
                                    # Select best rewrite for each query
                                    best_indices = torch.argmax(rewards_reshaped, dim=-1)  # [num_queries]
                                    best_rewards = rewards_reshaped[torch.arange(num_queries), best_indices]  # [num_queries]
                                    
                                    # Convert to absolute indices
                                    selected_indices = []
                                    for i, best_idx in enumerate(best_indices):
                                        absolute_idx = i * num_generations + best_idx.item()
                                        selected_indices.append(absolute_idx)
                                    
                                    use_original_query_for_infonce = False  # Use conditional query (based on reward)
                                    
                                    # Only print once at the start of training
                                    if batch_idx == 0 and epoch_idx == 0 and micro_idx == 0:
                                        print_rank_0(f"[GapActor] Using best rewrite selection: {len(selected_indices)} samples selected from {batch_size}")
                                else:
                                    if batch_idx == 0 and epoch_idx == 0: # 仅在第一次打印提示，避免刷屏
                                        print_rank_0(f"[GapActor] Micro-batch size ({batch_size}) is not a multiple of num_generations ({num_generations}). "
                                                    f"Best rewrite selection is disabled for this step. "
                                                    f"To enable, set PPO_MICRO_BATCH_SIZE to a multiple of {num_generations}.")
                                    selected_indices = None
                                    use_original_query_for_infonce = False
                            except Exception as e:
                                logger.warning(f"[GapActor] Failed to select best rewrites: {e}, using all samples")
                                selected_indices = None
                                use_original_query_for_infonce = False
                        
                        # 🔑 Step 3: Get passage texts first (before filtering)
                        # We need to check pos_texts length before filtering query_embeddings
                        pos_texts = model_inputs.get("pos_text", None)
                        query_prompt_no_hist_batch = model_inputs.get("query_prompt_no_hist", None)
                        
                        # Try to extract from extra_info if not found at top level
                        # This is the expected data flow: Rollout Worker -> RewardManager (writes to extra_info) -> Actor (reads from extra_info)
                        if pos_texts is None:
                            if extra_info is not None:
                                # Handle numpy array (common in verl)
                                if isinstance(extra_info, np.ndarray):
                                    try:
                                        extra_info_list = extra_info.tolist()
                                        pos_texts = [e.get("pos_text", "") if isinstance(e, dict) else "" for e in extra_info_list]
                                    except:
                                        pass
                                elif isinstance(extra_info, list):
                                    pos_texts = [e.get("pos_text", "") if isinstance(e, dict) else "" for e in extra_info]
                                elif isinstance(extra_info, dict):
                                    pos_texts = extra_info.get("pos_text", None)
                        
                        # Also try to extract query_prompt_no_hist from extra_info
                        if query_prompt_no_hist_batch is None and extra_info is not None:
                            if isinstance(extra_info, np.ndarray):
                                try:
                                    extra_info_list = extra_info.tolist()
                                    query_prompt_no_hist_batch = [e.get("query_prompt_no_hist", "") if isinstance(e, dict) else "" for e in extra_info_list]
                                except:
                                    pass
                            elif isinstance(extra_info, list):
                                query_prompt_no_hist_batch = [e.get("query_prompt_no_hist", "") if isinstance(e, dict) else "" for e in extra_info]
                            elif isinstance(extra_info, dict):
                                query_prompt_no_hist_batch = extra_info.get("query_prompt_no_hist", None)
                        
                        # Validate selected_indices against pos_texts before filtering
                        if selected_indices is not None and pos_texts is not None:
                            if hasattr(pos_texts, 'tolist'):
                                pos_texts = pos_texts.tolist()
                            max_idx = max(selected_indices) if selected_indices else 0
                            if max_idx >= len(pos_texts):
                                logger.warning(
                                    f"[GapActor] selected_indices max ({max_idx}) >= len(pos_texts) ({len(pos_texts)}), "
                                    f"falling back to all samples"
                                )
                                selected_indices = None
                        
                        # 🔑 Step 4: Compute query embeddings (potentially filtered)
                        if selected_indices is not None:
                            # Save references to original data (needed for replacing with original query)
                            original_model_inputs = model_inputs
                            original_last_hidden_state = last_hidden_state
                            
                            # Only compute embeddings for selected best rewrites
                            # Need to filter model_inputs and last_hidden_state
                            filtered_model_inputs = {}
                            for key, value in model_inputs.items():
                                if isinstance(value, torch.Tensor):
                                    filtered_model_inputs[key] = value[selected_indices]
                                elif isinstance(value, (list, tuple)):
                                    filtered_model_inputs[key] = [value[i] for i in selected_indices]
                                elif isinstance(value, np.ndarray):
                                    # Handle numpy array (e.g., extra_info)
                                    filtered_model_inputs[key] = np.array([value[i] for i in selected_indices], dtype=object)
                                else:
                                    filtered_model_inputs[key] = value
                            
                            filtered_hidden_state = last_hidden_state[selected_indices]
                            
                            # Extract best rewards for conditional query construction
                            filtered_rewards = None
                            if not use_original_query_for_infonce and retrieval_rewards_all is not None and len(retrieval_rewards_all) > 0:
                                filtered_rewards = retrieval_rewards_all[selected_indices]
                            
                            query_embeddings = self._compute_query_embeddings_for_infonce(
                                filtered_model_inputs, 
                                filtered_hidden_state, 
                                filtered_rewards,
                                force_use_original_query=use_original_query_for_infonce
                            )
                        else:
                            # Use all samples (fallback when selection failed)
                            query_embeddings = self._compute_query_embeddings_for_infonce(
                                model_inputs, 
                                last_hidden_state, 
                                retrieval_rewards_all,
                                force_use_original_query=use_original_query_for_infonce
                            )
                        
                        # 🔑 Step 5: Filter pos_texts if using best selection (after query_embeddings computed)
                        if pos_texts is not None:
                            if hasattr(pos_texts, 'tolist'):
                                pos_texts = pos_texts.tolist()
                            
                            if selected_indices is not None:
                                pos_texts = [pos_texts[i] for i in selected_indices]
                            
                            valid_pos_texts = [str(t) for t in pos_texts if t and str(t).strip()]
                            
                            # 🔍 Track why InfoNCE computation might fail
                            infonce_fail_reason = None
                            if not valid_pos_texts:
                                # Check if pos_texts contains empty strings
                                empty_count = sum(1 for t in pos_texts if not t or not str(t).strip())
                                infonce_fail_reason = f"valid_pos_texts is empty (pos_texts len={len(pos_texts) if pos_texts else 0}, empty_count={empty_count})"
                                # Print for all micro-batches (not just first) to debug why some fail
                                print_rank_0(f"[GapActor] ✗ micro_batch {micro_idx}: All pos_texts are empty (batch {batch_idx}, epoch {epoch_idx})")
                            elif len(valid_pos_texts) != query_embeddings.size(0):
                                infonce_fail_reason = f"length mismatch: {len(valid_pos_texts)} valid_pos_texts vs {query_embeddings.size(0)} query_embeddings"
                                print_rank_0(f"[GapActor] ✗ micro_batch {micro_idx}: Length mismatch - {len(valid_pos_texts)} vs {query_embeddings.size(0)} (batch {batch_idx}, epoch {epoch_idx})")
                            
                            if valid_pos_texts and len(valid_pos_texts) == query_embeddings.size(0):
                                # Encode passages WITH gradient for InfoNCE
                                pos_embeddings = self._encode_for_embedding(valid_pos_texts, allow_grad=True)
                                
                                if pos_embeddings is None:
                                    infonce_fail_reason = "_encode_for_embedding returned None"
                                elif pos_embeddings.size(0) != query_embeddings.size(0):
                                    infonce_fail_reason = f"pos_embeddings size mismatch: {pos_embeddings.size(0)} vs {query_embeddings.size(0)}"
                                else:
                                    # Compute InfoNCE loss (only on best rewrites!)
                                    # Note: With best rewrite selection, batch_size may be small (num_queries)
                                    # This reduces the effectiveness of in-batch negatives
                                    num_queries_infonce = query_embeddings.size(0)
                                    
                                    # Warning: If only 1 query, InfoNCE will have no in-batch negatives
                                    # Cross-device gather helps but gathers different queries, not negatives for the same query
                                    if num_queries_infonce == 1:
                                        if batch_idx == 0 and epoch_idx == 0:
                                            print_rank_0(f"[GapActor] ⚠️ Warning: Only 1 query in micro batch for InfoNCE. "
                                                        f"InfoNCE loss will have no in-batch negatives and may be ineffective. "
                                                        f"Consider increasing ppo_micro_batch_size_per_gpu to be > num_generations.")
                                    
                                    infonce_loss = self.infonce_loss_fn(
                                        query_embeddings,
                                        pos_embeddings,
                                        group_size=1,
                                    )
                                    infonce_loss_tensor = infonce_loss  # Keep tensor for combination
                                    infonce_loss_value = infonce_loss.detach().item()
                                    infonce_computed = True
                                    infonce_stats["success"] += 1
                                    
                                    # Track how many samples used original vs rewritten query
                                    if use_original_query_for_infonce:
                                        # Force original query mode (deduplication)
                                        micro_batch_metrics["actor/infonce_original_query_count"] = query_embeddings.size(0)
                                        micro_batch_metrics["actor/infonce_rewritten_query_count"] = 0
                                        micro_batch_metrics["actor/infonce_original_query_ratio"] = 1.0
                                        micro_batch_metrics["actor/infonce_mode"] = 0  # 0=deduplication mode
                                    elif retrieval_rewards_all is not None and selected_indices is not None:
                                        # Conditional mode (based on retrieval rewards)
                                        filtered_rewards = retrieval_rewards_all[selected_indices]
                                        num_using_original = (filtered_rewards < 0).sum().item()
                                        num_using_rewritten = len(filtered_rewards) - num_using_original
                                        micro_batch_metrics["actor/infonce_original_query_count"] = num_using_original
                                        micro_batch_metrics["actor/infonce_rewritten_query_count"] = num_using_rewritten
                                        micro_batch_metrics["actor/infonce_mode"] = 1  # 1=conditional mode
                                        if num_using_original > 0:
                                            micro_batch_metrics["actor/infonce_original_query_ratio"] = num_using_original / len(filtered_rewards)
                                    else:
                                        # All rewritten mode
                                        micro_batch_metrics["actor/infonce_original_query_count"] = 0
                                        micro_batch_metrics["actor/infonce_rewritten_query_count"] = query_embeddings.size(0)
                                        micro_batch_metrics["actor/infonce_original_query_ratio"] = 0.0
                                        micro_batch_metrics["actor/infonce_mode"] = 2  # 2=all rewritten mode
                        else:
                            infonce_fail_reason = "pos_texts is None"
                        
                        # Log failure reason for all micro-batches (not just first)
                        if not infonce_computed:
                            infonce_stats["failed"] += 1
                            if infonce_fail_reason:
                                # Store failure reason in metrics for analysis
                                micro_batch_metrics[f"actor/infonce_fail_reason"] = infonce_fail_reason
                                # Track failure reasons
                                if infonce_fail_reason not in infonce_stats["fail_reasons"]:
                                    infonce_stats["fail_reasons"][infonce_fail_reason] = 0
                                infonce_stats["fail_reasons"][infonce_fail_reason] += 1
                    
                    # Combine RL loss and Embedder loss with weights
                    # total_loss = rl_loss_weight * policy_loss + embedder_loss_weight * infonce_loss
                    if infonce_computed and infonce_loss_tensor is not None:
                        # InfoNCE loss was computed, combine with policy loss
                        total_loss = self.rl_loss_weight * policy_loss + self.embedder_loss_weight * infonce_loss_tensor
                    else:
                        # No InfoNCE loss, use policy loss only
                        total_loss = self.rl_loss_weight * policy_loss
                    
                    # Always log InfoNCE metrics for tracking (even when not computed)
                    # Convert to float to ensure proper serialization
                    micro_batch_metrics["actor/infonce_loss"] = float(infonce_loss_value * loss_scale_factor)
                    micro_batch_metrics["actor/infonce_weighted_loss"] = float(
                        self.embedder_loss_weight * infonce_loss_value * loss_scale_factor
                    )
                    
                    # Log micro-batch index for debugging
                    micro_batch_metrics["actor/total_loss"] = float(total_loss.detach().item() * loss_scale_factor)
                    
                    # Scale and backward
                    loss = total_loss * loss_scale_factor
                    loss.backward()
                    
                    micro_batch_metrics.update({
                        "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/total_loss": total_loss.detach().item() * loss_scale_factor,
                    })
                    
                    # Extract reward metrics from reward_extra_info_global
                    # All metrics are logged under custom/ prefix for consistency
                    if reward_extra_info_global:
                        try:
                            format_rewards = reward_extra_info_global.get("format_reward", None)
                            retrieval_rewards = reward_extra_info_global.get("retrieval_reward", None)
                            ranking_rewards = reward_extra_info_global.get("ranking_reward", None)
                            reranker_rewards = reward_extra_info_global.get("reranker_reward", None)
                            format_correct = reward_extra_info_global.get("format_correct", None)
                            rewritten_queries = reward_extra_info_global.get("rewritten_query", None)
                            
                            # Convert numpy arrays to lists for easier handling
                            if isinstance(format_rewards, np.ndarray):
                                format_rewards = format_rewards.tolist()
                            if isinstance(retrieval_rewards, np.ndarray):
                                retrieval_rewards = retrieval_rewards.tolist()
                            if isinstance(ranking_rewards, np.ndarray):
                                ranking_rewards = ranking_rewards.tolist()
                            if isinstance(reranker_rewards, np.ndarray):
                                reranker_rewards = reranker_rewards.tolist()
                            if isinstance(format_correct, np.ndarray):
                                format_correct = format_correct.tolist()
                            if isinstance(rewritten_queries, np.ndarray):
                                rewritten_queries = rewritten_queries.tolist()
                            
                            # Log format reward metrics
                            if format_rewards is not None and len(format_rewards) > 0:
                                micro_batch_metrics["custom/format_reward_mean"] = float(np.mean(format_rewards))
                            
                            # Log retrieval reward metrics
                            if retrieval_rewards is not None and len(retrieval_rewards) > 0:
                                micro_batch_metrics["custom/retrieval_reward_mean"] = float(np.mean(retrieval_rewards))
                                micro_batch_metrics["custom/retrieval_reward_min"] = float(np.min(retrieval_rewards))
                                micro_batch_metrics["custom/retrieval_reward_max"] = float(np.max(retrieval_rewards))
                                micro_batch_metrics["custom/retrieval_reward_std"] = float(np.std(retrieval_rewards))
                            
                            # Log ranking reward metrics
                            if ranking_rewards is not None and len(ranking_rewards) > 0:
                                micro_batch_metrics["custom/ranking_reward_mean"] = float(np.mean(ranking_rewards))
                            
                            # Log reranker reward metrics
                            if reranker_rewards is not None and len(reranker_rewards) > 0:
                                micro_batch_metrics["custom/reranker_reward_mean"] = float(np.mean(reranker_rewards))
                                micro_batch_metrics["custom/reranker_reward_min"] = float(np.min(reranker_rewards))
                                micro_batch_metrics["custom/reranker_reward_max"] = float(np.max(reranker_rewards))
                                micro_batch_metrics["custom/reranker_reward_std"] = float(np.std(reranker_rewards))
                            
                            # Log ranking top1 ratio
                            # Note: ranking_top1_ratio is now a list (repeated batch_size times) to prevent 0D array in chunk()
                            ranking_top1_ratio = reward_extra_info_global.get("ranking_top1_ratio", None)
                            if ranking_top1_ratio is not None:
                                # Extract first element (all elements are the same)
                                if isinstance(ranking_top1_ratio, (list, np.ndarray)) and len(ranking_top1_ratio) > 0:
                                    micro_batch_metrics["custom/ranking_top1_ratio"] = float(ranking_top1_ratio[0])
                                else:
                                    micro_batch_metrics["custom/ranking_top1_ratio"] = float(ranking_top1_ratio)
                            
                            # Log format correctness ratio
                            if format_correct is not None and len(format_correct) > 0:
                                correct_count = sum(1 for x in format_correct if x)
                                micro_batch_metrics["custom/format_correct_ratio"] = float(correct_count / len(format_correct))
                            
                            # Log rewrite success ratio
                            if rewritten_queries is not None and len(rewritten_queries) > 0:
                                non_empty_count = len([q for q in rewritten_queries if q and str(q).strip()])
                                micro_batch_metrics["custom/rewrite_success_ratio"] = float(non_empty_count / len(rewritten_queries))
                        except Exception as e:
                            logger.warning(f"[GapActor] Could not extract reward metrics: {e}")
                    
                    append_to_dict(metrics, micro_batch_metrics)
                
                # Print InfoNCE statistics for this mini-batch (only on first batch/epoch)
                total_micro_batches = len(micro_batches)
                success_rate = infonce_stats["success"] / total_micro_batches if total_micro_batches > 0 else 0.0
                if batch_idx == 0 and epoch_idx == 0:
                    print_rank_0(
                        f"[GapActor] InfoNCE: {infonce_stats['success']}/{total_micro_batches} success ({success_rate:.1%}), "
                        f"{infonce_stats['failed']} failed"
                    )
                
                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        
        self.actor_optimizer.zero_grad()
        
        return metrics
