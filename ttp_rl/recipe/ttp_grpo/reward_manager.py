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
GAP-GRPO v3 Reward Manager

Enhanced reward manager with:
1. Format Reward: Per-sample format checking
2. Retrieval Reward: Batch-level comparison of original vs rewritten query retrieval

Adapted from o1-embedder-training framework.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from .reward_function import (
    compute_format_reward,
    compute_rewrite_gain_reward,
    DEBUG_LOG_ENABLED,
)
from .utils import (
    compute_retrieval_reward,
    compute_ranking_reward,
    retrieval_reward_map_func,
)
# Import reranker utilities for batch GPU processing
try:
    import ray
    from .reranker import (
        get_ray_reranker_actor,
        preprocess_query_for_reranker,
    )
except ImportError:
    ray = None
    get_ray_reranker_actor = None
    preprocess_query_for_reranker = None


@register("gap_grpo")
class GapGRPOV3RewardManager(AbstractRewardManager):
    """
    Enhanced reward manager for GAP-GRPO v3 training.
    
    This reward manager computes:
    1. Format Reward: Per-sample, checks <think>...</think><emb> format
    2. Reranker Reward: Per-sample, uses external reranker model to compute rewrite gain
    3. Retrieval Reward: Batch-level, compares original vs rewritten query retrieval performance
    4. Ranking Reward: Batch-level, based on rank of positive passage in batch
    """
    
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        # Reward weights
        format_weight: float = 0.3,
        retrieval_weight: float = 0.5,
        ranking_weight: float = 0.2,
        reranker_reward_weight: float = 0.0,  # Reranker reward weight (0 to disable)
        # Reranker params
        reranker_model_path: str = "BAAI/bge-reranker-base",
        gain_scale: float = 1.0,
        penalty_scale: float = 0.5,
        min_reward: float = -1.0,
        max_reward: float = 1.0,
        # Format reward params
        format_reward_value: float = 0.5,
        format_penalty_value: float = -0.5,
        length_threshold: int = 30,
        length_penalty: float = -0.1,
        repetition_penalty: float = -0.2,
        containment_penalty: float = -0.5,
        # Retrieval params
        temperature: float = 0.02,
        num_generations: int = 4,  # GRPO sampling number
        retrieval_reward_pos_weight: float = 1.0,
        retrieval_reward_neg_weight: float = 1.0,
        use_retrieval_reward_map: bool = False,
        retrieval_reward_clip_range: float = 100.0,
        retrieval_reward_target_range: float = 2.0,
        # Asymmetric clip params 
        use_asymmetric_clip: bool = False,
        asymmetric_neg_clip: float = None,
        asymmetric_pos_clip: float = None,
        asymmetric_neg_target: float = None,
        asymmetric_pos_target: float = None,
        **kwargs,
    ) -> None:
        """
        Initialize the GAP-GRPO v3 reward manager.
        
        Args:
            tokenizer: Tokenizer for decoding responses
            num_examine: Number of samples to print for debugging
            compute_score: Custom compute score function (optional)
            reward_fn_key: Key for data source
            format_weight: Weight for format reward
            retrieval_weight: Weight for retrieval reward
            ranking_weight: Weight for ranking reward
            reranker_reward_weight: Weight for reranker reward (0 to disable)
            reranker_model_path: Path to reranker model for rewrite gain reward
            gain_scale: Scaling factor for positive gain (when similarity improves)
            penalty_scale: Scaling factor for negative gain (when similarity decreases)
            min_reward: Minimum reward value
            max_reward: Maximum reward value
            format_reward_value: Reward for correct format
            format_penalty_value: Penalty for incorrect format
            length_threshold: Length threshold for penalty
            length_penalty: Penalty per excess length (if length > threshold)
            repetition_penalty: Penalty for repeated special tokens
            containment_penalty: Penalty for missing characters from original query
            temperature: Temperature for similarity computation
            num_generations: Number of generations per query (GRPO)
            retrieval_reward_pos_weight: Weight for positive samples in retrieval reward
            retrieval_reward_neg_weight: Weight for negative samples in retrieval reward
            use_retrieval_reward_map: Whether to use reward mapping for retrieval reward
            retrieval_reward_clip_range: Clip range for reward mapping (default 100.0)
            retrieval_reward_target_range: Target range for reward mapping (default 2.0)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score_fn = compute_score
        self.reward_fn_key = reward_fn_key
        
        # Reward weights
        self.format_weight = format_weight
        self.retrieval_weight = retrieval_weight
        self.ranking_weight = ranking_weight
        
        # Reward params
        self.format_reward_value = format_reward_value
        self.format_penalty_value = format_penalty_value
        self.length_threshold = length_threshold
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.containment_penalty = containment_penalty
        self.temperature = temperature
        self.num_generations = num_generations
        self.retrieval_reward_pos_weight = retrieval_reward_pos_weight
        self.retrieval_reward_neg_weight = retrieval_reward_neg_weight
        self.use_retrieval_reward_map = use_retrieval_reward_map
        self.retrieval_reward_clip_range = retrieval_reward_clip_range
        self.retrieval_reward_target_range = retrieval_reward_target_range
        
        # Asymmetric clip params
        self.use_asymmetric_clip = use_asymmetric_clip
        self.asymmetric_neg_clip = asymmetric_neg_clip
        self.asymmetric_pos_clip = asymmetric_pos_clip
        self.asymmetric_neg_target = asymmetric_neg_target
        self.asymmetric_pos_target = asymmetric_pos_target
        
        # Reranker reward params
        self.reranker_reward_weight = reranker_reward_weight
        self.reranker_model_path = reranker_model_path
        self.gain_scale = gain_scale
        self.penalty_scale = penalty_scale
        self.min_reward = min_reward
        self.max_reward = max_reward
        
        # Lazy initialization
        self._rewrite_reward_model = None
        
        # Special tokens
        self.think_open = "<think>"
        self.think_close = "</think>"
        self.emb_tok = "<emb>"
    
    @property
    def rewrite_reward_model(self):
        """Lazy initialization of rewrite reward model."""
        if self._rewrite_reward_model is None:
            from .reranker import RewriteRewardModel
            self._rewrite_reward_model = RewriteRewardModel(
                model_name_or_path=self.reranker_model_path
            )
        return self._rewrite_reward_model
    
    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Compute rewards for the batch.
        
        Args:
            data: DataProto containing batch data
            return_dict: Whether to return additional info
        
        Returns:
            reward_tensor or dict with reward_tensor and extra_info
        """
        # Check if pre-computed rewards exist
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": {},
                }
            return data.batch["rm_scores"]
        
        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        # Initialize ranking_reward and reranker_reward lists
        reward_extra_info["ranking_reward"] = []
        reward_extra_info["reranker_reward"] = []
        # Store reranker scores for logging
        reward_extra_info["reranker_original_score"] = []
        reward_extra_info["reranker_rewritten_score"] = []
        
        already_print_data_sources = {}
        
        # Collect embeddings for retrieval rewards
        ori_query_embeddings = []
        rewritten_query_embeddings = []
        passage_embeddings = []
        has_retrieval_data = False
        
        # ===== Reranker Reward: Batch Processing (GPU) =====
        # Collect pairs for batch reranker computation (similar to gap_grpo)
        reranker_pairs = []  # List of (sample_idx, original_query, rewritten_query, positive_doc)
        reranker_rewards = [0.0] * batch_size  # Pre-initialize all samples
        
        for i in range(batch_size):
            data_item = data[i]
            
            # Get prompt and response
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            
            # 🔑 KEY FIX: Always get info from reward_model.ground_truth
            ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", {})
            original_query = ground_truth.get("query", "")
            if isinstance(original_query, (list, tuple)):
                original_query = str(original_query[-1]) if original_query else ""
            
            # Get vectors from Worker's new key
            # Note: retrieval_scores is stored per-item in data_item, not as a batch-level array
            retrieval_item = data_item.non_tensor_batch.get("retrieval_scores", None)
            retrieval_data = {}
            
            if retrieval_item is not None:
                # retrieval_item should be a dict containing embeddings and scores
                if isinstance(retrieval_item, dict):
                    retrieval_data = retrieval_item
                elif hasattr(retrieval_item, 'item'):
                    retrieval_data = retrieval_item.item()
            
            # Prepare retrieval_info for reward computation (local copy from retrieval_scores)
            # Note: This is different from extra_info_item below (which is written back to data_item)
            retrieval_info = dict(retrieval_data)
            retrieval_info["query_prompt_no_hist"] = ground_truth.get("query_prompt_no_hist", "")
            # Ensure pos_text is present for local use (not written back yet)
            if "pos_text" not in retrieval_info and isinstance(ground_truth, dict):
                retrieval_info["pos_text"] = ground_truth.get("pos_text", None)
            
            # 1. Format reward
            format_r, format_info = compute_format_reward(
                response_str=response_str,
                format_reward=self.format_reward_value,
                format_penalty=self.format_penalty_value,
                length_threshold=self.length_threshold,
                length_penalty=self.length_penalty,
                repetition_penalty=self.repetition_penalty,
                original_query=original_query,
                containment_penalty=self.containment_penalty,
            )
            
            rewritten_query = format_info.get("rewritten_query", response_str.strip())
            
            # 2. Collect reranker pairs for batch processing (computed after loop)
            # Get positive document from ground_truth
            positive_doc = ground_truth.get("pos", "")
            if self.reranker_reward_weight > 0 and rewritten_query and positive_doc:
                reranker_pairs.append((i, original_query, rewritten_query, positive_doc))
            
            # reranker_r will be set from reranker_rewards[i] after batch computation
            
            # Collect embeddings for retrieval rewards
            # These should be computed in rollout worker and passed via retrieval_scores
            if "ori_query_embedding" in retrieval_info:
                ori_query_embeddings.append(retrieval_info["ori_query_embedding"])
                rewritten_query_embeddings.append(retrieval_info.get("rewritten_query_embedding"))
                passage_embeddings.append(retrieval_info.get("passage_embeddings"))
                has_retrieval_data = True
            
            # Compute partial reward (format only, reranker will be added after batch computation)
            # reranker_r will be set from reranker_rewards[i] after batch computation
            partial_reward = self.format_weight * format_r
            
            # Store in tensor (at the last valid response position)
            reward_tensor[i, valid_response_length - 1] = partial_reward
            
            # Store extra info (reranker_reward will be appended after batch computation)
            reward_extra_info["format_reward"].append(format_r)
            reward_extra_info["format_correct"].append(format_info.get("format_correct", False))
            reward_extra_info["rewritten_query"].append(rewritten_query)
            # reranker_reward will be appended after batch computation
            
            # Ensure extra_info is propagated to actor for InfoNCE (pos_text / query_prompt_no_hist)
            extra_info_item = data_item.non_tensor_batch.get("extra_info", {})
            if not isinstance(extra_info_item, dict):
                extra_info_item = {}
            
            # Get pos_text from ground_truth (should be set by dataset)
            pos_text_from_gt = None
            if isinstance(ground_truth, dict):
                pos_text_from_gt = ground_truth.get("pos_text", None)
            
            # Ensure pos_text is in extra_info for Actor
            if "pos_text" not in extra_info_item:
                if pos_text_from_gt is not None:
                    extra_info_item["pos_text"] = pos_text_from_gt
                else:
                    # Fallback: try to get from top-level (if dataset set it)
                    pos_text_top = data_item.non_tensor_batch.get("pos_text", None)
                    if pos_text_top is not None:
                        extra_info_item["pos_text"] = pos_text_top
                    else:
                        # Last resort: empty string (will cause InfoNCE to skip)
                        extra_info_item["pos_text"] = ""
                        if i < 3:  # Only log first few samples to avoid spam
                            print(f"[GapGRPOV3RewardManager] Warning: Sample {i} has no pos_text in ground_truth or top-level")
            
            if "query_prompt_no_hist" not in extra_info_item and isinstance(ground_truth, dict):
                extra_info_item["query_prompt_no_hist"] = ground_truth.get("query_prompt_no_hist", "")
            
            # 🔑 Add original query to extra_info for InfoNCE (needed when force_use_original_query=True)
            if "query" not in extra_info_item and isinstance(ground_truth, dict):
                original_query = ground_truth.get("query", "")
                if isinstance(original_query, (list, tuple)):
                    original_query = str(original_query[-1]) if original_query else ""
                else:
                    original_query = str(original_query)
                extra_info_item["query"] = original_query
            
            data_item.non_tensor_batch["extra_info"] = extra_info_item
        
        # ===== Batch Reranker Reward Computation (GPU) =====
        # Process all reranker pairs in batch (similar to gap_grpo's compute_score_batch)
        # Store reranker scores for logging (initialize outside try block)
        reranker_original_scores = {}  # Store original scores by sample_idx
        reranker_rewritten_scores = {}  # Store rewritten scores by sample_idx
        
        if reranker_pairs and self.reranker_reward_weight > 0:
            try:
                # Prepare batch inputs
                all_queries_for_rerank = []
                all_docs_for_rerank = []
                pair_indices = []  # Track which samples have valid reranker input
                
                for sample_idx, orig_q, rew_q, pos_doc in reranker_pairs:
                    # Preprocess queries
                    if preprocess_query_for_reranker:
                        cleaned_original = preprocess_query_for_reranker(orig_q)
                    else:
                        # Fallback if import failed
                        cleaned_original = str(orig_q).split('geohash')[0].strip()
                    cleaned_rewritten = str(rew_q).strip()
                    
                    # Handle positive doc format
                    if isinstance(pos_doc, (list, tuple)):
                        cleaned_doc = str(pos_doc[-1]) if pos_doc else ""
                    else:
                        cleaned_doc = str(pos_doc).strip()
                    
                    # Validate all inputs are non-empty to avoid tokenizer errors
                    if cleaned_doc and cleaned_original and cleaned_rewritten:
                        # Add original query -> doc pair
                        all_queries_for_rerank.append(cleaned_original)
                        all_docs_for_rerank.append(cleaned_doc)
                        # Add rewritten query -> doc pair
                        all_queries_for_rerank.append(cleaned_rewritten)
                        all_docs_for_rerank.append(cleaned_doc)
                        pair_indices.append(sample_idx)
                
                # Batch compute reranker scores
                if all_queries_for_rerank:
                    use_ray_actor = False
                    if ray is not None and ray.is_initialized() and get_ray_reranker_actor is not None:
                        try:
                            use_ray_actor = True
                            # Use Ray Actor (GPU)
                            reranker_actor = get_ray_reranker_actor(self.reranker_model_path)
                            all_scores = ray.get(reranker_actor.compute_batch_scores.remote(
                                all_queries_for_rerank,
                                all_docs_for_rerank,
                            ))
                            print(f"[GapGRPOV3RewardManager] Processed {len(all_queries_for_rerank)} reranker pairs via Ray Actor (GPU)")
                        except Exception as e:
                            print(f"[GapGRPOV3RewardManager] Warning: Ray Actor failed, falling back to local batch: {e}")
                            use_ray_actor = False
                    
                    if not use_ray_actor:
                        # Use local batch computation
                        all_scores = self.rewrite_reward_model.compute_batch_scores(
                            all_queries_for_rerank,
                            all_docs_for_rerank,
                        )
                        print(f"[GapGRPOV3RewardManager] Processed {len(all_queries_for_rerank)} reranker pairs via local batch")
                    
                    # Parse scores back to samples (every 2 scores belong to one sample)
                    # 🔍 DEBUG: For first sample, verify score order by computing separately
                    if len(pair_indices) > 0 and DEBUG_LOG_ENABLED:
                        first_sample_idx, first_orig_q, first_rew_q, first_pos_doc = reranker_pairs[0]
                        # Preprocess first sample
                        if preprocess_query_for_reranker:
                            first_cleaned_orig = preprocess_query_for_reranker(first_orig_q)
                        else:
                            first_cleaned_orig = str(first_orig_q).split('geohash')[0].strip()
                        first_cleaned_rew = str(first_rew_q).strip()
                        if isinstance(first_pos_doc, (list, tuple)):
                            first_cleaned_doc = str(first_pos_doc[-1]) if first_pos_doc else ""
                        else:
                            first_cleaned_doc = str(first_pos_doc).strip()
                        
                        # Compute separately to verify order
                        try:
                            first_orig_score_sep = self.rewrite_reward_model.compute_batch_scores(
                                [first_cleaned_orig], [first_cleaned_doc]
                            )[0]
                            first_rew_score_sep = self.rewrite_reward_model.compute_batch_scores(
                                [first_cleaned_rew], [first_cleaned_doc]
                            )[0]
                            
                            # Compare with batch results
                            first_orig_score_batch = all_scores[0]
                            first_rew_score_batch = all_scores[1]
                            
                            print(f"\n[Reranker Score Verification] Sample {first_sample_idx}:")
                            print(f"  Original Query (preprocessed): {first_cleaned_orig[:100]}...")
                            print(f"  Rewritten Query: {first_cleaned_rew[:100]}...")
                            print(f"  Positive Doc: {first_cleaned_doc[:100]}...")
                            print(f"  Separate: orig={first_orig_score_sep:.4f}, rew={first_rew_score_sep:.4f}, gain={first_rew_score_sep - first_orig_score_sep:.4f}")
                            print(f"  Batch:     orig={first_orig_score_batch:.4f}, rew={first_rew_score_batch:.4f}, gain={first_rew_score_batch - first_orig_score_batch:.4f}")
                            
                            # Check if order is correct
                            if abs(first_orig_score_sep - first_orig_score_batch) > 0.01 or abs(first_rew_score_sep - first_rew_score_batch) > 0.01:
                                print(f"  WARNING: Score mismatch! Order might be wrong!")
                                print(f"     Expected: orig={first_orig_score_sep:.4f}, rew={first_rew_score_sep:.4f}")
                                print(f"     Got:      orig={first_orig_score_batch:.4f}, rew={first_rew_score_batch:.4f}")
                                # Try swapped order
                                if abs(first_orig_score_sep - first_rew_score_batch) < 0.01 and abs(first_rew_score_sep - first_orig_score_batch) < 0.01:
                                    print(f"    FIX: Scores are swapped! Should use swapped order.")
                        except Exception as e:
                            print(f"[GapGRPOV3RewardManager] Warning: Failed to verify reranker score order: {e}")
                    
                    for j, sample_idx in enumerate(pair_indices):
                        original_score = all_scores[j * 2]
                        rewritten_score = all_scores[j * 2 + 1]
                        
                        # Store scores for logging
                        reranker_original_scores[sample_idx] = original_score
                        reranker_rewritten_scores[sample_idx] = rewritten_score
                        
                        # Compute gain
                        gain = rewritten_score - original_score
                        
                        if j == 0 and DEBUG_LOG_ENABLED:
                            orig_q, rew_q, pos_doc = reranker_pairs[j][1], reranker_pairs[j][2], reranker_pairs[j][3]
                            if preprocess_query_for_reranker:
                                cleaned_orig = preprocess_query_for_reranker(orig_q)
                            else:
                                cleaned_orig = str(orig_q).split('geohash')[0].strip()
                            cleaned_rew = str(rew_q).strip()
                            print(f"\n[Reranker Reward Details] Sample {sample_idx}:")
                            print(f"  Original Query: {cleaned_orig[:150]}")
                            print(f"  Rewritten Query: {cleaned_rew[:150]}")
                            print(f"  Original Score: {original_score:.4f}")
                            print(f"  Rewritten Score: {rewritten_score:.4f}")
                            print(f"  Gain: {gain:.4f} ({'+' if gain >= 0 else ''}{gain:.4f})")
                            print(f"  Reward (before clip): {self.gain_scale * gain if gain >= 0 else self.penalty_scale * gain:.4f}")
                        
                        # Apply scaling factors
                        if gain < 0:
                            reward = self.penalty_scale * gain
                        else:
                            reward = self.gain_scale * gain
                        
                        # Clip reward
                        reward = max(self.min_reward, min(self.max_reward, reward))
                        reranker_rewards[sample_idx] = reward
                        
                        if j == 0 and DEBUG_LOG_ENABLED:
                            print(f"  Final Reward (after clip): {reward:.4f}")
                        
            except Exception as e:
                print(f"[GapGRPOV3RewardManager] Warning: Failed to compute batch reranker rewards: {e}")
                import traceback
                traceback.print_exc()
        
        # Update reward_tensor and reward_extra_info with reranker rewards
        for i in range(batch_size):
            reranker_r = reranker_rewards[i]
            # Add reranker reward to partial reward
            valid_response_length = data[i].batch["attention_mask"][data[i].batch["prompts"].shape[-1]:].sum()
            reward_tensor[i, valid_response_length - 1] += self.reranker_reward_weight * reranker_r
            # Append reranker reward to extra info
            reward_extra_info["reranker_reward"].append(reranker_r)
            # Append reranker scores for logging
            reward_extra_info["reranker_original_score"].append(reranker_original_scores.get(i, 0.0))
            reward_extra_info["reranker_rewritten_score"].append(reranker_rewritten_scores.get(i, 0.0))
        
        # 2. Batch-level retrieval reward
        # Verify GRPO data structure assumption
        # In GRPO, batch_size = num_queries * num_generations
        # We need to group by queries: each query has num_generations samples
        if batch_size % self.num_generations != 0:
            print(f"[GapGRPOV3RewardManager] Warning: batch_size ({batch_size}) not divisible by num_generations ({self.num_generations})")
            print(f"[GapGRPOV3RewardManager] Skipping retrieval reward computation due to invalid batch structure.")
            has_retrieval_data = False  # Skip retrieval reward if batch structure is invalid
        
        num_queries = batch_size // self.num_generations if batch_size % self.num_generations == 0 else 0
        
        # Critical check: ensure we have embeddings for all samples
        if has_retrieval_data:
            if len(ori_query_embeddings) != batch_size:
                print(f"[GapGRPOV3RewardManager] Warning: len(ori_query_embeddings)={len(ori_query_embeddings)} != batch_size={batch_size}")
                print(f"[GapGRPOV3RewardManager] Some samples may have been skipped in rollout. Skipping retrieval reward computation.")
                has_retrieval_data = False
        else:
            # Only print warning if we expected retrieval data but don't have it
            if batch_size > 0:
                print(f"[GapGRPOV3RewardManager] Warning: No retrieval data available (has_retrieval_data={has_retrieval_data}, len(ori_query_embeddings)={len(ori_query_embeddings)})")
        
        if has_retrieval_data and len(ori_query_embeddings) == batch_size:
            try:
                # Convert to tensors
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # In GRPO, same query's generations share the same original query embedding
                # Extract unique original query embeddings: take first generation of each query
                ori_emb_list = []
                for query_idx in range(num_queries):
                    sample_idx = query_idx * self.num_generations
                    if sample_idx < len(ori_query_embeddings):
                        e = ori_query_embeddings[sample_idx]
                        if isinstance(e, (list, np.ndarray)):
                            ori_emb_list.append(torch.tensor(e, device=device))
                        elif isinstance(e, torch.Tensor):
                            ori_emb_list.append(e.to(device))
                        else:
                            ori_emb_list.append(torch.tensor(e, device=device))
                    else:
                        # Fallback: use zero embedding
                        if ori_emb_list:
                            ori_emb_list.append(torch.zeros_like(ori_emb_list[0]))
                        else:
                            # Need to infer hidden_dim from first available embedding
                            first_emb = ori_query_embeddings[0] if ori_query_embeddings else None
                            if first_emb is not None:
                                if isinstance(first_emb, (list, np.ndarray)):
                                    hidden_dim = len(first_emb) if isinstance(first_emb[0], (int, float)) else len(first_emb[0])
                                else:
                                    hidden_dim = first_emb.size(-1) if isinstance(first_emb, torch.Tensor) else 768
                                ori_emb_list.append(torch.zeros(hidden_dim, device=device))
                            else:
                                ori_emb_list.append(torch.zeros(768, device=device))  # Default hidden_dim
                
                ori_emb_tensor = torch.stack(ori_emb_list)  # [num_queries, hidden_dim]
                
                # Rewritten query embeddings: [batch_size, hidden_dim] (all generations)
                rewritten_emb_list = []
                for i, e in enumerate(rewritten_query_embeddings):
                    if e is not None:
                        if isinstance(e, (list, np.ndarray)):
                            # If it's a list of embeddings (shouldn't happen in GRPO), take first
                            if len(e) > 0 and isinstance(e[0], (list, np.ndarray)):
                                rewritten_emb_list.append(torch.tensor(e[0], device=device))
                            else:
                                rewritten_emb_list.append(torch.tensor(e, device=device))
                        elif isinstance(e, torch.Tensor):
                            rewritten_emb_list.append(e.to(device))
                        else:
                            rewritten_emb_list.append(torch.tensor(e, device=device))
                    else:
                        # Fallback: use corresponding original query embedding
                        # Use original index i instead of len(rewritten_emb_list) to correctly map to query
                        query_idx = i // self.num_generations
                        if query_idx < ori_emb_tensor.size(0):
                            rewritten_emb_list.append(ori_emb_tensor[query_idx])
                        else:
                            rewritten_emb_list.append(torch.zeros(ori_emb_tensor.size(1), device=device))
                
                rewritten_emb_tensor = torch.stack(rewritten_emb_list)  # [batch_size, hidden_dim]
                
                # Passage embeddings: extract only positive passage for each query (in-batch negative mode)
                # In GRPO, same query's generations share the same passage embeddings
                # Extract unique passage embeddings: take first generation of each query
                passage_emb_list = []
                for query_idx in range(num_queries):
                    sample_idx = query_idx * self.num_generations
                    if sample_idx < len(passage_embeddings):
                        e = passage_embeddings[sample_idx]
                        if e is not None:
                            if isinstance(e, (list, np.ndarray)):
                                # If list/array, take first element (positive passage)
                                if len(e) > 0:
                                    # Check if e[0] is a scalar or array
                                    if isinstance(e[0], (list, np.ndarray)):
                                        # e is list of arrays, take first array
                                        passage_emb_list.append(torch.tensor(e[0], device=device))
                                    else:
                                        # e is 2D array, take first row
                                        passage_emb_list.append(torch.tensor(e[0], device=device))
                                else:
                                    # Fallback: use zero embedding
                                    passage_emb_list.append(torch.zeros(ori_emb_tensor.size(1), device=device))
                            elif isinstance(e, torch.Tensor):
                                # If tensor, take first element if 2D, otherwise use as is
                                if e.dim() > 1 and e.size(0) > 0:
                                    passage_emb_list.append(e[0].to(device))
                                else:
                                    passage_emb_list.append(e.to(device) if e.dim() == 1 else e[0].to(device))
                            else:
                                passage_emb_list.append(torch.tensor(e, device=device))
                        else:
                            # Fallback: use zero embedding if None
                            passage_emb_list.append(torch.zeros(ori_emb_tensor.size(1), device=device))
                    else:
                        # Fallback: use zero embedding
                        passage_emb_list.append(torch.zeros(ori_emb_tensor.size(1), device=device))
                
                if passage_emb_list:
                    # Stack to [num_queries, hidden_dim] - one positive passage per query
                    passage_emb_tensor = torch.stack(passage_emb_list)
                    
                    # Compute retrieval reward (improved: pos score + gap improvement)
                    # Use in-batch negatives: each query has one positive passage,
                    # negatives are other queries' positive passages in the batch
                    # Note: ori_emb_tensor is [num_queries, hidden_dim], passage_emb_tensor is [num_queries, hidden_dim]
                    # rewritten_emb_tensor is [batch_size, hidden_dim] = [num_queries * num_generations, hidden_dim]
                    # Handle None values for asymmetric clip parameters (fallback to symmetric values)
                    asymmetric_neg_clip = self.asymmetric_neg_clip if self.asymmetric_neg_clip is not None else self.retrieval_reward_clip_range
                    asymmetric_pos_clip = self.asymmetric_pos_clip if self.asymmetric_pos_clip is not None else self.retrieval_reward_clip_range
                    asymmetric_neg_target = self.asymmetric_neg_target if self.asymmetric_neg_target is not None else self.retrieval_reward_target_range
                    asymmetric_pos_target = self.asymmetric_pos_target if self.asymmetric_pos_target is not None else self.retrieval_reward_target_range
                    
                    retrieval_rewards = compute_retrieval_reward(
                        ori_query_embeddings=ori_emb_tensor,  # [num_queries, hidden_dim]
                        rewritten_query_embeddings=rewritten_emb_tensor,  # [num_queries * num_generations, hidden_dim]
                        passage_embeddings=passage_emb_tensor,  # [num_queries, hidden_dim]
                        num_generations=self.num_generations,
                        pos_weight=self.retrieval_reward_pos_weight,
                        gap_weight=self.retrieval_reward_neg_weight,  # Reuse as gap_weight
                        temperature=self.temperature,
                        use_reward_map=self.use_retrieval_reward_map,
                        use_in_batch_neg=True,  # Use in-batch negatives
                        reward_map_clip_range=self.retrieval_reward_clip_range,
                        reward_map_target_range=self.retrieval_reward_target_range,
                        use_asymmetric_clip=self.use_asymmetric_clip,
                        asymmetric_neg_clip=asymmetric_neg_clip,
                        asymmetric_pos_clip=asymmetric_pos_clip,
                        asymmetric_neg_target=asymmetric_neg_target,
                        asymmetric_pos_target=asymmetric_pos_target,
                    )
                    
                    # Compute ranking reward (based on rank of positive passage in batch)
                    # Note: If num_queries == 1, ranking reward will be all 1.0 (no discrimination)
                    # This is expected when micro_batch_size == num_generations
                    if num_queries == 1:
                        # Only one query, ranking reward has no meaning (rank is always 1)
                        # Skip computation and use all 1.0 rewards
                        ranking_rewards = torch.ones(batch_size, device=device)
                        ranking_ranks = torch.ones(batch_size, device=device, dtype=torch.long)  # All rank 1
                    else:
                        ranking_result = compute_ranking_reward(
                            rewritten_query_embeddings=rewritten_emb_tensor,  # [num_queries * num_generations, hidden_dim]
                            passage_embeddings=passage_emb_tensor,  # [num_queries, hidden_dim]
                            num_generations=self.num_generations,
                            temperature=self.temperature,
                            use_in_batch_neg=True,
                            return_ranks=True,  # Also return ranks for top1 ratio calculation
                        )
                        ranking_rewards, ranking_ranks = ranking_result
                    
                    # Add retrieval and ranking rewards to total
                    # Note: rewards are per generation, need to map back to batch
                    retrieval_rewards_list = retrieval_rewards.tolist()
                    ranking_rewards_list = ranking_rewards.tolist()
                    ranking_ranks_list = ranking_ranks.tolist()  # ranking_ranks is already set correctly for both cases
                    
                    # Calculate ranking top1 ratio (proportion of samples with rank == 1)
                    top1_count = sum(1 for rank in ranking_ranks_list if rank == 1)
                    ranking_top1_ratio = top1_count / len(ranking_ranks_list) if ranking_ranks_list else 0.0
                    # np.array(scalar) creates 0D array, which causes IndexError in chunk()
                    # Repeat the value batch_size times so it can be properly chunked
                    reward_extra_info["ranking_top1_ratio"] = [ranking_top1_ratio] * batch_size
                    reward_extra_info["ranking_ranks_all"] = ranking_ranks_list
                    
                    if num_queries >= 1:
                        import random
                        # Sample one query per step (use step number from meta_info if available)
                        step_num = 0
                        if hasattr(data[0], 'meta_info') and data[0].meta_info:
                            step_num = data[0].meta_info.get("global_steps", data[0].meta_info.get("step", 0))
                        # Use step_num as seed to ensure different query per step
                        random.seed(step_num)
                        sampled_query_idx = random.randint(0, num_queries - 1)
                        
                        # Get all generations for this query
                        start_idx = sampled_query_idx * self.num_generations
                        end_idx = start_idx + self.num_generations
                        
                        # Find best generation (highest total reward, which includes all reward components)
                        # We need to compute total reward for each generation
                        best_gen_idx = 0
                        best_total_r = -float('inf')
                        for gen_idx in range(self.num_generations):
                            sample_idx = start_idx + gen_idx
                            if sample_idx < batch_size:
                                # Get all reward components for this generation
                                format_r = reward_extra_info["format_reward"][sample_idx] if sample_idx < len(reward_extra_info["format_reward"]) else 0.0
                                reranker_r = reward_extra_info["reranker_reward"][sample_idx] if sample_idx < len(reward_extra_info.get("reranker_reward", [])) else 0.0
                                retrieval_r = retrieval_rewards_list[sample_idx] if sample_idx < len(retrieval_rewards_list) else 0.0
                                ranking_r = ranking_rewards_list[sample_idx] if sample_idx < len(ranking_rewards_list) else 0.0
                                
                                # Calculate total reward
                                total_r = (
                                    self.format_weight * format_r +
                                    self.reranker_reward_weight * reranker_r +
                                    self.retrieval_weight * retrieval_r +
                                    self.ranking_weight * ranking_r
                                )
                                
                                if total_r > best_total_r:
                                    best_total_r = total_r
                                    best_gen_idx = gen_idx
                        
                        best_sample_idx = start_idx + best_gen_idx
                        
                        # Get data for sampled query (use best_sample_idx to match reranker_score)
                        # All generations of the same query should have the same original_query and pos_text
                        if best_sample_idx < batch_size:
                            data_item = data[best_sample_idx]
                            ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", {})
                            
                            # Get original query
                            original_query = ground_truth.get("query", "")
                            if isinstance(original_query, (list, tuple)):
                                original_query = str(original_query[-1]) if original_query else ""
                            else:
                                original_query = str(original_query)
                            
                            # Get positive passage
                            pos_text = ground_truth.get("pos_text", "")
                            if not pos_text:
                                pos_passages = ground_truth.get("pos", [])
                                if isinstance(pos_passages, list) and len(pos_passages) > 0:
                                    pos_text = pos_passages[0] if isinstance(pos_passages[0], str) else str(pos_passages[0])
                                elif isinstance(pos_passages, str):
                                    pos_text = pos_passages
                            
                            # Get best rewritten query
                            best_rewritten_query = reward_extra_info["rewritten_query"][best_sample_idx] if best_sample_idx < len(reward_extra_info["rewritten_query"]) else ""
                            
                            # Get rewards for best generation
                            best_format_r = reward_extra_info["format_reward"][best_sample_idx] if best_sample_idx < len(reward_extra_info["format_reward"]) else 0.0
                            best_retrieval_r = retrieval_rewards_list[best_sample_idx] if best_sample_idx < len(retrieval_rewards_list) else 0.0
                            best_ranking_r = ranking_rewards_list[best_sample_idx] if best_sample_idx < len(ranking_rewards_list) else 0.0
                            best_reranker_r = reward_extra_info["reranker_reward"][best_sample_idx] if best_sample_idx < len(reward_extra_info.get("reranker_reward", [])) else 0.0
                            best_format_correct = reward_extra_info["format_correct"][best_sample_idx] if best_sample_idx < len(reward_extra_info["format_correct"]) else False
                            best_ranking_rank = ranking_ranks_list[best_sample_idx] if best_sample_idx < len(ranking_ranks_list) else -1
                            best_total_r = (
                                self.format_weight * best_format_r +
                                self.reranker_reward_weight * best_reranker_r +
                                self.retrieval_weight * best_retrieval_r +
                                self.ranking_weight * best_ranking_r
                            )
                            
                            # Get similarity scores from retrieval_scores if available
                            retrieval_scores_item = None
                            # Try multiple ways to get retrieval_scores_item
                            # Method 1: Try from best_sample_idx's data item
                            if best_sample_idx < batch_size:
                                data_item_for_scores = data[best_sample_idx]
                                retrieval_scores_raw = data_item_for_scores.non_tensor_batch.get("retrieval_scores", None)
                                if retrieval_scores_raw is not None:
                                    # Case 1: retrieval_scores is a numpy array of dicts
                                    if isinstance(retrieval_scores_raw, np.ndarray):
                                        if best_sample_idx < len(retrieval_scores_raw) and retrieval_scores_raw[best_sample_idx] is not None:
                                            retrieval_scores_item = retrieval_scores_raw[best_sample_idx]
                                    # Case 2: retrieval_scores is a dict with sample indices as keys
                                    elif isinstance(retrieval_scores_raw, dict):
                                        if best_sample_idx in retrieval_scores_raw:
                                            retrieval_scores_item = retrieval_scores_raw[best_sample_idx]
                                        # Case 3: retrieval_scores_raw is directly the dict we need
                                        elif "ori_sim_scores" in retrieval_scores_raw:
                                            retrieval_scores_item = retrieval_scores_raw
                            
                            # Method 2: Fallback - try from start_idx's data item
                            if retrieval_scores_item is None and start_idx < batch_size:
                                data_item_for_scores = data[start_idx]
                                retrieval_scores_raw = data_item_for_scores.non_tensor_batch.get("retrieval_scores", None)
                                if retrieval_scores_raw is not None:
                                    if isinstance(retrieval_scores_raw, np.ndarray):
                                        if best_sample_idx < len(retrieval_scores_raw) and retrieval_scores_raw[best_sample_idx] is not None:
                                            retrieval_scores_item = retrieval_scores_raw[best_sample_idx]
                                    elif isinstance(retrieval_scores_raw, dict):
                                        if best_sample_idx in retrieval_scores_raw:
                                            retrieval_scores_item = retrieval_scores_raw[best_sample_idx]
                                        elif "ori_sim_scores" in retrieval_scores_raw:
                                            retrieval_scores_item = retrieval_scores_raw
                            
                            # Method 3: Try from top-level data.non_tensor_batch (if retrieval_scores is stored there)
                            if retrieval_scores_item is None and hasattr(data, 'non_tensor_batch'):
                                retrieval_scores_raw = data.non_tensor_batch.get("retrieval_scores", None)
                                if retrieval_scores_raw is not None:
                                    if isinstance(retrieval_scores_raw, np.ndarray):
                                        if best_sample_idx < len(retrieval_scores_raw) and retrieval_scores_raw[best_sample_idx] is not None:
                                            retrieval_scores_item = retrieval_scores_raw[best_sample_idx]
                                    elif isinstance(retrieval_scores_raw, dict):
                                        if best_sample_idx in retrieval_scores_raw:
                                            retrieval_scores_item = retrieval_scores_raw[best_sample_idx]
                            
                            # Print detailed log
                            print(f"\n{'='*80}")
                            print(f"[RewardManager] Step {step_num} - Sampled Query {sampled_query_idx} (Best Generation {best_gen_idx})")
                            print(f"{'='*80}")
                            
                            # Original Query
                            print(f"\nOriginal Query")
                            print(original_query)
                            
                            # Rewritten Query
                            print(f"\nRewritten Query(Best, Gen {best_gen_idx})")
                            print(best_rewritten_query)
                            
                            # Positive Passage
                            print(f"\nPositive Passage")
                            pos_text_str = str(pos_text) if pos_text else "N/A"
                            print(f"{pos_text_str}")
                            
                            # Scores (with weighted values)
                            weighted_format_r = self.format_weight * best_format_r
                            weighted_reranker_r = self.reranker_reward_weight * best_reranker_r
                            weighted_retrieval_r = self.retrieval_weight * best_retrieval_r
                            weighted_ranking_r = self.ranking_weight * best_ranking_r
                            
                            print(f"\n")
                            print(f"  Format Reward: {best_format_r:.4f} (weight={self.format_weight:.2f}, weighted={weighted_format_r:.4f}, correct={best_format_correct})")
                            if self.reranker_reward_weight > 0:
                                print(f"  Reranker Reward: {best_reranker_r:.4f} (weight={self.reranker_reward_weight:.2f}, weighted={weighted_reranker_r:.4f})")
                            print(f"  Retrieval Reward: {best_retrieval_r:.4f} (weight={self.retrieval_weight:.2f}, weighted={weighted_retrieval_r:.4f})")
                            print(f"  Ranking Reward: {best_ranking_r:.4f} (rank={best_ranking_rank}, weight={self.ranking_weight:.2f}, weighted={weighted_ranking_r:.4f})")
                            print(f"  Total Reward: {best_total_r:.4f}")
                            
                            # Reranker Scores (if available)
                            if self.reranker_reward_weight > 0:
                                best_reranker_original_score = reward_extra_info.get("reranker_original_score", [])
                                best_reranker_rewritten_score = reward_extra_info.get("reranker_rewritten_score", [])
                                
                                if best_sample_idx < len(best_reranker_original_score) and best_sample_idx < len(best_reranker_rewritten_score):
                                    orig_reranker_score = best_reranker_original_score[best_sample_idx]
                                    rew_reranker_score = best_reranker_rewritten_score[best_sample_idx]
                                    
                                    print(f"\nReranker Score")
                                    print(f" origin Query vs Pos: {orig_reranker_score:.4f}")
                                    print(f"  rewrite Query vs Pos: {rew_reranker_score:.4f}")
                                    reranker_improvement = rew_reranker_score - orig_reranker_score
                                    print(f"  improvement: {reranker_improvement:+.4f}")
                            
                            # Retrieval Similarity Scores (if available)
                            if retrieval_scores_item is not None and isinstance(retrieval_scores_item, dict):
                                ori_sim_scores = retrieval_scores_item.get("ori_sim_scores", None)
                                rewritten_sim_scores = retrieval_scores_item.get("rewritten_sim_scores", None)
                                num_pos_passages = retrieval_scores_item.get("num_pos_passages", 1)
                                
                                if ori_sim_scores is not None and rewritten_sim_scores is not None:
                                    ori_sim = np.array(ori_sim_scores)
                                    rewritten_sim = np.array(rewritten_sim_scores)
                                    
                                    if len(ori_sim) > 0 and len(rewritten_sim) > 0:
                                        print(f"\nRetrieval Similarity Score(Temperature={self.temperature:.3f})")
                                        
                                        if num_pos_passages > 0 and len(ori_sim) >= num_pos_passages:
                                            ori_pos_scores = ori_sim[:num_pos_passages]
                                            rew_pos_scores = rewritten_sim[:num_pos_passages]
                                            print(f" origin Query vs Pos: mean={np.mean(ori_pos_scores):.4f}, max={np.max(ori_pos_scores):.4f}, min={np.min(ori_pos_scores):.4f}")
                                            print(f"  rewrite Query vs Pos: mean={np.mean(rew_pos_scores):.4f}, max={np.max(rew_pos_scores):.4f}, min={np.min(rew_pos_scores):.4f}")
                                            pos_improvement = np.mean(rew_pos_scores) - np.mean(ori_pos_scores)
                                            print(f"  Pos Improvement: {pos_improvement:+.4f}")
                                        
                                        if len(ori_sim) > num_pos_passages:
                                            ori_neg_scores = ori_sim[num_pos_passages:]
                                            rew_neg_scores = rewritten_sim[num_pos_passages:]
                                            print(f"  origin Query vs Neg: mean={np.mean(ori_neg_scores):.4f}, max={np.max(ori_neg_scores):.4f}, min={np.min(ori_neg_scores):.4f}")
                                            print(f"  rewrite Query vs Neg: mean={np.mean(rew_neg_scores):.4f}, max={np.max(rew_neg_scores):.4f}, min={np.min(rew_neg_scores):.4f}")
                                            neg_improvement = np.mean(rew_neg_scores) - np.mean(ori_neg_scores)
                                            print(f"  Neg improvement: {neg_improvement:+.4f}")
                                        
                                        # Overall improvement
                                        if len(ori_sim) > 0:
                                            overall_ori = np.mean(ori_sim)
                                            overall_rew = np.mean(rewritten_sim)
                                            overall_improvement = overall_rew - overall_ori
                                            print(f"  overall: {overall_improvement:+.4f} (ori_mean={overall_ori:.4f}, rew_mean={overall_rew:.4f})")
                            else:
                                # Debug: print why retrieval scores are not available
                                if step_num % 10 == 0:  # Only print every 10 steps to avoid spam
                                    print(f"\n[RewardManager] Debug: retrieval_scores_item is None (best_sample_idx={best_sample_idx}, batch_size={batch_size})")
                                    if best_sample_idx < batch_size:
                                        data_item_debug = data[best_sample_idx]
                                        has_retrieval_scores = "retrieval_scores" in data_item_debug.non_tensor_batch
                                        print(f"  data[{best_sample_idx}].non_tensor_batch has 'retrieval_scores': {has_retrieval_scores}")
                                        if has_retrieval_scores:
                                            rs_raw = data_item_debug.non_tensor_batch.get("retrieval_scores", None)
                                            print(f"  retrieval_scores type: {type(rs_raw)}")
                                            if isinstance(rs_raw, np.ndarray):
                                                print(f"  retrieval_scores array length: {len(rs_raw)}")
                                            elif isinstance(rs_raw, dict):
                                                print(f"  retrieval_scores dict keys: {list(rs_raw.keys())[:5]}...")
                            
                            # Batch Statistics (summary)
                            print(f"\nBatch Statistics")
                            print(f"  Retrieval Reward: min={min(retrieval_rewards_list):.3f}, max={max(retrieval_rewards_list):.3f}, mean={sum(retrieval_rewards_list)/len(retrieval_rewards_list):.3f}")
                            print(f"  Ranking Top1 Ratio: {ranking_top1_ratio:.3f} ({top1_count}/{len(ranking_ranks_list)})")
                            
                            print(f"{'='*80}\n")
                    
                    for i in range(batch_size):
                        valid_response_length = data[i].batch["attention_mask"][data[i].batch["prompts"].shape[-1]:].sum()
                        # retrieval_rewards is [batch_size] where batch_size = num_queries * num_generations
                        # Each element i corresponds to the reward for sample i
                        if i < len(retrieval_rewards):
                            retrieval_r = retrieval_rewards[i].item()
                            ranking_r = ranking_rewards[i].item() if i < len(ranking_rewards) else 0.0
                            
                            # Add weighted rewards to total
                            reward_tensor[i, valid_response_length - 1] += (
                                self.retrieval_weight * retrieval_r +
                                self.ranking_weight * ranking_r
                            )
                            
                            # Store individual rewards for tracking
                            reward_extra_info["retrieval_reward"].append(retrieval_r)
                            reward_extra_info["ranking_reward"].append(ranking_r)
                            
                            # Get format and reranker rewards for this sample
                            format_r = reward_extra_info["format_reward"][i] if i < len(reward_extra_info["format_reward"]) else 0.0
                            reranker_r = reward_extra_info["reranker_reward"][i] if i < len(reward_extra_info.get("reranker_reward", [])) else 0.0
                            
                            # Calculate total reward (for best rewrite selection)
                            total_reward = (
                                self.format_weight * format_r +
                                self.reranker_reward_weight * reranker_r +
                                self.retrieval_weight * retrieval_r +
                                self.ranking_weight * ranking_r
                            )
                            
                            extra_info_item = data[i].non_tensor_batch.get("extra_info", {})
                            if not isinstance(extra_info_item, dict):
                                extra_info_item = {}
                            extra_info_item["retrieval_reward"] = retrieval_r
                            extra_info_item["ranking_reward"] = ranking_r
                            extra_info_item["reranker_reward"] = reranker_r
                            extra_info_item["total_reward"] = total_reward.item() if isinstance(total_reward, torch.Tensor) else total_reward
                            data[i].non_tensor_batch["extra_info"] = extra_info_item
                        else:
                            reward_extra_info["retrieval_reward"].append(0.0)
                            reward_extra_info["ranking_reward"].append(0.0)
                            
                            # Get format and reranker rewards for this sample
                            format_r = reward_extra_info["format_reward"][i] if i < len(reward_extra_info["format_reward"]) else 0.0
                            reranker_r = reward_extra_info["reranker_reward"][i] if i < len(reward_extra_info.get("reranker_reward", [])) else 0.0
                            
                            # Calculate total reward (for best rewrite selection)
                            total_reward = (
                                self.format_weight * format_r +
                                self.reranker_reward_weight * reranker_r +
                                self.retrieval_weight * 0.0 +
                                self.ranking_weight * 0.0
                            )
                            
                            extra_info_item = data[i].non_tensor_batch.get("extra_info", {})
                            if not isinstance(extra_info_item, dict):
                                extra_info_item = {}
                            extra_info_item["retrieval_reward"] = 0.0
                            extra_info_item["ranking_reward"] = 0.0
                            extra_info_item["reranker_reward"] = reranker_r
                            extra_info_item["total_reward"] = total_reward.item() if isinstance(total_reward, torch.Tensor) else total_reward
                            data[i].non_tensor_batch["extra_info"] = extra_info_item
                    
                    reward_extra_info["retrieval_rewards_all"] = retrieval_rewards_list
                    reward_extra_info["ranking_rewards_all"] = ranking_rewards_list
                    reward_extra_info["ranking_ranks_all"] = ranking_ranks_list
                    reward_extra_info["ranking_top1_ratio"] = [ranking_top1_ratio] * batch_size
                
            except Exception as e:
                import traceback
                print(f"[GapGRPOV3RewardManager] Warning: Failed to compute retrieval reward: {e}")
                print(traceback.format_exc())
                for i in range(batch_size):
                    valid_response_length = data[i].batch["attention_mask"][data[i].batch["prompts"].shape[-1]:].sum()
                    format_correct = reward_extra_info["format_correct"][i]
                    bonus = 0.25 if format_correct else 0.0
                    reward_tensor[i, valid_response_length - 1] += (
                        self.retrieval_weight * bonus
                    )
                    reward_extra_info["retrieval_reward"].append(0.0)
                    reward_extra_info["ranking_reward"].append(0.0)
                    
                    # Get format and reranker rewards for this sample
                    format_r = reward_extra_info["format_reward"][i] if i < len(reward_extra_info["format_reward"]) else 0.0
                    reranker_r = reward_extra_info["reranker_reward"][i] if i < len(reward_extra_info.get("reranker_reward", [])) else 0.0
                    
                    # Calculate total reward (for best rewrite selection)
                    total_reward = (
                        self.format_weight * format_r +
                        self.reranker_reward_weight * reranker_r +
                        self.retrieval_weight * 0.0 +
                        self.ranking_weight * 0.0
                    )
                    
                    extra_info_item = data[i].non_tensor_batch.get("extra_info", {})
                    if not isinstance(extra_info_item, dict):
                        extra_info_item = {}
                    extra_info_item["retrieval_reward"] = 0.0
                    extra_info_item["ranking_reward"] = 0.0
                    extra_info_item["reranker_reward"] = reranker_r
                    extra_info_item["total_reward"] = total_reward.item() if isinstance(total_reward, torch.Tensor) else total_reward
                    data[i].non_tensor_batch["extra_info"] = extra_info_item
        else:
            for i in range(batch_size):
                valid_response_length = data[i].batch["attention_mask"][data[i].batch["prompts"].shape[-1]:].sum()
                format_correct = reward_extra_info["format_correct"][i]
                bonus = 0.25 if format_correct else 0.0
                reward_tensor[i, valid_response_length - 1] += (
                    self.retrieval_weight * bonus
                )
                reward_extra_info["retrieval_reward"].append(0.0)
                reward_extra_info["ranking_reward"].append(0.0)
                
                # Get format and reranker rewards for this sample
                format_r = reward_extra_info["format_reward"][i] if i < len(reward_extra_info["format_reward"]) else 0.0
                reranker_r = reward_extra_info["reranker_reward"][i] if i < len(reward_extra_info.get("reranker_reward", [])) else 0.0
                
                # Calculate total reward (for best rewrite selection)
                total_reward = (
                    self.format_weight * format_r +
                    self.reranker_reward_weight * reranker_r +
                    self.retrieval_weight * 0.0 +
                    self.ranking_weight * 0.0
                )
                
                extra_info_item = data[i].non_tensor_batch.get("extra_info", {})
                if not isinstance(extra_info_item, dict):
                    extra_info_item = {}
                extra_info_item["retrieval_reward"] = 0.0
                extra_info_item["ranking_reward"] = 0.0
                extra_info_item["reranker_reward"] = reranker_r
                extra_info_item["total_reward"] = total_reward.item() if isinstance(total_reward, torch.Tensor) else total_reward
                data[i].non_tensor_batch["extra_info"] = extra_info_item
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        
        return reward_tensor
