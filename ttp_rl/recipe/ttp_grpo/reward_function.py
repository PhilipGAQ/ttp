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
GAP-GRPO Reward Function

Implements two types of rewards:
1. Format Reward: Check if model output follows <think>rewrite query</think><emb> format
2. Rewrite Gain Reward: Reward improvement in query-doc relevance after rewriting
   - Uses different scales for positive gain (gain_scale) and negative gain (penalty_scale)

The reward model uses a BERT-based reranker model to compute query-doc relevance scores.
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import ray
import torch.nn.functional as F
from verl.utils.logger import print_rank_0

# Import Reranker classes from separate file to avoid serialization issues
try:
    from recipe.ttp_grpo.reranker import (
        RewriteRewardModel,
        RayRewriteRewardModel,
        preprocess_query_for_reranker,
        get_reranker_model_path,
        set_reranker_model_path,
        get_ray_reranker_actor
    )
except ImportError:
    # Fallback for relative import if running as module
    from .reranker import (
        RewriteRewardModel,
        RayRewriteRewardModel,
        preprocess_query_for_reranker,
        get_reranker_model_path,
        set_reranker_model_path,
        get_ray_reranker_actor
    )

logger = logging.getLogger(__name__)
STAGE_LOG_ENABLED = os.environ.get("GAP_STAGE_LOG", "1") not in {"0", "false", "False"}
# Extra debug print flag (stdout), to guarantee visibility in Ray worker logs
DEBUG_LOG_ENABLED = os.environ.get("GAP_DEBUG_LOG", "0") not in {"0", "false", "False"}
# Extra prompt debug flag: dump prompt + output + rewards for first sample in batch
DEBUG_PROMPT_ENABLED = os.environ.get("GAP_DEBUG_PROMPT", "0") not in {"0", "false", "False"}


def log_stage(message: str):
    # Disabled for cleaner logs
    pass


# ================== Format Reward ==================

def compute_format_reward(
    response_str: str,
    think_open: str = "<think>",
    think_close: str = "</think>",
    emb_tok: str = "<emb>",
    format_reward: float = 0.5,
    format_penalty: float = -0.5,
    length_threshold: int = 30,
    length_penalty: float = -0.1,
    repetition_penalty: float = -0.2,
    original_query: Optional[str] = None,
    containment_penalty: float = -0.5,
) -> Tuple[float, Dict[str, Any]]:
    """
    Check if the response follows the expected format:
    <think>rewritten query content</think><emb>
    
    Template reward policy:
    - Template correct: no reward (0.0)
    - Template error: format_penalty
    - Template repetition: repetition_penalty (repeated special tokens)
    
    Length penalty:
    - If output length > length_threshold: apply length_penalty
    
    Args:
        response_str: The model's generated response
        think_open: Opening think tag
        think_close: Closing think tag
        emb_tok: Embedding token
        format_reward: Reward for correct format (deprecated, now returns 0.0)
        format_penalty: Penalty for incorrect format
        length_threshold: Length threshold for penalty
        length_penalty: Penalty per excess length (if length > threshold)
        repetition_penalty: Penalty for repeated special tokens
    
    Returns:
        Tuple of (reward, info_dict)
    """
    info = {
        "has_think_open": think_open in response_str,
        "has_think_close": think_close in response_str,
        "has_emb_token": emb_tok in response_str,
        "format_correct": False,
        "rewritten_query": "",
        "length": len(response_str),
        "has_repetition": False,
        # New fields for query containment check
        "containment_missing_chars": 0,
        "containment_total_chars": 0,
        "containment_missing_ratio": 0.0,
        "containment_penalty": 0.0,
    }
    
    reward = 0.0
    
    # Check format: <think>...</think><emb>
    pattern = rf"{re.escape(think_open)}(.+?){re.escape(think_close)}\s*{re.escape(emb_tok)}"
    match = re.search(pattern, response_str, re.DOTALL)
    
    if match:
        info["format_correct"] = True
        rewritten_query = match.group(1).strip()
        info["rewritten_query"] = rewritten_query
        
        # Template correct: no reward (0.0)
        reward = 0.0
        
        # Check for repetition of special tokens
        # Each special token should appear exactly once (think_open, think_close, emb_tok)
        special_tokens = [think_open, think_close, emb_tok]
        for token in special_tokens:
            count = response_str.count(token)
            if count > 1:
                info["has_repetition"] = True
                reward += repetition_penalty
                break
    else:
        # Template error: apply penalty
        reward = format_penalty
    
    # Length penalty: if output length > threshold
    output_length = len(response_str)
    info["length"] = output_length
    if output_length > length_threshold:
        excess_length = output_length - length_threshold
        # Apply penalty proportional to excess length
        length_penalty_value = length_penalty * (excess_length / length_threshold)
        reward += length_penalty_value
        info["length_penalty"] = length_penalty_value
    else:
        info["length_penalty"] = 0.0

    # ================== Query containment penalty ==================
    # Only check content within Chinese parentheses "（）" in the original query.
    # If the original query has content in "（）", the rewritten query should also contain it.
    #
    # Implementation detail:
    # - Extract all content within Chinese parentheses "（）" from original query
    # - Check if rewritten query contains each extracted content
    # - Penalty is proportional to missing content ratio:
    #       missing_ratio = (# of missing parentheses content) / (# of total parentheses content)
    #       reward += containment_penalty * missing_ratio
    #
    # This keeps the penalty in the "format reward" component.
    if original_query is not None and info.get("rewritten_query"):
        try:
            cleaned_original = preprocess_query_for_reranker(original_query)
            cleaned_rewritten = str(info["rewritten_query"])

            # Extract content within Chinese parentheses "（）"
            # Pattern: match content between "（" and "）"
            parentheses_pattern = r'（([^）]+)）'
            parentheses_contents = re.findall(parentheses_pattern, cleaned_original)
            
            if parentheses_contents:
                # Check if rewritten query contains each parentheses content
                missing_count = 0
                for content in parentheses_contents:
                    # Remove whitespace for comparison
                    content_clean = ''.join(content.split())
                    rewritten_clean = ''.join(cleaned_rewritten.split())
                    
                    # Check if rewritten query contains the parentheses content
                    if content_clean not in rewritten_clean:
                        missing_count += 1

                missing_ratio = missing_count / len(parentheses_contents)
                # Apply proportional penalty (containment_penalty should be negative)
                containment_penalty_value = containment_penalty * missing_ratio
                reward += containment_penalty_value

                info["containment_missing_chars"] = missing_count
                info["containment_total_chars"] = len(parentheses_contents)
                info["containment_missing_ratio"] = missing_ratio
                info["containment_penalty"] = containment_penalty_value
            else:
                # No Chinese parentheses in original query, no penalty
                info["containment_missing_chars"] = 0
                info["containment_total_chars"] = 0
                info["containment_missing_ratio"] = 0.0
                info["containment_penalty"] = 0.0
        except Exception as _e:
            # Fail-safe: do not break reward computation if preprocessing fails
            info["containment_penalty"] = 0.0
    
    return reward, info


# ================== Rewrite Gain Reward ==================

def compute_rewrite_gain_reward(
    original_query: str,
    rewritten_query: str,
    positive_doc: str,
    reward_model: Optional[RewriteRewardModel] = None,
    gain_scale: float = 1.0,
    penalty_scale: float = 0.5,  # 当相似度降低时的惩罚系数，默认0.5表示惩罚减半
    min_reward: float = -1.0,
    max_reward: float = 1.0,
    reward_model_path: Optional[str] = None,  # If None, uses global setting
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute reward based on the gain in relevance after query rewriting.
    
    Reward = gain_scale * (score(rewritten, pos) - score(original, pos))  if gain >= 0
    Reward = penalty_scale * (score(rewritten, pos) - score(original, pos))  if gain < 0
    
    Query preprocessing:
    - Remove instruction prefix
    - Apply query.split('geohash')[0].strip() to remove geohash suffix
    
    Args:
        original_query: Original user query (will be preprocessed)
        rewritten_query: Rewritten query from model generation
        positive_doc: Positive document
        reward_model: Pre-initialized reward model (optional)
        gain_scale: Scaling factor for positive gain (when similarity improves)
        penalty_scale: Scaling factor for negative gain (when similarity decreases), default 0.5
        min_reward: Minimum reward value
        max_reward: Maximum reward value
        reward_model_path: Path to the reward model
    
    Returns:
        Tuple of (reward, info_dict)
    """
    if reward_model is None:
        reward_model = RewriteRewardModel.get_instance(model_name_or_path=reward_model_path)
    
    # Preprocess original query (remove instruction, geohash suffix)
    cleaned_original_query = preprocess_query_for_reranker(original_query)
    
    # Rewritten query is already clean (from model generation)
    cleaned_rewritten_query = str(rewritten_query).strip()
    
    # Preprocess positive doc
    if isinstance(positive_doc, (list, tuple)):
        cleaned_positive_doc = str(positive_doc[-1]) if positive_doc else ""
    else:
        cleaned_positive_doc = str(positive_doc).strip()
    
    # Use batch processing for efficiency (2 queries with same doc)
    queries = [cleaned_original_query, cleaned_rewritten_query]
    docs = [cleaned_positive_doc, cleaned_positive_doc]
    
    scores = reward_model.compute_batch_scores(queries, docs)
    original_score = scores[0]
    rewritten_score = scores[1]
    
    # Compute gain
    gain = rewritten_score - original_score
    
    # 如果相似度降低（gain < 0），使用较小的惩罚系数
    if gain < 0:
        reward = penalty_scale * gain
    else:
        reward = gain_scale * gain
    
    # Clip reward
    reward = max(min_reward, min(max_reward, reward))
    
    info = {
        "original_query_clean": cleaned_original_query,
        "rewritten_query_clean": cleaned_rewritten_query,
        "positive_doc_len": len(cleaned_positive_doc),
        "original_score": original_score,
        "rewritten_score": rewritten_score,
        "gain": gain,
        "rewrite_gain_reward": reward,
    }
    
    return reward, info


# ================== Main Compute Score Function ==================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None,
    # Reward weights
    format_weight: float = 0.2,
    rewrite_gain_weight: float = 0.8,
    # Format reward params
    format_reward: float = 0.5,  # Deprecated: correct format now returns 0.0
    format_penalty: float = -0.5,
    length_threshold: int = 30,
    length_penalty: float = -0.1,
    repetition_penalty: float = -0.2,
    containment_penalty: float = -0.5,
    # Rewrite gain params
    reward_model_path: Optional[str] = None,  # If None, uses global setting
    gain_scale: float = 1.0,
    penalty_scale: float = 0.5,  # 当相似度降低时的惩罚系数
) -> Dict[str, Any]:
    """
    Main reward computation function for GAP-GRPO.
    
    Combines two types of rewards:
    1. Format Reward (format_weight): Correct <think>...</think><emb> format
    2. Rewrite Gain Reward (rewrite_gain_weight): Query improvement based on reranker scores
    
    Args:
        data_source: Data source identifier (should be "gap_grpo")
        solution_str: Model's generated response
        ground_truth: Dict containing query, query_gen, pos, neg
        extra_info: Additional information
        format_weight: Weight for format reward
        rewrite_gain_weight: Weight for rewrite gain reward
        ...other params...
    
    Returns:
        Dict with 'score' key and additional metrics
    """
    if data_source != "gap_grpo":
        raise ValueError(f"Unsupported data source: {data_source}")
    
    extra_info = extra_info or {}
    
    # Extract ground truth - keep original format for preprocessing later
    original_query = ground_truth.get("query", "")
    
    positive_doc = ground_truth.get("pos", "")
    
    # 1. Compute format reward (includes length, repetition, and containment penalties)
    format_r, format_info = compute_format_reward(
        response_str=solution_str,
        format_reward=format_reward,
        format_penalty=format_penalty,
        length_threshold=length_threshold,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        original_query=original_query,
        containment_penalty=containment_penalty,
    )
    
    # Extract rewritten query from format check
    rewritten_query = format_info.get("rewritten_query", "")
    if not rewritten_query:
        # If format is incorrect, use the whole response as rewritten query
        rewritten_query = solution_str.strip()
    
    # 2. Compute rewrite gain reward
    rewrite_r = 0.0
    rewrite_info = {}
    if positive_doc and rewritten_query:
        try:
            # Use provided path or global setting
            actual_model_path = reward_model_path or get_reranker_model_path()
            rewrite_r, rewrite_info = compute_rewrite_gain_reward(
                original_query=original_query,
                rewritten_query=rewritten_query,
                positive_doc=positive_doc,
                reward_model_path=actual_model_path,
                gain_scale=gain_scale,
                penalty_scale=penalty_scale,
            )
        except Exception as e:
            print(f"[compute_score] Warning: Failed to compute rewrite gain reward: {e}")
            rewrite_r = 0.0
            rewrite_info = {"error": str(e)}
    
    # Combine rewards (removed retrieval reward)
    total_reward = (
        format_weight * format_r +
        rewrite_gain_weight * rewrite_r
    )
    
    # Get clean query for display
    display_original_query = preprocess_query_for_reranker(original_query)
    display_positive_doc = str(positive_doc[-1]) if isinstance(positive_doc, (list, tuple)) else str(positive_doc)
    
    result = {
        "score": total_reward,
        # Individual rewards
        "format_reward": format_r,
        "rewrite_gain_reward": rewrite_r,
        # Weights
        "format_weight": format_weight,
        "rewrite_gain_weight": rewrite_gain_weight,
        # Detailed info
        "format_correct": format_info.get("format_correct", False),
        "rewritten_query": rewritten_query,
        "original_query": display_original_query,
        "positive_doc": display_positive_doc,
        # Additional metrics
        **{f"format_{k}": v for k, v in format_info.items()},
        **{f"rewrite_{k}": v for k, v in rewrite_info.items()},
    }
    
    return result


# ================== Batch Compute Score Function ==================

def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Dict[str, Any]],
    extra_infos: Optional[List[Dict[str, Any]]] = None,
    # Reward weights
    format_weight: float = 0.2,
    rewrite_gain_weight: float = 0.8,
    # Format reward params
    format_reward: float = 0.5,  # Deprecated: correct format now returns 0.0
    format_penalty: float = -0.5,
    length_threshold: int = 30,
    length_penalty: float = -0.1,
    repetition_penalty: float = -0.2,
    containment_penalty: float = -0.5,
    # Rewrite gain params
    reward_model_path: Optional[str] = None,
    gain_scale: float = 1.0,
    penalty_scale: float = 0.5,  # 当相似度降低时的惩罚系数
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Batch reward computation function for GAP-GRPO.
    
    This function is designed for use with BatchRewardManager.
    It computes rewards for a batch of samples with GPU-accelerated reranker.
    
    Args:
        data_sources: List of data source identifiers
        solution_strs: List of model generated responses
        ground_truths: List of ground truth dicts
        extra_infos: List of extra info dicts
        ...other params...
    
    Returns:
        List of result dicts with 'score' key and metrics
    """
    batch_size = len(solution_strs)
    # extra_infos 可能是 None、list 或 numpy 数组，不能直接用 `or` 判断真值
    if extra_infos is None:
        extra_infos = [{}] * batch_size
    else:
        # 将 numpy.array 等统一转成 Python list，方便后续按索引取用
        if hasattr(extra_infos, "tolist"):
            extra_infos = list(extra_infos.tolist())
        elif not isinstance(extra_infos, list):
            extra_infos = list(extra_infos)
    
    # Check if rewrite gain reward is disabled (skip reranker entirely)
    skip_reranker = (rewrite_gain_weight == 0 or rewrite_gain_weight is None)
    
    log_stage(f"Start reward batch (batch_size={batch_size}, skip_reranker={skip_reranker})")

    # Only load reward model if needed
    reward_model = None
    use_ray_actor = False
    
    if not skip_reranker:
        actual_model_path = reward_model_path or get_reranker_model_path()
        
        # Determine execution mode: Ray Actor (GPU) vs Local (CPU/GPU)
        # We prefer Ray Actor if Ray is initialized to share GPU resource persistently
        if ray.is_initialized():
            use_ray_actor = True
            # Note: We'll retrieve the actor later when needed to minimize startup race conditions
        else:
            reward_model = RewriteRewardModel.get_instance(model_name_or_path=actual_model_path)
    
    # Step 1: Compute format rewards for all samples (CPU, fast)
    format_results = []
    rewritten_queries = []
    original_queries = []
    positive_docs = []
    
    for i in range(batch_size):
        solution_str = solution_strs[i]
        ground_truth = ground_truths[i] or {}
        
        # Extract original query and positive doc first (for format reward containment check)
        original_query = ground_truth.get("query", "")
        positive_doc = ground_truth.get("pos", "")
        
        # Compute format reward (includes length, repetition, and containment penalties)
        format_r, format_info = compute_format_reward(
            response_str=solution_str,
            format_reward=format_reward,
            format_penalty=format_penalty,
            length_threshold=length_threshold,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            original_query=original_query,
            containment_penalty=containment_penalty,
        )
        format_results.append((format_r, format_info))
        
        # Extract rewritten query
        rewritten_query = format_info.get("rewritten_query", "")
        if not rewritten_query:
            rewritten_query = solution_str.strip()
        rewritten_queries.append(rewritten_query)
        
        # Store original query and positive doc for later rewrite gain computation
        original_queries.append(original_query)
        positive_docs.append(positive_doc)
    
    log_stage("Format reward stage completed")
    # Step 2: Batch compute rewrite gain rewards (GPU, batch)
    # Skip entirely if rewrite_gain_weight=0
    rewrite_rewards = [0.0] * batch_size
    # Initialize rewrite_infos with default structure to ensure all samples have same keys
    # This prevents DataProto consistency errors when some samples don't have reranker scores
    rewrite_infos = [{
        "original_score": 0.0,
        "rewritten_score": 0.0,
        "gain": 0.0,
        "rewrite_gain_reward": 0.0,
    } for _ in range(batch_size)]
    
    if not skip_reranker:
        log_stage("Preparing reranker inputs")
        
        # Prepare queries for batch reranker scoring
        all_queries_for_rerank = []
        all_docs_for_rerank = []
        query_indices = []  # Track which samples have valid reranker input
        
        for i in range(batch_size):
            original_query = original_queries[i]
            rewritten_query = rewritten_queries[i]
            positive_doc = positive_docs[i]
            
            if positive_doc and rewritten_query:
                # Preprocess queries
                cleaned_original = preprocess_query_for_reranker(original_query)
                cleaned_rewritten = str(rewritten_query).strip()
                
                # Handle positive doc format
                if isinstance(positive_doc, (list, tuple)):
                    cleaned_doc = str(positive_doc[-1]) if positive_doc else ""
                else:
                    cleaned_doc = str(positive_doc).strip()
                
                # Validate all inputs are non-empty to avoid tokenizer errors
                if cleaned_doc and cleaned_original and cleaned_rewritten:
                    # Add original query -> doc pair
                    all_queries_for_rerank.append(cleaned_original)
                    all_docs_for_rerank.append(cleaned_doc)
                    # Add rewritten query -> doc pair
                    all_queries_for_rerank.append(cleaned_rewritten)
                    all_docs_for_rerank.append(cleaned_doc)
                    query_indices.append(i)
        
        # Batch compute reranker scores
        if all_queries_for_rerank:
            try:
                if use_ray_actor:
                    log_stage(f"Running reranker via Ray Actor on {len(all_queries_for_rerank)} pairs")
                    
                    # Get or create the actor
                    reranker_actor = get_ray_reranker_actor(actual_model_path)
                    
                    # Remote call
                    t_call_start = time.time()
                    all_scores = ray.get(reranker_actor.compute_batch_scores.remote(
                        all_queries_for_rerank,
                        all_docs_for_rerank,
                    ))
                    t_call_end = time.time()
                    print_rank_0(f"[Reranker] Processed {len(all_queries_for_rerank)} pairs in {t_call_end-t_call_start:.2f}s")
                else:
                    log_stage(
                        f"Running reranker on {len(all_queries_for_rerank)} pairs "
                        f"(preferred_device={reward_model.device}, lazy_move={reward_model.lazy_move_to_device})"
                    )
                    t_local_start = time.time()
                    all_scores = reward_model.compute_batch_scores(
                        all_queries_for_rerank,
                        all_docs_for_rerank,
                    )
                    print_rank_0(f"[Reranker] Processed {len(all_queries_for_rerank)} pairs in {time.time()-t_local_start:.2f}s")
                
                log_stage("Reranker stage completed")
                
                # Parse scores back to samples (every 2 scores belong to one sample)
                for j, sample_idx in enumerate(query_indices):
                    original_score = all_scores[j * 2]
                    rewritten_score = all_scores[j * 2 + 1]
                    
                    gain = rewritten_score - original_score
                    # 如果相似度降低（gain < 0），使用较小的惩罚系数
                    if gain < 0:
                        reward = penalty_scale * gain
                    else:
                        reward = gain_scale * gain
                    reward = max(-1.0, min(1.0, reward))  # Clip
                    
                    rewrite_rewards[sample_idx] = reward
                    rewrite_infos[sample_idx] = {
                        "original_score": original_score,
                        "rewritten_score": rewritten_score,
                        "gain": gain,
                        "rewrite_gain_reward": reward,
                    }
            except Exception as e:
                print(f"[compute_score_batch] Batch reranker failed: {e}")
    
    # Step 3: Combine rewards and build results
    log_stage("Combining reward components")
    results = []
    for i in range(batch_size):
        format_r, format_info = format_results[i]
        rewrite_r = rewrite_rewards[i]
        rewrite_info = rewrite_infos[i]
        
        # Combine rewards (removed retrieval reward)
        total_reward = (
            format_weight * format_r +
            rewrite_gain_weight * rewrite_r
        )
        
        # Build result
        display_original_query = preprocess_query_for_reranker(original_queries[i])
        positive_doc = positive_docs[i]
        display_positive_doc = str(positive_doc[-1]) if isinstance(positive_doc, (list, tuple)) else str(positive_doc)
        
        result = {
            "score": total_reward,
            "format_reward": format_r,
            "rewrite_gain_reward": rewrite_r,
            "format_weight": format_weight,
            "rewrite_gain_weight": rewrite_gain_weight,
            "format_correct": format_info.get("format_correct", False),
            "rewritten_query": rewritten_queries[i],
            "original_query": display_original_query,
            "positive_doc": display_positive_doc,
            **{f"format_{k}": v for k, v in format_info.items()},
            **{f"rewrite_{k}": v for k, v in rewrite_info.items()},
        }
        results.append(result)
    

    # Key sample inspection (first sample only, when enabled)
    if DEBUG_PROMPT_ENABLED and batch_size > 0:
        try:
            idx = 0
            extra = extra_infos[idx] if extra_infos else {}
            prompt_str = str(extra.get("query_prompt", "[N/A]")).replace(os.linesep, "\\n")
            out_str = solution_strs[idx].replace(os.linesep, "\\n")
            fmt_r = format_results[idx][0]
            rwt_r = rewrite_rewards[idx]
            
            print_rank_0(f"[Sample] Input: {prompt_str[:200]}...")
            print_rank_0(f"[Sample] Output: {out_str}")
            print_rank_0(f"[Sample] Rewards: Format={fmt_r:.4f}, RewriteGain={rwt_r:.4f}")
        except Exception as _e:
            pass

    return results
