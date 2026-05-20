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
Utility functions for retrieval reward and nDCG reward computation.
Adapted from o1-embedder-training framework.
"""

from typing import List, Dict
from collections import defaultdict

import torch
import torch.nn.functional as F


def retrieval_reward_map_func(old_rewards: torch.Tensor, device) -> torch.Tensor:
    """
    Map retrieval rewards to a wider range for better training signal.
    Adapted from o1-embedder-training.
    """
    rewards = old_rewards.clone()
    rewards = torch.where(rewards > 0.05, torch.tensor(10.0, device=device), rewards)
    rewards = torch.where((rewards > 0.045) & (rewards <= 0.05), torch.tensor(7.5, device=device), rewards)
    rewards = torch.where((rewards > 0.04) & (rewards <= 0.045), torch.tensor(5.0, device=device), rewards)
    rewards = torch.where((rewards > 0.035) & (rewards <= 0.04), torch.tensor(4.5, device=device), rewards)
    rewards = torch.where((rewards > 0.03) & (rewards <= 0.035), torch.tensor(4.0, device=device), rewards)
    rewards = torch.where((rewards > 0.025) & (rewards <= 0.03), torch.tensor(2.5, device=device), rewards)
    rewards = torch.where((rewards > 0.02) & (rewards <= 0.025), torch.tensor(2.0, device=device), rewards)
    rewards = torch.where((rewards > 0.015) & (rewards <= 0.02), torch.tensor(1.5, device=device), rewards)
    rewards = torch.where((rewards > 0.01) & (rewards <= 0.015), torch.tensor(1.0, device=device), rewards)
    rewards = torch.where((rewards > 0) & (rewards <= 0.01), torch.tensor(0.5, device=device), rewards)
    rewards = torch.where(rewards <= 0, torch.tensor(-3.5, device=device), rewards)
    return rewards


def evaluate_ndcg(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Dict[str, float]:
    """Evaluate nDCG metrics using pytrec_eval.
    
    Args:
        qrels: Ground truth relevance judgments
        results: Search results to evaluate
        k_values: Cutoff values for nDCG
        
    Returns:
        Dictionary mapping metric names to scores
    """
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string})
    scores = evaluator.evaluate(results)

    ndcg_dict = {}
    for k in k_values:
        ndcg_dict[f"ndcg@{k}"] = {
            query_id: scores[query_id]["ndcg_cut_" + str(k)]
            for query_id in scores.keys()
        }

    return ndcg_dict


def get_ndcg_rewards(
    sim_scores: torch.Tensor, 
    num_generations: int, 
    ndcg_value: int = 1
) -> torch.Tensor:
    """
    Compute nDCG metric values as rewards based on similarity scores.
    
    Args:
        sim_scores: [batch_size, batch_size * group_size] similarity scores
        num_generations: Number of generations per query (for GRPO)
        ndcg_value: nDCG cutoff value
        
    Returns:
        nDCG rewards: [batch_size] tensor
    """
    device = sim_scores.device
    batch_size = sim_scores.size(0)
    embedder_group_size = sim_scores.size(1) // sim_scores.size(0)
    
    # Build targets: for each query, mark its positive passages
    cur_targets = torch.zeros(
        (batch_size, batch_size * embedder_group_size),
        device=device,
        dtype=torch.long,
    )
    
    index_list = []
    src_list = []
    for j in range(0, batch_size, num_generations):
        tmp_index = []
        for k in range(num_generations):
            tmp_index.append((j + k) * embedder_group_size)
        tmp_src = [1] * num_generations
        for _ in range(num_generations):
            index_list.append(tmp_index)
            src_list.append(tmp_src)
    
    index = torch.tensor(index_list, device=device, dtype=torch.int64)
    src = torch.tensor(src_list, device=device, dtype=torch.long)
    cur_targets = cur_targets.scatter(dim=-1, index=index, src=src)
    
    # Fill qrels and search_results for pytrec_eval
    qrels = defaultdict(dict)
    search_results = defaultdict(dict)
    
    for i in range(batch_size):
        qid = f"q-{i}"
        pos_count = 0
        
        for j in range(batch_size * embedder_group_size):
            docid = f"d-{i}-{j}"
            rel = int(cur_targets[i][j].item())
            if rel == 1:
                pos_count += 1
                qrels[qid][docid] = rel
            
            search_results[qid][docid] = float(sim_scores[i][j].item())
        
        assert pos_count == num_generations, f"pos_count ({pos_count}) != num_generations ({num_generations})"
    
    # Compute nDCG metric
    ndcg_dict = evaluate_ndcg(qrels, search_results, [ndcg_value])[f"ndcg@{ndcg_value}"]
    ndcg_list = []
    for i in range(batch_size):
        qid = f"q-{i}"
        ndcg_list.append(ndcg_dict[qid])
    
    ndcg_rewards = torch.tensor(ndcg_list, dtype=torch.float32, device=device)
    return ndcg_rewards


def ndcg_reward_map_func(old_rewards: torch.Tensor, device) -> torch.Tensor:
    """
    Map nDCG rewards to a wider range for better training signal.
    """
    rewards = old_rewards.clone()
    rewards = torch.where(rewards > 0.95, torch.tensor(10.0, device=device), rewards)
    rewards = torch.where((rewards > 0.85) & (rewards <= 0.95), torch.tensor(7.5, device=device), rewards)
    rewards = torch.where((rewards > 0.75) & (rewards <= 0.85), torch.tensor(6.0, device=device), rewards)
    rewards = torch.where((rewards > 0.65) & (rewards <= 0.75), torch.tensor(5.0, device=device), rewards)
    rewards = torch.where((rewards > 0.55) & (rewards <= 0.65), torch.tensor(4.0, device=device), rewards)
    rewards = torch.where((rewards > 0.45) & (rewards <= 0.55), torch.tensor(2.0, device=device), rewards)
    rewards = torch.where((rewards > 0.35) & (rewards <= 0.45), torch.tensor(1.0, device=device), rewards)
    rewards = torch.where(rewards <= 0.05, torch.tensor(-3.5, device=device), rewards)
    rewards = torch.where((rewards > 0.05) & (rewards <= 0.15), torch.tensor(-2.5, device=device), rewards)
    rewards = torch.where((rewards > 0.15) & (rewards <= 0.25), torch.tensor(-1.5, device=device), rewards)
    rewards = torch.where((rewards > 0.25) & (rewards <= 0.35), torch.tensor(-0.5, device=device), rewards)
    return rewards


def retrieval_reward_map_func(old_rewards: torch.Tensor, device, 
                               clip_range: float = 100.0, 
                               target_range: float = 2.0,
                               use_asymmetric: bool = False,
                               asymmetric_neg_clip: float = None,
                               asymmetric_pos_clip: float = None,
                               asymmetric_neg_target: float = None,
                               asymmetric_pos_target: float = None) -> torch.Tensor:
    """
    Map retrieval rewards to a smaller range for stable training.
    
    Strategy (Symmetric): Clip to [-clip_range, clip_range] then linearly scale to [-target_range, target_range]
    
    Strategy (Asymmetric): Different clip/target for positive and negative rewards
        - Negative: clip to [-asymmetric_neg_clip, 0], scale to [-asymmetric_neg_target, 0]
        - Positive: clip to [0, asymmetric_pos_clip], scale to [0, asymmetric_pos_target]
    
    Example (Symmetric):
        Raw rewards: [-218, 114] with mean=-69
        After clipping to [-100, 100]: [-100, 100]
        After scaling to [-2, 2]: rewards in range [-2, 2]
    
    Example (Asymmetric):
        Config: neg_clip=200, pos_clip=50, neg_target=5, pos_target=1
        Raw rewards: [-218, 114] with mean=-69
        Negative: clip to [-200, 0], scale to [-5, 0] → -218 becomes -5 (not fully clipped!)
        Positive: clip to [0, 50], scale to [0, 1] → 114 becomes 1 (clipped)
        Result: reward range is [-5, 1], negatives dominate less
    
    Args:
        old_rewards: Raw retrieval rewards (e.g., range [-218, 114])
        device: Device for tensor operations
        clip_range: Clip rewards to [-clip_range, clip_range] (default 100.0, symmetric mode)
        target_range: Target range after scaling (default 2.0, symmetric mode)
        use_asymmetric: Whether to use asymmetric clipping (default False)
        asymmetric_neg_clip: Clip range for negative rewards (e.g., 200 → clip to [-200, 0])
        asymmetric_pos_clip: Clip range for positive rewards (e.g., 50 → clip to [0, 50])
        asymmetric_neg_target: Target range for negative rewards (e.g., 5 → scale to [-5, 0])
        asymmetric_pos_target: Target range for positive rewards (e.g., 1 → scale to [0, 1])
    
    Returns:
        Mapped rewards
    """
    if use_asymmetric:
        # Asymmetric mode: different clip/target for positive and negative rewards
        neg_clip = asymmetric_neg_clip if asymmetric_neg_clip is not None else clip_range
        pos_clip = asymmetric_pos_clip if asymmetric_pos_clip is not None else clip_range
        neg_target = asymmetric_neg_target if asymmetric_neg_target is not None else target_range
        pos_target = asymmetric_pos_target if asymmetric_pos_target is not None else target_range
        
        # Separate positive and negative rewards
        positive_mask = old_rewards > 0
        negative_mask = old_rewards < 0
        zero_mask = old_rewards == 0
        
        rewards = torch.zeros_like(old_rewards)
        
        # Handle negative rewards: clip to [-neg_clip, 0], scale to [-neg_target, 0]
        if negative_mask.any():
            neg_rewards = old_rewards[negative_mask]
            neg_clipped = torch.clamp(neg_rewards, min=-neg_clip, max=0)
            neg_scaled = neg_clipped / neg_clip * neg_target
            rewards[negative_mask] = neg_scaled
        
        # Handle positive rewards: clip to [0, pos_clip], scale to [0, pos_target]
        if positive_mask.any():
            pos_rewards = old_rewards[positive_mask]
            pos_clipped = torch.clamp(pos_rewards, min=0, max=pos_clip)
            pos_scaled = pos_clipped / pos_clip * pos_target
            rewards[positive_mask] = pos_scaled
        
        # Zero rewards stay zero
        rewards[zero_mask] = 0
        
        return rewards
    else:
        # Symmetric mode (original behavior)
        # Clip to avoid extreme outliers
        rewards = torch.clamp(old_rewards, min=-clip_range, max=clip_range)
        
        # Linearly scale to target range
        rewards = rewards / clip_range * target_range
        
        return rewards


def compute_retrieval_reward(
    ori_query_embeddings: torch.Tensor,
    rewritten_query_embeddings: torch.Tensor,
    passage_embeddings: torch.Tensor,
    num_generations: int,
    pos_weight: float = 1.0,
    gap_weight: float = 1.0,
    temperature: float = 0.02,
    use_reward_map: bool = False,
    use_in_batch_neg: bool = True,
    reward_map_clip_range: float = 100.0,
    reward_map_target_range: float = 2.0,
    use_asymmetric_clip: bool = False,
    asymmetric_neg_clip: float = None,
    asymmetric_pos_clip: float = None,
    asymmetric_neg_target: float = None,
    asymmetric_pos_target: float = None,
) -> torch.Tensor:
    """
    Compute retrieval reward with improved strategy:
    1. Reward positive score improvement
    2. Reward positive-negative gap improvement
    
    New Formula:
        reward = pos_weight * delta_pos + gap_weight * delta_gap
        
        where:
        - delta_pos = (rewritten_pos - ori_pos)  # 正样本分数提升
        - delta_gap = (rewritten_gap - ori_gap)  # 正负分差提升
        - gap = pos_score - neg_mean
    
    Args:
        ori_query_embeddings: [batch_size, hidden_dim] original query embeddings
        rewritten_query_embeddings: [batch_size * num_generations, hidden_dim] rewritten query embeddings
        passage_embeddings: [batch_size, hidden_dim] passage embeddings (one positive per query)
            OR [batch_size * group_size, hidden_dim] if use_in_batch_neg=False (legacy mode)
        num_generations: Number of generations per query
        pos_weight: Weight for positive score improvement (default 1.0)
        gap_weight: Weight for gap improvement (default 1.0)
        temperature: Temperature for similarity computation
        use_reward_map: Whether to use reward mapping function
        use_in_batch_neg: If True, use in-batch negatives (other queries' positives as negatives).
                          If False, use legacy mode (assume group_size passages per query, first is positive).
        reward_map_clip_range: Clip range for reward mapping (default 100.0)
        reward_map_target_range: Target range for reward mapping (default 2.0)
        
    Returns:
        retrieval_rewards: [batch_size * num_generations] tensor
    """
    device = ori_query_embeddings.device
    batch_size = ori_query_embeddings.size(0)
    
    # Normalize embeddings
    ori_query_embeddings = F.normalize(ori_query_embeddings, dim=-1)
    rewritten_query_embeddings = F.normalize(rewritten_query_embeddings, dim=-1)
    passage_embeddings = F.normalize(passage_embeddings, dim=-1)
    
    if use_in_batch_neg:
        # In-batch negative mode: each query has one positive passage
        # Negatives are other queries' positive passages in the batch
        # passage_embeddings: [num_queries, hidden_dim] where num_queries = batch_size // num_generations
        # ori_query_embeddings: [num_queries, hidden_dim]
        # rewritten_query_embeddings: [num_queries * num_generations, hidden_dim]
        num_queries = ori_query_embeddings.size(0)
        assert passage_embeddings.size(0) == num_queries, \
            f"Expected passage_embeddings shape [num_queries={num_queries}, hidden_dim], got {passage_embeddings.shape}"
        assert rewritten_query_embeddings.size(0) == num_queries * num_generations, \
            f"Expected rewritten_query_embeddings shape [num_queries * num_generations={num_queries * num_generations}, hidden_dim], got {rewritten_query_embeddings.shape}"
        
        # Update batch_size to num_queries for this mode
        batch_size = num_queries
        
        # Compute similarity scores: [num_queries, num_queries]
        # Each row i: query i's similarity with all passages (including its own positive)
        ori_scores = torch.matmul(ori_query_embeddings, passage_embeddings.T) / temperature  # [num_queries, num_queries]
        
        # Rewritten query scores: [num_queries * num_generations, num_queries]
        rewritten_scores = torch.matmul(rewritten_query_embeddings, passage_embeddings.T) / temperature  # [num_queries * num_generations, num_queries]
        
        # Reshape rewritten_scores: [num_queries, num_generations, num_queries]
        rewritten_scores_reshaped = rewritten_scores.view(batch_size, num_generations, batch_size)
        
        # Extract positive scores (diagonal elements)
        # For query i, its positive passage is at index i
        batch_indices = torch.arange(batch_size, device=device)
        ori_pos_scores = ori_scores[batch_indices, batch_indices]  # [batch_size]
        
        # For rewritten queries: extract diagonal elements across the first and last dimensions
        # rewritten_scores_reshaped: [num_queries, num_generations, num_queries]
        # For query i, generation j, positive passage is at index i
        # Use torch.diagonal to extract diagonal across dim 0 and dim 2
        # torch.diagonal(input, dim1=0, dim2=2) gives [num_generations, num_queries]
        # Then transpose to get [num_queries, num_generations]
        rewritten_pos_scores = torch.diagonal(rewritten_scores_reshaped, dim1=0, dim2=2).T  # [num_queries, num_generations]
        
        # Extract negative scores (all other passages in batch)
        # For query i, negatives are all passages except passage i
        # Create mask: [batch_size, batch_size], True for negatives
        neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)  # [batch_size, batch_size]
        
        # Compute negative mean for original queries
        ori_neg_scores = ori_scores[neg_mask].view(batch_size, batch_size - 1)  # [batch_size, batch_size - 1]
        ori_neg_mean = ori_neg_scores.mean(dim=-1)  # [batch_size]
        
        # Compute negative mean for rewritten queries (vectorized)
        # rewritten_scores_reshaped: [batch_size, num_generations, batch_size]
        # For each query i, negatives are all passages except passage i
        # Sum all scores, subtract positive, divide by (batch_size - 1)
        rewritten_sum_all = rewritten_scores_reshaped.sum(dim=-1)  # [batch_size, num_generations]
        rewritten_pos_scores_for_subtract = rewritten_pos_scores  # [batch_size, num_generations]
        rewritten_neg_sum = rewritten_sum_all - rewritten_pos_scores_for_subtract
        # Avoid division by zero if batch_size is 1
        neg_count = max(1, batch_size - 1)
        rewritten_neg_mean = rewritten_neg_sum / neg_count  # [batch_size, num_generations]
        
    else:
        # Legacy mode: assume group_size passages per query, first is positive
        group_size = passage_embeddings.size(0) // batch_size
        
        # Compute similarity scores
        ori_scores = torch.matmul(ori_query_embeddings, passage_embeddings.T) / temperature  # [batch_size, batch_size * group_size]
        rewritten_scores = torch.matmul(rewritten_query_embeddings, passage_embeddings.T) / temperature  # [batch_size * num_generations, batch_size * group_size]
        
        # Reshape rewritten_scores: [batch_size, num_generations, batch_size * group_size]
        rewritten_scores_reshaped = rewritten_scores.view(batch_size, num_generations, -1)
        
        # Extract passage scores for each query
        passage_indices = (torch.arange(batch_size, device=device).unsqueeze(1) * group_size + 
                           torch.arange(group_size, device=device).unsqueeze(0))  # [batch_size, group_size]
        
        batch_indices = torch.arange(batch_size, device=device)
        ori_query_passage_scores = ori_scores[batch_indices[:, None], passage_indices]  # [batch_size, group_size]
        
        rewritten_query_passage_scores = rewritten_scores_reshaped[
            batch_indices[:, None, None], 
            torch.arange(num_generations, device=device)[None, :, None],
            passage_indices[:, None, :]
        ]  # [batch_size, num_generations, group_size]
        
        # Compute metrics (first passage is positive)
        ori_pos_scores = ori_query_passage_scores[:, 0]  # [batch_size]
        if group_size > 1:
            ori_neg_scores = ori_query_passage_scores[:, 1:]  # [batch_size, group_size-1]
            ori_neg_mean = ori_neg_scores.mean(dim=-1)  # [batch_size]
        else:
            ori_neg_mean = torch.zeros(batch_size, device=device)
        
        rewritten_pos_scores = rewritten_query_passage_scores[:, :, 0]  # [batch_size, num_generations]
        if group_size > 1:
            rewritten_neg_scores = rewritten_query_passage_scores[:, :, 1:]  # [batch_size, num_generations, group_size-1]
            rewritten_neg_mean = rewritten_neg_scores.mean(dim=-1)  # [batch_size, num_generations]
        else:
            rewritten_neg_mean = torch.zeros(batch_size, num_generations, device=device)
    
    # Compute gaps
    ori_gap = ori_pos_scores - ori_neg_mean  # [batch_size]
    rewritten_gap = rewritten_pos_scores - rewritten_neg_mean  # [batch_size, num_generations]
    
    # Expand ori metrics to match rewritten shape
    ori_pos_scores_expanded = ori_pos_scores.unsqueeze(1).expand(-1, num_generations)  # [batch_size, num_generations]
    ori_gap_expanded = ori_gap.unsqueeze(1).expand(-1, num_generations)  # [batch_size, num_generations]
    
    # Compute improvements
    delta_pos = rewritten_pos_scores - ori_pos_scores_expanded  # [batch_size, num_generations]
    delta_gap = rewritten_gap - ori_gap_expanded  # [batch_size, num_generations]
    
    # Combined reward
    raw_rewards = pos_weight * delta_pos + gap_weight * delta_gap  # [batch_size, num_generations]
    
    # Flatten to [batch_size * num_generations]
    raw_rewards = raw_rewards.flatten()
    
    # 🔍 DEBUG: Print detailed retrieval reward calculation for first sample
    import os
    DEBUG_LOG_ENABLED = os.environ.get("GAP_DEBUG_LOG", "0") not in {"0", "false", "False"}
    if DEBUG_LOG_ENABLED and batch_size > 0:
        # Get first query's values
        first_query_idx = 0
        first_ori_pos = ori_pos_scores[first_query_idx].item()
        first_ori_neg_mean = ori_neg_mean[first_query_idx].item()
        first_ori_gap = ori_gap[first_query_idx].item()
        
        # Get first generation's values
        first_gen_idx = first_query_idx * num_generations
        if first_gen_idx < rewritten_pos_scores.shape[0]:
            first_rew_pos = rewritten_pos_scores[first_query_idx, 0].item()
            first_rew_neg_mean = rewritten_neg_mean[first_query_idx, 0].item()
            first_rew_gap = rewritten_gap[first_query_idx, 0].item()
            first_delta_pos = delta_pos[first_query_idx, 0].item()
            first_delta_gap = delta_gap[first_query_idx, 0].item()
            first_raw_reward = raw_rewards[first_gen_idx].item()
            
            print(f"\n[🔍 Retrieval Reward Details] Query 0, Generation 0:")
            print(f"  Original: pos={first_ori_pos:.4f}, neg_mean={first_ori_neg_mean:.4f}, gap={first_ori_gap:.4f}")
            print(f"  Rewritten: pos={first_rew_pos:.4f}, neg_mean={first_rew_neg_mean:.4f}, gap={first_rew_gap:.4f}")
            print(f"  Delta: pos={first_delta_pos:.4f}, gap={first_delta_gap:.4f}")
            print(f"  Weights: pos_weight={pos_weight:.4f}, gap_weight={gap_weight:.4f}")
            print(f"  Raw Reward: {pos_weight:.4f} * {first_delta_pos:.4f} + {gap_weight:.4f} * {first_delta_gap:.4f} = {first_raw_reward:.4f}")
    
    # Apply reward mapping if needed
    if use_reward_map:
        rewards = retrieval_reward_map_func(
            raw_rewards, 
            device, 
            clip_range=reward_map_clip_range, 
            target_range=reward_map_target_range,
            use_asymmetric=use_asymmetric_clip,
            asymmetric_neg_clip=asymmetric_neg_clip,
            asymmetric_pos_clip=asymmetric_pos_clip,
            asymmetric_neg_target=asymmetric_neg_target,
            asymmetric_pos_target=asymmetric_pos_target
        )
        
        if DEBUG_LOG_ENABLED and batch_size > 0:
            first_gen_idx = 0
            if first_gen_idx < len(rewards):
                first_mapped_reward = rewards[first_gen_idx].item() if isinstance(rewards, torch.Tensor) else rewards[first_gen_idx]
                print(f"  Mapped Reward: {first_mapped_reward:.4f} (from {first_raw_reward:.4f})")
    else:
        rewards = raw_rewards
    
    return rewards


def compute_ranking_reward(
    rewritten_query_embeddings: torch.Tensor,
    passage_embeddings: torch.Tensor,
    num_generations: int,
    temperature: float = 0.02,
    use_in_batch_neg: bool = True,
    return_ranks: bool = False,
) -> torch.Tensor:
    """
    Compute ranking reward based on the rank of positive passage in batch.
    
    For each query's generation, compute similarity scores with all passages in batch,
    then check the rank of the positive passage. Reward is higher for better ranks.
    
    Args:
        rewritten_query_embeddings: [num_queries * num_generations, hidden_dim] - rewritten query embeddings
        passage_embeddings: [num_queries, hidden_dim] - positive passage embeddings (one per query)
        num_generations: Number of generations per query
        temperature: Temperature for similarity computation
        use_in_batch_neg: If True, use in-batch negatives (other queries' positives as negatives).
        
    Returns:
        ranking_rewards: [num_queries * num_generations] tensor
        ranking_rewards[i] = reward based on rank of positive passage for generation i
        Reward formula: (num_queries - rank) / (num_queries - 1), normalized to [0, 1]
        Rank 1 (best) gets reward 1.0, rank num_queries gets reward 0.0
    """
    device = rewritten_query_embeddings.device
    
    # Normalize embeddings
    rewritten_query_embeddings = F.normalize(rewritten_query_embeddings, dim=-1)
    passage_embeddings = F.normalize(passage_embeddings, dim=-1)
    
    if use_in_batch_neg:
        # In-batch negative mode: each query has one positive passage
        # Negatives are other queries' positive passages in the batch
        num_queries = passage_embeddings.size(0)
        assert rewritten_query_embeddings.size(0) == num_queries * num_generations, \
            f"Expected rewritten_query_embeddings shape [num_queries * num_generations={num_queries * num_generations}, hidden_dim], got {rewritten_query_embeddings.shape}"
        
        # Compute similarity scores: [num_queries * num_generations, num_queries]
        # Each row: one generation's similarity with all passages in batch
        sim_scores = torch.matmul(rewritten_query_embeddings, passage_embeddings.T) / temperature  # [num_queries * num_generations, num_queries]
        
        # Reshape to [num_queries, num_generations, num_queries]
        sim_scores_reshaped = sim_scores.view(num_queries, num_generations, num_queries)
        
        # For each query i, its positive passage is at index i
        # Get the rank of passage i for each generation
        # Rank = number of passages with higher similarity score + 1
        query_indices = torch.arange(num_queries, device=device)  # [num_queries]
        
        # Get positive passage scores: [num_queries, num_generations]
        # sim_scores_reshaped[query_indices, :, query_indices] directly gives [num_queries, num_generations]
        # where result[i, j] = sim_scores_reshaped[i, j, i] (query i's generation j with passage i)
        pos_scores = sim_scores_reshaped[query_indices, :, query_indices]  # [num_queries, num_generations]
        
        # For each generation, count how many passages have higher scores than positive
        # This gives us the rank (1-indexed)
        # sim_scores_reshaped: [num_queries, num_generations, num_queries]
        # For query i, generation j: count passages with score > pos_scores[i, j]
        pos_scores_expanded = pos_scores.unsqueeze(-1)  # [num_queries, num_generations, 1]
        
        # Count passages with higher scores: [num_queries, num_generations]
        higher_count = (sim_scores_reshaped > pos_scores_expanded).sum(dim=-1)  # [num_queries, num_generations]
        
        # Rank = higher_count + 1 (1-indexed, rank 1 is best)
        ranks = higher_count + 1  # [num_queries, num_generations]
        
        # Convert rank to reward: (num_queries - rank) / (num_queries - 1)
        # Rank 1 -> reward = (num_queries - 1) / (num_queries - 1) = 1.0
        # Rank num_queries -> reward = (num_queries - num_queries) / (num_queries - 1) = 0.0
        if num_queries > 1:
            ranking_rewards = (num_queries - ranks.float()) / (num_queries - 1)  # [num_queries, num_generations]
        else:
            # Edge case: only one query, rank is always 1
            # Ranking reward has no meaning when there's only one query (no comparison)
            # Return all 1.0 rewards (no discrimination), but this is expected behavior
            ranking_rewards = torch.ones(num_queries, num_generations, device=device)
        
        # Flatten to [num_queries * num_generations]
        ranking_rewards = ranking_rewards.flatten()
        ranks_flat = ranks.flatten()  # [num_queries * num_generations]
        
    else:
        # Legacy mode not supported for ranking reward
        raise NotImplementedError("Ranking reward only supports use_in_batch_neg=True")
    
    if return_ranks:
        return ranking_rewards, ranks_flat
    else:
        return ranking_rewards

