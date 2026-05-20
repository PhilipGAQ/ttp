#!/bin/bash
# =============================================================================
# Stage 3: GRPO Reinforcement Learning Training Script
# =============================================================================
# Paper: Think-to-Personalize (TTP)
# Section 3.4 + Section 4.1.3 Implementation Details
#
# Paper Parameters (Section 4.1.3):
#   - Backbone: Qwen2.5-3B-Instruct
#   - RL Framework: VeRL
#   - Group size (G): 8
#   - Epochs: 3
#   - Learning rate: 2e-7
#   - Total training batch size: 128
#   - Format weight (λ_fmt): 0.5
#   - Length penalty weight (λ_len): 0.5, threshold L=64 tokens
#   - Retrieval reward weight (λ_ret): 2.0
#   - R_retrieval = α·ΔS_pos + β·ΔS_margin, α=2.0, β=1.0, clipped to [-1,1]
#   - InfoNCE loss weight (λ_cl) in total objective: 0.1
#   - InfoNCE temperature: 0.02
#   - Input truncation: query + item = 512 tokens
#   - GPUs: 8x NVIDIA A100
#   - Dynamic Positive Selection: enabled (best rewrite by retrieval reward)
# =============================================================================

set -e

export WANDB_DISABLED=true
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ===== Paths (fill in your actual paths) =====
MODEL_PATH="path/to/sft_checkpoint"              # Stage 2 SFT output model
TRAIN_DATA="path/to/rl_train_data.jsonl"         # Top 30% highest-gain samples
VAL_DATA="path/to/rl_val_data.jsonl"
OUTPUT_DIR="path/to/output/grpo_checkpoint"

# Optional: Reranker for reranker reward (disabled by default per paper)
# export RERANKER_MODEL="BAAI/bge-reranker-v2-m3"

# ===== Navigate to ttp_rl root =====
cd "$(dirname "$0")/../ttp_rl"

# ===== Launch GRPO Training =====
python -m recipe.ttp_grpo.main_gap_grpo \
    --config-name gap_grpo_trainer \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.train_group_size=8 \
    \
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.max_model_len=1024 \
    \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    reward_model.reward_kwargs.format_weight=0.5 \
    reward_model.reward_kwargs.retrieval_weight=2.0 \
    reward_model.reward_kwargs.ranking_weight=0.0 \
    reward_model.reward_kwargs.reranker_reward_weight=0.0 \
    reward_model.reward_kwargs.format_reward_value=0.0 \
    reward_model.reward_kwargs.format_penalty_value=-1.0 \
    reward_model.reward_kwargs.length_threshold=64 \
    reward_model.reward_kwargs.length_penalty=-0.5 \
    reward_model.reward_kwargs.retrieval_reward_pos_weight=2.0 \
    reward_model.reward_kwargs.retrieval_reward_neg_weight=1.0 \
    reward_model.reward_kwargs.temperature=0.02 \
    reward_model.reward_kwargs.use_retrieval_reward_map=true \
    reward_model.reward_kwargs.use_asymmetric_clip=false \
    \
    gap_config.embedder_loss_weight=0.1 \
    gap_config.rl_loss_weight=1.0 \
    gap_config.infonce_temperature=0.02 \
    gap_config.normalized_embeddings=true \
    gap_config.use_best_rewrite_selection=true \
    gap_config.mask_hist=true \
    \
    trainer.total_epochs=3 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=50 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=ttp_grpo \
    trainer.experiment_name=ttp_grpo_paper
