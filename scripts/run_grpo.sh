#!/bin/bash
set -e

export WANDB_DISABLED=true
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL_PATH="path/to/sft_checkpoint"              
TRAIN_DATA="path/to/rl_train_data.jsonl"         
VAL_DATA="path/to/rl_val_data.jsonl"
OUTPUT_DIR="path/to/output/grpo_checkpoint"

cd "$(dirname "$0")/../ttp_rl"

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
    reward_model.reward_kwargs.retrieval_reward_clip_range=100.0 \
    reward_model.reward_kwargs.retrieval_reward_target_range=1.0 \
    reward_model.reward_kwargs.use_asymmetric_clip=false \
    \
    gap_config.embedder_loss_weight=0.1 \
    gap_config.rl_loss_weight=1.0 \
    gap_config.infonce_temperature=0.02 \
    gap_config.normalized_embeddings=true \
    gap_config.use_best_rewrite_selection=true \
    gap_config.mask_hist=false \
    \
    trainer.total_epochs=3 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=50 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=ttp_grpo \
    trainer.experiment_name=ttp_grpo_paper
