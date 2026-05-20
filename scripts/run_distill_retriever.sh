#!/bin/bash
# =============================================================================
# Stage: Distill Retriever Training Script
# =============================================================================
# Paper: Think-to-Personalize (TTP)
# Section 4.4 Online Deployment - Distilled Real-Time Retrieval
#
# Paper Parameters (Section 4.4):
#   - Teacher: Full TTP model (3B)
#   - Student: Encoder-only bi-encoder (305M, BERT-style)
#   - Objective: InfoNCE-style contrastive learning
#   - Three positive pairs with equal weights (1:1:1):
#     1. ⟨q, p+⟩: original query vs positive item
#     2. ⟨q_r, p+⟩: intent-enhanced query vs positive item
#     3. ⟨q, q_r⟩: original query vs intent-enhanced query
#   - In-batch negatives: enabled
#   - Temperature (τ): 0.02
#   - Input truncation: 512 tokens (query and passage)
#   - Embedding normalization: enabled
#   - GPUs: 8x NVIDIA A100 (4 used via accelerate config)
# =============================================================================

set -e

export WANDB_DISABLED=true
export SWANLAB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ===== Paths (fill in your actual paths) =====
BASE_MODEL="path/to/encoder_model"               # 305M encoder-only model (e.g., BERT-based)
TRAIN_DATA="path/to/distill_train_data.jsonl"    # Data with (query, intent-enhanced query, positive item)
OUTPUT_DIR="path/to/output/distill_retriever_checkpoint"
TB_LOG_DIR="path/to/logs/tensorboard/distill_retriever"
CACHE_DIR="path/to/cache"

# ===== Navigate to code root =====
cd "$(dirname "$0")/.."

# ===== Launch Distill Retriever Training =====
accelerate launch \
    --config_file scripts/accelerate_config.yaml \
    -m distill_retriever \
    --model_name_or_path $BASE_MODEL \
    --trust_remote_code True \
    --cache_dir $CACHE_DIR \
    --ddp_find_unused_parameters False \
    --train_data $TRAIN_DATA \
    --train_group_size 1 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --fp16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 40 \
    --sub_batch_size 8 \
    --dataloader_drop_last True \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --logging_steps 1 \
    --logging_dir $TB_LOG_DIR \
    --report_to tensorboard \
    --save_steps 1000 \
    --temperature 0.02 \
    --negatives_cross_device \
    --normalize_embeddings True \
    --query_pos_loss_weight 1.0 \
    --query_gen_pos_loss_weight 1.0 \
    --query_query_gen_loss_weight 1.0
