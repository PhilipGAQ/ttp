#!/bin/bash
# =============================================================================
# Stage 2: Supervised Fine-Tuning (SFT) Training Script
# =============================================================================
# Paper: Think-to-Personalize (TTP)
# Section 3.3 Cold-Start SFT + Section 4.1.3 Implementation Details
#
# Paper Parameters (Section 4.1.3):
#   - Backbone: Qwen2.5-3B-Instruct
#   - LoRA: r=16, α=32
#   - Per-device batch size: 64
#   - Learning rate: 1e-4
#   - Generation loss weight (λ_gen): 0.5
#   - InfoNCE temperature (τ): 0.02
#   - Query-side and item-side truncation: 512 tokens
#   - Embedding dimension: 2048
#   - Embedding normalization: enabled
#   - In-batch negatives: enabled
#   - Special tokens: <think>, </think>, <embed>
#   - Training: full cleaned dataset (~1M samples)
#   - GPUs: 8x NVIDIA A100
#
# Paper Formula (Eq. 6):
#   L_Stage1 = λ_gen * L_gen + L_cl
#   where L_gen is NTP loss, L_cl is InfoNCE loss
#   λ_gen = 0.5 means: loss = 0.5 * gen_loss + 1.0 * contrast_loss
# =============================================================================

set -e

export WANDB_DISABLED=true
export SWANLAB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ===== Paths (fill in your actual paths) =====
MODEL_PATH="path/to/Qwen2.5-3B-Instruct"
TRAIN_DATA="path/to/sft_train_data.jsonl"
OUTPUT_DIR="path/to/output/sft_checkpoint"

# ===== Navigate to code root =====
cd "$(dirname "$0")/.."

# ===== Launch SFT Training =====
accelerate launch \
    --config_file scripts/accelerate_config.yaml \
    -m ttp_sft.run \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --train_files $TRAIN_DATA \
    --lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --per_device_train_batch_size 64 \
    --sub_batch_size 32 \
    --learning_rate 1e-4 \
    --weight_decay 0.02 \
    --warmup_ratio 0.02 \
    --num_train_epochs 1 \
    --bf16 True \
    --negatives_cross_device True \
    --query_max_len 512 \
    --passage_max_len 512 \
    --generative_max_len 256 \
    --train_group_size 1 \
    --loss_gen_factor 0.5 \
    --loss_contrast_factor 1.0 \
    --temperature 0.02 \
    --embedding_view both \
    --mask_history True \
    --gradient_checkpointing True \
    --dataloader_drop_last True \
    --save_steps 500 \
    --logging_steps 1 \
    --overwrite_output_dir True
