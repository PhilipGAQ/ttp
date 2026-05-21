#!/bin/bash
set -e

export WANDB_DISABLED=true
export SWANLAB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL_PATH="path/to/model"
TRAIN_DATA="path/to/train_data.jsonl"
OUTPUT_DIR="path/to/output/sft_checkpoint"

cd "$(dirname "$0")/.."

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
    --mask_history False \
    --gradient_checkpointing True \
    --dataloader_drop_last True \
    --save_steps 500 \
    --logging_steps 1 \
    --overwrite_output_dir True
