#!/bin/bash
set -e

export WANDB_DISABLED=true
export SWANLAB_MODE=disabled
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

BASE_MODEL="path/to/encoder_model"               
TRAIN_DATA="path/to/distill_train_data.jsonl"   
OUTPUT_DIR="path/to/output/distill_retriever_checkpoint"
TB_LOG_DIR="path/to/logs/tensorboard/distill_retriever"
CACHE_DIR="path/to/cache"

cd "$(dirname "$0")/.."

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
