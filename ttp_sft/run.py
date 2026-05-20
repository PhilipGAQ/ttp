import os
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from transformers.trainer_utils import set_seed

from .arguments import ModelArguments, DataArguments, CustomTrainingArguments
from .data import MultiFileDataset, QueryGenCollator, SameFileBatchSampler
from .model import QueryGenModel
from .trainer import QueryGenTrainer

from torch.utils.tensorboard import SummaryWriter

# ChatML-like tokens
BASE_BOS: str = ""
TURN_SEP: str = "\n"
IM_START: str = "<|im_start|>"
IM_END: str = "<|im_end|>"
USER_BOS: str = "<|im_start|>user\n"
USER_EOS: str = "<|im_end|>\n"
ASSISTANT_BOS: str = "<|im_start|>assistant\n"
ASSISTANT_EOS: str = "<|im_end|>"

logger = logging.getLogger(__name__)


def args_to_dtype(args):
    if args.bf16: return torch.bfloat16
    if args.fp16: return torch.float16
    return torch.float32


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Set global seed for reproducibility
    set_seed(42)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        padding_side="right",
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=1,
    )

    if not(tokenizer.pad_token) and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info('Set pad token to eos token: %s', tokenizer.pad_token)

    if training_args.local_rank in [-1, 0]:
        tensorboard_dir = os.path.join(training_args.output_dir, "tensorboard_logs")
        os.makedirs(tensorboard_dir, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=tensorboard_dir, max_queue=100)
    else:
        summary_writer = None

    # Ensure special tokens exist for think/emb; this makes span detection exact
    added = False
    special_to_add = []
    vocab_dict = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else tokenizer.vocab if hasattr(tokenizer, 'vocab') else {}
    
    for tok in ["<think>", "</think>", "<emb>"]:
        if tok in vocab_dict:
            token_id = vocab_dict[tok]
            logger.info(f"[special-init] Token {tok!r} already in vocab with id={token_id}")
        else:
            token_id = tokenizer.convert_tokens_to_ids(tok)
            unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else -1
            if token_id is None or token_id == unk_id or token_id < 0:
                special_to_add.append(tok)
                logger.info(f"[special-init] Token {tok!r} not in vocab, will add it")
            else:
                logger.info(f"[special-init] Token {tok!r} found via convert_tokens_to_ids with id={token_id}")
    
    if special_to_add:
        num_added = tokenizer.add_tokens(special_to_add, special_tokens=True)
        logger.info(f"[special-init] Added {num_added} special tokens: {special_to_add}")
        added = True

    dataset = MultiFileDataset(data_args.train_files, data_args, tokenizer)

    # Validate loss factors before creating model
    if training_args.loss_gen_factor == 0 and training_args.loss_contrast_factor == 0:
        raise ValueError(
            "loss_gen_factor and loss_contrast_factor cannot both be 0. "
            "At least one must be > 0. "
            "Set loss_gen_factor=0 for contrastive-only training, "
            "or loss_contrast_factor=0 for SFT-only training."
        )
    
    model = QueryGenModel(
        model_name_or_path=model_args.model_name_or_path,
        normalized=model_args.normalized,
        negatives_cross_device=training_args.negatives_cross_device,
        temperature=training_args.temperature,
        torch_dtype=args_to_dtype(training_args),
        loss_gen_factor=training_args.loss_gen_factor,
        loss_contrast_factor=training_args.loss_contrast_factor,
        embedding_view=training_args.embedding_view,
        mask_history=training_args.mask_history,
        mrl_dim=training_args.mrl_dim,
        sub_batch_size=training_args.sub_batch_size,
        use_cache=False,
        low_cpu_mem_usage=True,
    )

    # === Handle special token embedding resizing & initialization ===
    input_embeddings = model.model.get_input_embeddings()
    old_num_embeddings = input_embeddings.weight.size(0)
    new_num_embeddings = len(tokenizer)
    
    if old_num_embeddings != new_num_embeddings:
        logger.info(f'[special-init] Resizing token embeddings: {old_num_embeddings} -> {new_num_embeddings}')
        model.model.resize_token_embeddings(new_num_embeddings)
        model.model.config.vocab_size = new_num_embeddings
        model.gen_loss_fn.vocab_size = new_num_embeddings
        input_embeddings = model.model.get_input_embeddings()
        logger.info(f'[special-init] Resized successfully. Model config vocab_size: {model.model.config.vocab_size}')
    else:
        if model.gen_loss_fn.vocab_size != new_num_embeddings:
            logger.warning(f"[special-init] Updating gen_loss_fn.vocab_size from {model.gen_loss_fn.vocab_size} to {new_num_embeddings}")
            model.gen_loss_fn.vocab_size = new_num_embeddings
        logger.info(f'[special-init] No resize needed; existing size = {old_num_embeddings}, tokenizer size = {new_num_embeddings}')

    # Initialize new special tokens
    if added:
        with torch.no_grad():
            weight = input_embeddings.weight
            existing_weight = weight[:old_num_embeddings] if old_num_embeddings <= weight.size(0) else weight
            mean_embedding = existing_weight.mean(dim=0)
            std_embedding = existing_weight.std(dim=0).mean().item()
            
            logger.info(f"[special-init] Existing embeddings mean norm: {mean_embedding.norm().item():.4f}, std: {std_embedding:.4f}")
            
            vocab_dict = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else tokenizer.vocab if hasattr(tokenizer, 'vocab') else {}
            
            for special_tok in ["<think>", "</think>", "<emb>"]:
                if special_tok in vocab_dict:
                    special_id = vocab_dict[special_tok]
                else:
                    special_id = tokenizer.convert_tokens_to_ids(special_tok)
                
                if special_id is None or special_id < 0:
                    logger.warning(f"[special-init] special token {special_tok!r} not found in tokenizer, skip init.")
                    continue
                
                unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else -1
                if special_id == unk_id:
                    logger.warning(f"[special-init] special token {special_tok!r} maps to unk_token_id, skip init.")
                    continue

                if special_id >= weight.size(0):
                    logger.warning(f"[special-init] special_id={special_id} out of range (num_embeddings={weight.size(0)}), skip init for {special_tok!r}.")
                    continue

                random_noise = torch.randn_like(mean_embedding) * (std_embedding * 0.1)
                new_embedding = mean_embedding + random_noise
                target_norm = mean_embedding.norm().item()
                new_norm = new_embedding.norm().item()
                if new_norm > 0:
                    new_embedding = new_embedding * (target_norm / new_norm)
                
                weight[special_id].copy_(new_embedding)
                
                final_norm = weight[special_id].norm().item()
                logger.info(f"[special-init] Initialized {special_tok!r} (id={special_id}) with improved strategy, norm: {final_norm:.4f}")
            
            # Also initialize lm_head for new tokens
            output_embeddings = model.model.get_output_embeddings()
            if output_embeddings is not None:
                output_weight = output_embeddings.weight
                if output_weight.size(0) == new_num_embeddings:
                    logger.info("[special-init] Initializing new tokens in lm_head...")
                    
                    existing_output_weight = output_weight[:old_num_embeddings] if old_num_embeddings <= output_weight.size(0) else output_weight
                    mean_output = existing_output_weight.mean(dim=0)
                    std_output = existing_output_weight.std(dim=0).mean().item()
                    logger.info(f"[special-init] lm_head mean norm: {mean_output.norm().item():.4f}, std: {std_output:.4f}")

                    vocab_dict = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else tokenizer.vocab if hasattr(tokenizer, 'vocab') else {}
                    for special_tok in ["<think>", "</think>", "<emb>"]:
                        if special_tok in vocab_dict:
                            special_id = vocab_dict[special_tok]
                        else:
                            special_id = tokenizer.convert_tokens_to_ids(special_tok)
                        
                        if special_id is not None and 0 <= special_id < output_weight.size(0):
                            random_noise_out = torch.randn_like(mean_output) * (std_output * 0.1)
                            new_output_emb = mean_output + random_noise_out
                            
                            target_norm_out = mean_output.norm().item()
                            new_norm_out = new_output_emb.norm().item()
                            if new_norm_out > 0:
                                new_output_emb = new_output_emb * (target_norm_out / new_norm_out)
                                
                            output_weight[special_id].copy_(new_output_emb)
                            logger.info(f"[special-init] Initialized {special_tok!r} in lm_head (id={special_id}) using lm_head stats")


    # LoRA / QLoRA
    if training_args.lora or training_args.qlora:
        if training_args.qlora:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            from peft import prepare_model_for_kbit_training
            model.model = prepare_model_for_kbit_training(
                model.model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        from peft import get_peft_model, LoraConfig, TaskType
        target_modules = [m.strip() for m in training_args.lora_target_modules.split(',') if m.strip()]
        
        if training_args.train_embed_token_only:
            logger.info("[lora-config] train_embed_token_only is True: Clearing LoRA target_modules to avoid training other layers.")
            target_modules = []

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
        )
        
        modules_to_save = ["embed_tokens", "lm_head"]
        if added or training_args.train_embed_token_only:
            logger.info(f"[lora-config] Setting modules_to_save to: {modules_to_save}")
            peft_config.modules_to_save = modules_to_save
        
        if hasattr(model.model, "config"):
            model.model.config.tie_word_embeddings = False
            logger.info("[lora-config] Forced tie_word_embeddings = False")

        model.model = get_peft_model(model.model, peft_config)
        try:
            model.model.print_trainable_parameters()
        except Exception:
            pass

    elif training_args.train_embed_token_only:
        logger.info("[special-init] train_embed_token_only is True (no LoRA): Freezing all params except embed_tokens and lm_head.")
        for param in model.model.parameters():
            param.requires_grad = False
            
        input_emb = model.model.get_input_embeddings()
        if input_emb:
            input_emb.weight.requires_grad = True
            logger.info("[special-init] Unfrozen input_embeddings")
            
        output_emb = model.model.get_output_embeddings()
        if output_emb:
            output_emb.weight.requires_grad = True
            logger.info("[special-init] Unfrozen output_embeddings")
        
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.model.parameters())
        logger.info(f"[special-init] Trainable params: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.2f}%)")

    collator = QueryGenCollator(
        tokenizer,
        query_max_len=data_args.query_max_len,
        passage_max_len=data_args.passage_max_len,
        generative_max_len=data_args.generative_max_len,
        base_bos=BASE_BOS,
        user_bos=USER_BOS,
        user_eos=USER_EOS,
        assistant_bos=ASSISTANT_BOS,
        assistant_eos=ASSISTANT_EOS,
        sub_batch_size=training_args.sub_batch_size,
    )

    trainer = QueryGenTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.summary_writer = summary_writer

    # Custom sampler to keep batches within the same file using per-file batch sizes
    total_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    if dist.is_initialized():
        total_bs *= dist.get_world_size()

    trainer._get_train_sampler = lambda: SameFileBatchSampler(
        dataset=dataset, 
        global_batch_size=total_bs, 
        seed=42
    )
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.train()

    # Save: if LoRA/QLoRA, save adapter only; otherwise full model
    if training_args.lora or training_args.qlora:
        model.model.save_pretrained(training_args.output_dir)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
            config.to_json_file(os.path.join(training_args.output_dir, "config.json"))
    else:
        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
            config.to_json_file(os.path.join(training_args.output_dir, "config.json"))


if __name__ == "__main__":
    main()
