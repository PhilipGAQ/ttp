from dataclasses import dataclass, field
import os
from typing import Optional, List

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HF model name or local path"})
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    normalized: bool = field(default=True)


@dataclass
class DataArguments:
    train_files: List[str] = field(
        default_factory=list,
        metadata={"help": "List of jsonl paths or a single folder path"},
    )
    query_max_len: int = field(default=32)
    passage_max_len: int = field(default=128)
    generative_max_len: int = field(default=None, metadata={"help": "Defaults to query_max_len if None"})
    max_example_num_per_dataset: int = field(default=100_000_000)
    # Defaults when per-file metadata is missing
    train_group_size: int = field(default=2, metadata={"help": "Default train group size (1 pos + train_group_size-1 neg) when file lacks train_group_size"})
    batch_size: int = field(default=-1, metadata={"help": "Default per-file batch size used by sampler when file lacks batch_size; <=0 falls back to global"})

    def __post_init__(self):
        if isinstance(self.train_files, str):
            # comma separated or directory
            if os.path.isdir(self.train_files):
                self.train_files = [os.path.join(self.train_files, f) for f in os.listdir(self.train_files) if f.endswith((".jsonl", ".json"))]
            else:
                self.train_files = [x for x in self.train_files.split(",") if x]
        if self.generative_max_len is None:
            self.generative_max_len = self.query_max_len
        if not self.train_files:
            raise ValueError("No training files provided")
        for p in self.train_files:
            if not os.path.exists(p):
                raise FileNotFoundError(p)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(default=True)
    temperature: float = field(default=0.02)
    save_safetensors: bool = field(default=False)
    sub_batch_size: int = field(default=-1, metadata={"help": "If >0, chunk inside-batch into sub-batches for memory"})
    lora: bool = field(default=False)
    qlora: bool = field(default=False)
    lora_alpha: int = field(default=32)
    lora_rank: int = field(default=16)
    lora_target_modules: str = field(default="q_proj,o_proj,v_proj,k_proj,gate_proj,up_proj,down_proj", metadata={"help": "Comma-separated module names for LoRA targets"})
    loss_gen_factor: float = field(default=1.0, metadata={"help": "Weight for generation loss. Set to 0 to disable generation task."})
    loss_contrast_factor: float = field(default=1.0, metadata={"help": "Weight for contrastive (InfoNCE) loss. Set to 0 to disable contrastive learning."})
    embedding_view: str = field(default="query", metadata={"help": "query | gen | both - What the embedding can see on query side"})
    mask_history: bool = field(default=True, metadata={"help": "If True, mask history sequence so embedding only sees query + generation"})
    train_embed_token_only: bool = field(default=False, metadata={"help": "If True, only train embed_tokens and lm_head (and LoRA adapters if configured, but typically used to disable adapters)"})
    mrl_dim: str = field(default=None, metadata={"help": "Comma-separated list of dimensions for MRL (Matryoshka Representation Learning). Example: '32,64,128'"})
