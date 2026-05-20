from dataclasses import dataclass, field

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderModelArguments,
    AbsEmbedderDataArguments,
    AbsEmbedderTrainingArguments,
)


EncoderOnlyEmbedderDistillModelArguments = AbsEmbedderModelArguments
EncoderOnlyEmbedderDistillDataArguments = AbsEmbedderDataArguments


@dataclass
class EncoderOnlyEmbedderDistillTrainingArguments(AbsEmbedderTrainingArguments):
    """Training arguments for encoder-only query-rewrite distillation."""

    query_pos_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of contrastive loss between query and positive passage."},
    )
    query_gen_pos_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of contrastive loss between rewritten query and positive passage."},
    )
    query_query_gen_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight of contrastive loss between query and rewritten query."},
    )
