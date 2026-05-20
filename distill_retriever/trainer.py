import os
import logging
from typing import Optional

import torch

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainer

logger = logging.getLogger(__name__)


class EncoderOnlyEmbedderDistillTrainer(AbsEmbedderTrainer):
    """Trainer class for encoder-only distillation."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        if model.training:
            log_values = {
                "distill_total_loss": loss.detach().float().mean().item()
            }

            model_for_loss_log = model.module if hasattr(model, "module") else model
            loss_breakdown = getattr(model_for_loss_log, "latest_loss_breakdown", None)
            if not isinstance(loss_breakdown, dict):
                # Fallback when wrappers hide model attributes.
                loss_breakdown = {
                    key: getattr(outputs, key, None)
                    for key in [
                        "query_pos_loss",
                        "query_gen_pos_loss",
                        "query_query_gen_loss",
                        "weighted_query_pos_loss",
                        "weighted_query_gen_pos_loss",
                        "weighted_query_query_gen_loss",
                    ]
                    if getattr(outputs, key, None) is not None
                }

            if isinstance(loss_breakdown, dict):
                for key in [
                    "query_pos_loss",
                    "query_gen_pos_loss",
                    "query_query_gen_loss",
                    "weighted_query_pos_loss",
                    "weighted_query_gen_pos_loss",
                    "weighted_query_query_gen_loss",
                ]:
                    value = loss_breakdown.get(key)
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor):
                        value = value.detach().float().mean().item()
                    else:
                        value = float(value)
                    log_values[key] = value

            self.log(log_values)
            if self.is_world_process_zero():
                logger.info(
                    "step=%s distill_total_loss=%.6f query_pos_loss=%.6f query_gen_pos_loss=%.6f query_query_gen_loss=%.6f weighted_query_pos_loss=%.6f weighted_query_gen_pos_loss=%.6f weighted_query_query_gen_loss=%.6f",
                    self.state.global_step,
                    log_values["distill_total_loss"],
                    log_values.get("query_pos_loss", float("nan")),
                    log_values.get("query_gen_pos_loss", float("nan")),
                    log_values.get("query_query_gen_loss", float("nan")),
                    log_values.get("weighted_query_pos_loss", float("nan")),
                    log_values.get("weighted_query_gen_pos_loss", float("nan")),
                    log_values.get("weighted_query_query_gen_loss", float("nan")),
                )

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        if not hasattr(self.model, "save"):
            raise NotImplementedError(
                f"MODEL {self.model.__class__.__name__} does not support save interface"
            )

        self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
