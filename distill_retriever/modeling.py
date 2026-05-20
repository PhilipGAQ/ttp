import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderModel, EmbedderOutput

logger = logging.getLogger(__name__)


@dataclass
class DistillEmbedderOutput(EmbedderOutput):
    query_pos_loss: Optional[torch.Tensor] = None
    query_gen_pos_loss: Optional[torch.Tensor] = None
    query_query_gen_loss: Optional[torch.Tensor] = None
    weighted_query_pos_loss: Optional[torch.Tensor] = None
    weighted_query_gen_pos_loss: Optional[torch.Tensor] = None
    weighted_query_query_gen_loss: Optional[torch.Tensor] = None


class BiEncoderOnlyEmbedderDistillModel(AbsEmbedderModel):
    """Encoder-only model for query rewrite distillation with 3 contrastive losses."""

    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        base_model: AutoModel,
        tokenizer: AutoTokenizer = None,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        sub_batch_size: int = -1,
        kd_loss_type: str = "kl_div",
        sentence_pooling_method: str = "cls",
        normalize_embeddings: bool = False,
        query_pos_loss_weight: float = 1.0,
        query_gen_pos_loss_weight: float = 1.0,
        query_query_gen_loss_weight: float = 1.0,
        mrl_dim: str = None,
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=negatives_cross_device,
            temperature=temperature,
            sub_batch_size=sub_batch_size,
            kd_loss_type=kd_loss_type,
        )
        self.sentence_pooling_method = sentence_pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")

        self.query_pos_loss_weight = query_pos_loss_weight
        self.query_gen_pos_loss_weight = query_gen_pos_loss_weight
        self.query_query_gen_loss_weight = query_query_gen_loss_weight

        if (
            self.query_pos_loss_weight < 0
            or self.query_gen_pos_loss_weight < 0
            or self.query_query_gen_loss_weight < 0
        ):
            raise ValueError("All loss weights must be >= 0.")
        if (
            self.query_pos_loss_weight
            + self.query_gen_pos_loss_weight
            + self.query_query_gen_loss_weight
        ) <= 0:
            raise ValueError("At least one distill loss weight must be > 0.")

        self.mrl_dims = []
        if mrl_dim is not None and mrl_dim.strip():
            self.mrl_dims = [int(dim.strip()) for dim in mrl_dim.split(",") if dim.strip()]
            for dim in self.mrl_dims:
                if dim <= 0:
                    raise ValueError(f"Invalid mrl_dim {dim}, each dimension must be greater than 0.")

    def encode(self, features, normalize: bool = None):
        if features is None:
            return None
        do_normalize = self.normalize_embeddings if normalize is None else normalize

        if not isinstance(features, list):
            if self.sub_batch_size is not None and self.sub_batch_size > 0:
                all_reps = []
                for i in range(0, len(features["attention_mask"]), self.sub_batch_size):
                    end_idx = min(i + self.sub_batch_size, len(features["attention_mask"]))
                    sub_features = {k: v[i:end_idx] for k, v in features.items()}
                    last_hidden_state = self.model(**sub_features, return_dict=True).last_hidden_state
                    reps = self._sentence_embedding(last_hidden_state, sub_features["attention_mask"])
                    all_reps.append(reps)
                all_reps = torch.cat(all_reps, dim=0).contiguous()
                if do_normalize:
                    all_reps = F.normalize(all_reps, dim=-1)
                return all_reps.contiguous()

            last_hidden_state = self.model(**features, return_dict=True).last_hidden_state
            reps = self._sentence_embedding(last_hidden_state, features["attention_mask"])
            if do_normalize:
                reps = F.normalize(reps, dim=-1)
            return reps.contiguous()

        all_reps = []
        for sub_features in features:
            last_hidden_state = self.model(**sub_features, return_dict=True).last_hidden_state
            reps = self._sentence_embedding(last_hidden_state, sub_features["attention_mask"])
            all_reps.append(reps)
        all_reps = torch.cat(all_reps, dim=0).contiguous()
        if do_normalize:
            all_reps = F.normalize(all_reps, dim=-1)
        return all_reps.contiguous()

    def _sentence_embedding(self, last_hidden_state, attention_mask):
        if self.sentence_pooling_method == "cls":
            return last_hidden_state[:, 0]
        if self.sentence_pooling_method == "mean":
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        if self.sentence_pooling_method == "last_token":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return last_hidden_state[:, -1]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            return last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                sequence_lengths,
            ]
        raise NotImplementedError(f"pooling method {self.sentence_pooling_method} not implemented")

    def compute_score(self, q_reps, p_reps):
        scores = self._compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def compute_pair_score(self, q_reps, q_gen_reps):
        scores = self._compute_similarity(q_reps, q_gen_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def _compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def _compute_score_for_dim(self, q_reps, p_reps, mrl_dim: int):
        q_reps = q_reps[:, :mrl_dim]
        p_reps = p_reps[:, :mrl_dim]

        if self.normalize_embeddings:
            q_reps = F.normalize(q_reps, dim=-1)
            p_reps = F.normalize(p_reps, dim=-1)

        return self.compute_score(q_reps, p_reps)

    def _compute_pair_score_for_dim(self, q_reps, q_gen_reps, mrl_dim: int):
        q_reps = q_reps[:, :mrl_dim]
        q_gen_reps = q_gen_reps[:, :mrl_dim]

        if self.normalize_embeddings:
            q_reps = F.normalize(q_reps, dim=-1)
            q_gen_reps = F.normalize(q_gen_reps, dim=-1)

        return self.compute_pair_score(q_reps, q_gen_reps)

    def _compute_query_query_gen_loss(
        self,
        q_reps,
        q_gen_reps,
        no_in_batch_neg_flag: bool = False,
        compute_score_func=None,
        **kwargs,
    ):
        if no_in_batch_neg_flag:
            return None, q_reps.new_zeros(())

        if self.negatives_cross_device:
            cross_q_reps = self._dist_gather_tensor(q_reps)
            cross_q_gen_reps = self._dist_gather_tensor(q_gen_reps)
            if compute_score_func is None:
                scores = self.compute_pair_score(cross_q_reps, cross_q_gen_reps)
            else:
                scores = compute_score_func(cross_q_reps, cross_q_gen_reps, **kwargs)
        else:
            if compute_score_func is None:
                scores = self.compute_pair_score(q_reps, q_gen_reps)
            else:
                scores = compute_score_func(q_reps, q_gen_reps, **kwargs)

        targets = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        loss = self.compute_loss(scores, targets)
        return scores, loss

    def forward(
        self,
        queries: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]] = None,
        query_gens: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]] = None,
        passages: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]] = None,
        teacher_scores: Union[None, List[float]] = None,
        no_in_batch_neg_flag: bool = False,
    ):
        query_pos_loss = None
        query_gen_pos_loss = None
        query_query_gen_loss = None
        weighted_query_pos_loss = None
        weighted_query_gen_pos_loss = None
        weighted_query_query_gen_loss = None

        q_reps = self.encode(queries, normalize=False if self.mrl_dims else None)
        q_gen_reps = self.encode(
            query_gens if query_gens is not None else queries,
            normalize=False if self.mrl_dims else None,
        )
        p_reps = self.encode(passages, normalize=False if self.mrl_dims else None)

        if self.training:
            if teacher_scores is not None:
                teacher_scores = torch.tensor(teacher_scores, device=q_reps.device)
                teacher_scores = teacher_scores.view(q_reps.size(0), -1).detach()
                teacher_targets = F.softmax(teacher_scores, dim=-1)
            else:
                teacher_targets = None

            if no_in_batch_neg_flag:
                compute_loss_func = self._compute_no_in_batch_neg_loss
            else:
                if self.negatives_cross_device:
                    compute_loss_func = self._compute_cross_device_neg_loss
                else:
                    compute_loss_func = self._compute_in_batch_neg_loss

            if not self.mrl_dims:
                _, query_pos_loss = compute_loss_func(q_reps, p_reps, teacher_targets=teacher_targets)
                _, query_gen_pos_loss = compute_loss_func(q_gen_reps, p_reps, teacher_targets=teacher_targets)
                _, query_query_gen_loss = self._compute_query_query_gen_loss(
                    q_reps,
                    q_gen_reps,
                    no_in_batch_neg_flag=no_in_batch_neg_flag,
                )
            else:
                valid_dims = [dim for dim in self.mrl_dims if dim <= q_reps.size(-1)]
                invalid_dims = [dim for dim in self.mrl_dims if dim > q_reps.size(-1)]
                for dim in invalid_dims:
                    logger.warning("Skip mrl_dim %s because hidden size is %s.", dim, q_reps.size(-1))
                if not valid_dims:
                    raise ValueError(
                        f"All mrl_dim values {self.mrl_dims} are larger than hidden size {q_reps.size(-1)}."
                    )

                hidden_size = q_reps.size(-1)
                train_dims = list(valid_dims)
                if hidden_size not in train_dims:
                    train_dims.append(hidden_size)

                query_pos_loss = q_reps.new_zeros(())
                query_gen_pos_loss = q_reps.new_zeros(())
                query_query_gen_loss = q_reps.new_zeros(())
                for dim in train_dims:
                    _, dim_q_pos_loss = compute_loss_func(
                        q_reps,
                        p_reps,
                        teacher_targets=teacher_targets,
                        compute_score_func=self._compute_score_for_dim,
                        mrl_dim=dim,
                    )
                    _, dim_q_gen_pos_loss = compute_loss_func(
                        q_gen_reps,
                        p_reps,
                        teacher_targets=teacher_targets,
                        compute_score_func=self._compute_score_for_dim,
                        mrl_dim=dim,
                    )
                    _, dim_q_q_gen_loss = self._compute_query_query_gen_loss(
                        q_reps,
                        q_gen_reps,
                        no_in_batch_neg_flag=no_in_batch_neg_flag,
                        compute_score_func=self._compute_pair_score_for_dim,
                        mrl_dim=dim,
                    )
                    query_pos_loss += dim_q_pos_loss
                    query_gen_pos_loss += dim_q_gen_pos_loss
                    query_query_gen_loss += dim_q_q_gen_loss

                query_pos_loss = query_pos_loss / len(train_dims)
                query_gen_pos_loss = query_gen_pos_loss / len(train_dims)
                query_query_gen_loss = query_query_gen_loss / len(train_dims)

            weighted_query_pos_loss = self.query_pos_loss_weight * query_pos_loss
            weighted_query_gen_pos_loss = self.query_gen_pos_loss_weight * query_gen_pos_loss
            weighted_query_query_gen_loss = self.query_query_gen_loss_weight * query_query_gen_loss
            loss = weighted_query_pos_loss + weighted_query_gen_pos_loss + weighted_query_query_gen_loss
        else:
            loss = None

        output = DistillEmbedderOutput(loss=loss)
        self.latest_loss_breakdown = None
        if query_pos_loss is not None:
            loss_breakdown = {
                "query_pos_loss": query_pos_loss.detach(),
                "query_gen_pos_loss": query_gen_pos_loss.detach(),
                "query_query_gen_loss": query_query_gen_loss.detach(),
                "weighted_query_pos_loss": weighted_query_pos_loss.detach(),
                "weighted_query_gen_pos_loss": weighted_query_gen_pos_loss.detach(),
                "weighted_query_query_gen_loss": weighted_query_query_gen_loss.detach(),
            }
            self.latest_loss_breakdown = loss_breakdown
            output = DistillEmbedderOutput(loss=loss, **loss_breakdown)
        return output

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
