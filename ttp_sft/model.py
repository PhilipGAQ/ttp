from dataclasses import dataclass
import logging
from typing import Dict, Optional, List, Union

import torch
import torch.distributed as dist
from torch import Tensor
from transformers.file_utils import ModelOutput

from .base_model import BaseModel


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class TrainOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    loss_emb: Optional[Tensor] = None
    loss_gen: Optional[Tensor] = None


class DistributedContrastiveLoss:
    def __init__(self, temperature: float, negatives_cross_device: bool=True):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('negatives_cross_device requires distributed training')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def __call__(self, q_reps: torch.Tensor, p_reps: torch.Tensor, group_size: int):
        """
        q_reps: [B_local, D]
        p_reps: [B_local * group_size, D]
        FlagEmbedding style: when negatives_cross_device=True, gather both queries and passages
        across devices, then compute scores using global tensors. Target is simply idx * group_size
        for each query in the global ordering.
        """
        if self.negatives_cross_device:
            # Gather both queries and passages across devices (FlagEmbedding style)
            cross_q_reps = self._gather(q_reps)  # [world_size * B_local, D]
            cross_p_reps = self._gather(p_reps)  # [world_size * B_local * group_size, D]
            
            # Compute scores using global tensors
            scores = torch.matmul(cross_q_reps, cross_p_reps.transpose(0, 1)) / self.temperature  # [world*B_local, world*B_local*group_size]
            
            # Target: each query i's positive is at column i * group_size (no rank offset needed)
            cross_idxs = torch.arange(cross_q_reps.size(0), device=scores.device, dtype=torch.long)
            targets = cross_idxs * group_size  # [world_size * B_local]
            
            return self.cross_entropy(scores, targets)
        else:
            # Local in-batch negatives only
            B_local = q_reps.size(0)
            scores = torch.matmul(q_reps, p_reps.transpose(0, 1)) / self.temperature  # [B_local, B_local*group_size]
            targets = torch.arange(B_local, device=scores.device, dtype=torch.long) * group_size
            return self.cross_entropy(scores, targets)

    def _gather(self, t: Optional[torch.Tensor]):
        if t is None: return None
        t = t.contiguous()
        bufs = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(bufs, t)
        bufs[self.rank] = t
        return torch.cat(bufs, dim=0)


class NextTokenLoss:
    def __init__(self, vocab_size: int, loss_gen_factor: float = 1.0):
        self.vocab_size = vocab_size
        self.loss_gen_factor = loss_gen_factor
        self.ce = torch.nn.CrossEntropyLoss(reduction="mean")

    def __call__(self, labels, logits):
        shift_logits = logits[..., :-1, :].contiguous().view(-1, self.vocab_size)
        shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
        loss = self.ce(shift_logits, shift_labels)
        return loss * self.loss_gen_factor


class QueryGenModel(BaseModel):
    """
    Training model for query-side generation + embedding, passage-side embedding only.
    
    - Query side: generation task (think to embed token) + embedding at <emb> token
    - Passage side: embedding only at <emb> token
    
    embedding_view controls what the query embedding can see:
    - "query": embedding only sees query part (before <think>)
    - "gen": embedding only sees generation part (<think> to </think>)
    - "both": embedding sees everything
    
    mask_history: if True, mask out history sequence so embedding only sees query + generation
    """

    def __init__(
        self,
        temperature: float = 0.02,
        negatives_cross_device: bool = True,
        loss_gen_factor: float = 1.0,
        loss_contrast_factor: float = 1.0,
        embedding_view: str = "query",
        mask_history: bool = True,
        mrl_dim: str = None,
        sub_batch_size: int = -1,
        is_inference=False,
        **kwargs,
    ):
        super().__init__(**kwargs, is_inference=is_inference)
        # Validate that at least one loss factor is non-zero
        if loss_gen_factor == 0 and loss_contrast_factor == 0:
            raise ValueError(
                "loss_gen_factor and loss_contrast_factor cannot both be 0. "
                "At least one must be > 0."
            )
        self.loss_gen_factor = loss_gen_factor
        self.loss_contrast_factor = loss_contrast_factor
        self.emb_loss_fn = DistributedContrastiveLoss(temperature, negatives_cross_device)
        self.gen_loss_fn = NextTokenLoss(self.model.config.vocab_size, loss_gen_factor)
        self.config = self.model.config
        assert embedding_view in ("query", "gen", "both")
        self.embedding_view = embedding_view
        self.mask_history = mask_history
        self.sub_batch_size = sub_batch_size
        self.mrl_dims = [int(x) for x in mrl_dim.split(',')] if mrl_dim else []

    def encode(self, features, normalize: bool = None):
        """Simple encoding for passage side (no generation, just embedding)"""
        if features is None: return None
        attention_mask = features.get('attention_mask')
        kwargs = {'input_ids': features.get('input_ids'), 'attention_mask': attention_mask}
        out = self.model(**kwargs, return_dict=True, output_hidden_states=True).hidden_states[-1]
        # Prefer pooling at <emb> positions if provided
        emb_pos = features.get('embed_pos')
        if emb_pos is not None:
            b, n, d = out.size()
            idx = emb_pos.to(out.device).unsqueeze(-1).repeat(1, d).unsqueeze(1)
            reps = torch.gather(out, 1, idx).squeeze(1)
        else:
            # fallback to masked mean
            s = torch.sum(out * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)
            reps = s / d
        
        do_norm = normalize if normalize is not None else self.normalized
        if do_norm:
            in_dtype = reps.dtype
            return torch.nn.functional.normalize(reps, dim=-1).contiguous().to(in_dtype)
        return reps.contiguous()

    def forward(
        self,
        query: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]], None] = None,
        passage: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]], None] = None,
        q_grad: bool = True,
        p_grad: bool = True,
    ):
        loss_gen = None
        q_reps = None
        p_reps = None
        
        # If MRL is used, we need unnormalized embeddings to slice them properly
        # If MRL is NOT used, internal_normalize is None -> uses self.normalized (default behavior)
        internal_normalize = False if self.mrl_dims else None

        # Handle query: generation task + embedding
        if query is not None:
            if isinstance(query, list):
                # Collator returned list of sub-batches
                all_q_reps = []
                all_loss_gen = []
                for q_sub in query:
                    q_reps_sub, loss_gen_sub = self._encode_query_sub_batch(q_sub, normalize=internal_normalize)
                    all_q_reps.append(q_reps_sub)
                    if loss_gen_sub is not None:
                        all_loss_gen.append(loss_gen_sub)
                
                q_reps = torch.cat(all_q_reps, dim=0)
                
                if all_loss_gen:
                    loss_gen = sum(all_loss_gen) / len(all_loss_gen)
                else:
                    loss_gen = None
            else:
                # Single batch dict - check if we need model-level sub-batch processing
                if self.sub_batch_size is not None and self.sub_batch_size > 0:
                    all_q_reps = []
                    all_loss_gen = []
                    query_batch_size = query['attention_mask'].size(0)
                    
                    for i in range(0, query_batch_size, self.sub_batch_size):
                        end_idx = min(i + self.sub_batch_size, query_batch_size)
                        query_sub = {}
                        for k, v in query.items():
                            if isinstance(v, torch.Tensor):
                                query_sub[k] = v[i:end_idx]
                            else:
                                query_sub[k] = v
                        
                        q_reps_sub, loss_gen_sub = self._encode_query_sub_batch(query_sub, normalize=internal_normalize)
                        all_q_reps.append(q_reps_sub)
                        if loss_gen_sub is not None:
                            all_loss_gen.append(loss_gen_sub)
                    
                    q_reps = torch.cat(all_q_reps, dim=0)
                    
                    if all_loss_gen:
                        loss_gen = sum(all_loss_gen) / len(all_loss_gen)
                    else:
                        loss_gen = None
                else:
                    # Process entire batch at once
                    q_reps, loss_gen = self._encode_query_sub_batch(query, normalize=internal_normalize)

        # Handle passage: embedding only (no generation)
        # Skip passage processing if contrastive loss is disabled (SFT-only mode)
        if self.loss_contrast_factor > 0 and passage is not None:
            if isinstance(passage, list):
                # Collator returned list of sub-batches
                all_p_reps = []
                for p_sub in passage:
                    p_reps_sub = self.encode(p_sub, normalize=internal_normalize) if p_grad else self.encode_no_grad(p_sub, normalize=internal_normalize)
                    all_p_reps.append(p_reps_sub)
                p_reps = torch.cat(all_p_reps, dim=0)
            else:
                # Single batch dict
                p_reps = self.encode(passage, normalize=internal_normalize) if p_grad else self.encode_no_grad(passage, normalize=internal_normalize)
        else:
            # Skip passage embedding when contrastive loss is disabled
            p_reps = None

        # Compute contrastive loss (only if loss_contrast_factor > 0)
        loss_emb = None
        if self.loss_contrast_factor > 0 and (q_reps is not None) and (p_reps is not None):
            group_size = max(1, p_reps.size(0) // max(q_reps.size(0), 1))
            
            # If MRL is enabled, we compute multiple losses
            if self.mrl_dims:
                losses = []
                
                # 1. Full dimension loss
                q_full = torch.nn.functional.normalize(q_reps, dim=-1)
                p_full = torch.nn.functional.normalize(p_reps, dim=-1)
                losses.append(self.emb_loss_fn(q_full, p_full, group_size))
                
                # 2. MRL dimensions
                for d in self.mrl_dims:
                    if d > q_reps.size(-1):
                        continue
                    q_slice = torch.nn.functional.normalize(q_reps[:, :d], dim=-1)
                    p_slice = torch.nn.functional.normalize(p_reps[:, :d], dim=-1)
                    losses.append(self.emb_loss_fn(q_slice, p_slice, group_size))
                
                loss_emb_raw = sum(losses) / len(losses)
                
                # Normalize output reps if needed (to match expected output format)
                if self.normalized:
                    in_dtype = q_reps.dtype
                    q_reps = torch.nn.functional.normalize(q_reps, dim=-1).contiguous().to(in_dtype)
                    p_reps = torch.nn.functional.normalize(p_reps, dim=-1).contiguous().to(in_dtype)
            else:
                # Standard path
                loss_emb_raw = self.emb_loss_fn(q_reps, p_reps, group_size)

            loss_emb = loss_emb_raw * self.loss_contrast_factor
        elif self.loss_contrast_factor == 0:
            # Skip contrastive loss computation when disabled
            loss_emb = None

        # Combine losses
        loss = None
        if (loss_emb is not None) and (loss_gen is not None):
            loss = loss_emb + loss_gen
        elif loss_emb is not None:
            loss = loss_emb
        elif loss_gen is not None:
            loss = loss_gen
        else:
            # This should not happen due to validation in __init__, but add safety check
            raise RuntimeError("Both loss_gen and loss_emb are None. This should not happen.")

        return TrainOutput(q_reps=q_reps, p_reps=p_reps, loss=loss, loss_emb=loss_emb, loss_gen=loss_gen)

    def encode_no_grad(self, features, normalize: bool = None):
        with torch.no_grad():
            return self.encode(features, normalize=normalize)

    def _encode_query_sub_batch(self, query_sub: Dict[str, torch.Tensor], normalize: bool = None):
        """
        Encode a sub-batch of queries with generation task and embedding.
        
        embedding_view controls what the embedding can see:
        - "query": mask out gen part ([gen_start, gen_end]) for the <emb> query row
        - "gen": mask out query part ([0, gen_start-1]) for the <emb> query row
        - "both": no masking, embedding sees everything
        
        mask_history: if True, additionally mask out history sequence ([hist_start, hist_end])
        """
        # Determine if we need custom mask
        need_embedding_view_mask = (self.embedding_view in ("query", "gen")) and ("gen_start" in query_sub) and ("embed_pos" in query_sub)
        need_history_mask = self.mask_history and ("hist_start" in query_sub) and ("embed_pos" in query_sub)
        use_custom_mask = need_embedding_view_mask or need_history_mask
        
        attn_mask = query_sub.get('attention_mask')
        excluded_keys = ['labels', 'embed_pos', 'gen_start', 'gen_end', 'hist_start', 'hist_end']
        
        if use_custom_mask and attn_mask is not None:
            bsz, seqlen = attn_mask.size()
            device = attn_mask.device
            dtype = torch.float32
            # base4: 0 keep, -inf mask
            base4 = torch.zeros((bsz, 1, seqlen, seqlen), device=device, dtype=dtype)
            # mask padded keys (columns where attn_mask==0)
            pad_keys = (attn_mask == 0).to(dtype)
            base4 = base4 + pad_keys.view(bsz, 1, 1, seqlen) * (-1e9)
            # causal mask: mask j > i
            causal = torch.triu(torch.ones((seqlen, seqlen), device=device, dtype=dtype), diagonal=1)
            base4 = base4 + causal.view(1, 1, seqlen, seqlen) * (-1e9)
            
            # Modify rows corresponding to <emb> token
            for i in range(bsz):
                epos = int(query_sub['embed_pos'][i].item())
                
                # Mask history if enabled
                if need_history_mask:
                    h_start = int(query_sub['hist_start'][i].item())
                    h_end = int(query_sub['hist_end'][i].item())
                    if (0 <= epos < seqlen) and (h_start >= 0) and (h_end >= h_start) and (h_end < seqlen):
                        base4[i, 0, epos, h_start:h_end+1] = -1e9
                
                # Mask based on embedding_view
                if need_embedding_view_mask:
                    s = int(query_sub['gen_start'][i].item())
                    e = int(query_sub['gen_end'][i].item())
                    
                    if self.embedding_view == "query":
                        # Mask out gen part for the <emb> query row
                        if (0 <= epos < seqlen) and (s >= 0) and (e >= s) and (e < seqlen):
                            base4[i, 0, epos, s:e+1] = -1e9
                    elif self.embedding_view == "gen":
                        # Mask out query part for the <emb> query row
                        if (0 <= epos < seqlen) and (s > 0):
                            base4[i, 0, epos, 0:s] = -1e9
            
            query_inputs = {k: v for k, v in query_sub.items() if k not in excluded_keys}
            outputs = self.model(**{**query_inputs, 'attention_mask': base4}, return_dict=True, output_hidden_states=True)
        else:
            query_inputs = {k: v for k, v in query_sub.items() if k not in excluded_keys}
            outputs = self.model(**query_inputs, return_dict=True, output_hidden_states=True)
        
        # Compute generation loss if labels are present and loss_gen_factor != 0
        loss_gen = None
        labels = query_sub.get('labels')
        
        if self.loss_gen_factor != 0 and labels is not None and hasattr(outputs, 'logits'):
            # Check if there are any valid labels (not all -100)
            if (labels != -100).any():
                logits = outputs.logits
                loss_gen = self.gen_loss_fn(labels, logits)
        
        # Extract embeddings at <emb> position
        last_hidden = outputs.hidden_states[-1]
        emb_pos = query_sub.get('embed_pos')
        if emb_pos is not None:
            b, n, d = last_hidden.size()
            idx = emb_pos.to(last_hidden.device).unsqueeze(-1).repeat(1, d).unsqueeze(1)  # [b,1,d]
            q_reps = torch.gather(last_hidden, 1, idx).squeeze(1)
        else:
            # Fallback to masked mean pooling if embed_pos is not available
            attention_mask = query_sub.get('attention_mask')
            if attention_mask is not None:
                # Handle 4D attention mask
                if attention_mask.dim() == 4:
                    attention_mask = (query_sub.get('attention_mask') != 0).any(dim=(1, 2)).long()
                s = torch.sum(last_hidden * attention_mask.unsqueeze(-1).float(), dim=1)
                d = attention_mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)
                q_reps = s / d
            else:
                # Last resort: use last token
                q_reps = last_hidden[:, -1]
        
        do_norm = normalize if normalize is not None else self.normalized
        if do_norm:
            in_dtype = q_reps.dtype
            q_reps = torch.nn.functional.normalize(q_reps, dim=-1).contiguous().to(in_dtype)
        else:
            q_reps = q_reps.contiguous()
        
        return q_reps, loss_gen

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)