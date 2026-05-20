from typing import Dict, List, Optional, Tuple, Union, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModel(torch.nn.Module):
    """
    A lightweight base class for training/inference that focuses on:
    - Causal LM backbone (AutoModelForCausalLM)
    - No projection / attn-mixing / embed_eos logic
    - Embedding obtained via the hidden state at the <emb> token position
    - DDP-friendly: no DataParallel, no implicit device scatter
    """

    def __init__(
        self,
        model_name_or_path: str,
        normalized: bool = True,
        is_inference: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        tokenizer_name: Optional[str] = None,
        **hf_kwargs,
    ) -> None:
        super().__init__()
        self.normalized=normalized
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        # Backbone
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True, **hf_kwargs
        )
        # `generate` passthrough for convenience
        self.generate = self.model.generate

        # Tokenizer (only when needed)
        self.tokenizer = None
        if is_inference:
            name = tokenizer_name if tokenizer_name else model_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side='right', trust_remote_code=True)
            if not self.tokenizer.pad_token and self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()
            # Let higher-level launcher (HF Trainer / Accelerate) move the model
            if ("device_map" not in hf_kwargs) and (not hf_kwargs.get("load_in_4bit", False)) and (not hf_kwargs.get("load_in_8bit", False)):
                self.model.to(self.device)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def pool_hidden_at_positions(self, hidden_state: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Pool by gathering the hidden state at token positions.
        hidden_state: [b, n, d]
        positions: [b] (index of <emb> per sequence)
        returns: [b, d]
        """
        b, n, d = hidden_state.size()
        idx = positions.to(hidden_state.device).unsqueeze(-1).repeat(1, d).unsqueeze(1)  # [b,1,d]
        return torch.gather(hidden_state, 1, idx).squeeze(1)

    @torch.no_grad()
    def encode_with_emb_token(
        self,
        texts: Union[str, List[str]],
        emb_token: str = "<emb>",
        max_length: int = 512,
        add_special_tokens: bool = False,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Minimal inference helper: tokenize, run forward, gather hidden at the last <emb> token.
        """
        assert self.tokenizer is not None, "Tokenizer not initialized. Set is_inference=True or pass a tokenizer." 
        if isinstance(texts, str):
            texts = [texts]
        tok = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors='pt', add_special_tokens=add_special_tokens,
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**tok, output_hidden_states=False, return_dict=True)
            last_hidden = outputs.last_hidden_state
        emb_id = self.tokenizer.convert_tokens_to_ids(emb_token)
        emb_pos: List[int] = []
        for ids in tok["input_ids"]:
            ids_list = ids.tolist()
            pos = max([i for i, tid in enumerate(ids_list) if tid == emb_id], default=len(ids_list)-1)
            emb_pos.append(pos)
        positions = torch.tensor(emb_pos, dtype=torch.long, device=last_hidden.device)
        reps = self.pool_hidden_at_positions(last_hidden, positions)
        if normalize:
            dtype_in = reps.dtype
            reps = torch.nn.functional.normalize(reps, dim=-1).to(dtype_in)
        return reps

    # Convenience forward that mirrors HF style to keep DDP happy (no DataParallel inside)
    def forward(self, **kwargs):
        return self.model(**kwargs)
