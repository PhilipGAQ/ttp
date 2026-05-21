# Copyright 2024 verl-gap authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom FSDP Worker for GAP-GRPO training.

This module provides GapGRPOV3RolloutRefWorker, which extends ActorRolloutRefWorker
to use GapGRPOActor and compute retrieval scores during rollout.
"""

import logging
import os
import re
from typing import List, Optional
import copy

import numpy as np
import torch
import torch.nn.functional as F

from omegaconf import OmegaConf, open_dict
from verl.single_controller.base.decorator import Dispatch, register, make_nd_compute_dataproto_dispatch_fn
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.logger import print_rank_0
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.device import get_device_id
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.config import ActorConfig
from verl import DataProto

logger = logging.getLogger(__name__)


class GapGRPOV3RolloutRefWorker(ActorRolloutRefWorker):
    """
    Custom Worker for GAP-GRPO training.
    
    This worker extends ActorRolloutRefWorker and:
    1. Uses GapGRPOActor instead of the default DataParallelPPOActor when gap_config is present
    2. Computes retrieval scores (original query vs rewritten query) during rollout
    """
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize model with GapGRPOActor if gap_config is present."""
        from verl.workers.actor import DataParallelPPOActor
        
        # Get rank safely
        rank = getattr(self, 'rank', 0)
        
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                from verl.utils.config import omega_conf_to_dataclass
                from verl.workers.config import FSDPEngineConfig
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                from verl.workers.config import FSDPEngineConfig
                fsdp_config = FSDPEngineConfig()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            from verl.utils.fsdp_utils import fsdp_version
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                from verl.utils.fsdp_utils import offload_fsdp_optimizer
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            from verl.utils.config import omega_conf_to_dataclass
            
            # Set struct and update config (same as original verl logic)
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels
            
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            
            # Check if gap_config is present
            gap_config = None
            if hasattr(self.config, 'gap_config'):
                gap_config = self.config.gap_config
            
            # Use GapGRPOActor if gap_config is present, otherwise use default
            if gap_config is not None:
                from recipe.ttp_grpo.gap_actor import GapGRPOActor
                
                # Get tokenizer
                tokenizer = getattr(self, 'tokenizer', None)
                if tokenizer is None:
                    local_path = copy_to_local(
                        self.config.model.path,
                        use_shm=self.config.model.get("use_shm", False)
                    )
                    tokenizer = hf_tokenizer(
                        local_path,
                        trust_remote_code=self.config.model.get("trust_remote_code", False)
                    )
                
                self.actor = GapGRPOActor(
                    config=actor_cfg,
                    actor_module=self.actor_module_fsdp,
                    actor_optimizer=self.actor_optimizer,
                    tokenizer=tokenizer,
                    gap_config=gap_config,
                )
            else:
                # Use default DataParallelPPOActor
                self.actor = DataParallelPPOActor(
                    config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
                )

        if self._is_rollout:
            reward_ref_model_path_config = self.config.model.get("reward_ref_model_path", None)
            
            if "reward_ref_model_path" in self.config.model:
                with open_dict(self.config.model):
                    del self.config.model["reward_ref_model_path"]
            
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
            
            gap_config = None
            if hasattr(self.config, 'gap_config'):
                gap_config = self.config.gap_config
            
            if gap_config is not None:
                tokenizer = getattr(self, 'tokenizer', None)
                if tokenizer is None:
                    local_path = copy_to_local(
                        self.config.model.path,
                        use_shm=self.config.model.get("use_shm", False)
                    )
                    tokenizer = hf_tokenizer(
                        local_path,
                        trust_remote_code=self.config.model.get("trust_remote_code", False)
                    )
                self.rollout_tokenizer = tokenizer
                self.gap_config = gap_config
                
                self.emb_token = gap_config.get("emb_token", "<emb>")
                try:
                    self.emb_token_id = tokenizer.convert_tokens_to_ids(self.emb_token)
                    if self.emb_token_id == tokenizer.unk_token_id:
                        print_rank_0(f"[GapGRPOV3RolloutRefWorker] Warning: emb_token '{self.emb_token}' not in vocabulary")
                        self.emb_token_id = None
                except Exception as e:
                    self.emb_token_id = None
                    print_rank_0(f"[GapGRPOV3RolloutRefWorker] Warning: Failed to get emb_token_id: {e}")

        if self._is_ref:
            from verl.utils.config import omega_conf_to_dataclass
            
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
        
        if self._is_rollout:
            if hasattr(self, 'ref_module_fsdp') and self.ref_module_fsdp is not None:
                self.reward_ref_module_fsdp = self.ref_module_fsdp
                if rank == 0:
                    print_rank_0(f"[GapGRPO] Reusing ref_module as reward_ref_module for stable embeddings")
                    print_rank_0(f"[GapGRPO] This saves ~7GB GPU memory by avoiding duplicate model loading")
            else:
                if rank == 0:
                    print_rank_0(f"[GapGRPO] ref_module not available, loading dedicated reward_ref_module")
                
                reward_ref_model_path = None
                try:
                    reward_ref_model_path = reward_ref_model_path_config
                except NameError:
                    pass
                
                if reward_ref_model_path is None:
                    reward_ref_model_path = self.config.model.path
                
                if rank == 0:
                    print_rank_0(f"[GapGRPO] Loading reward ref model from: {reward_ref_model_path}")
                
                local_path = copy_to_local(reward_ref_model_path, use_shm=use_shm)
                self.reward_ref_module_fsdp = self._build_model_optimizer(
                    model_path=local_path,
                    fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                    optim_config=None,
                    override_model_config=override_model_config,
                    use_remove_padding=use_remove_padding,
                    use_fused_kernels=use_fused_kernels,
                    trust_remote_code=self.config.model.get("trust_remote_code", False),
                    use_liger=self.config.model.get("use_liger", False),
                    role="ref",
                )[0]
                
                self.reward_ref_module_fsdp.eval()
                for param in self.reward_ref_module_fsdp.parameters():
                    param.requires_grad = False
                
                if rank == 0:
                    total_params = sum(p.numel() for p in self.reward_ref_module_fsdp.parameters())
                    print_rank_0(f"[GapGRPO] Reward ref model loaded and frozen ({total_params / 1e9:.2f}B parameters)")
                    print_rank_0(f"[GapGRPO] Reward ref model will be used for computing stable embeddings")

        if self._is_actor:
            from verl.utils.flops_counter import FlopsCounter
            from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
            
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # we need to set actor_module_fsdp for rollout to use
            # This happens when ref_in_actor is False
            pass
    
    def _find_emb_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.size(0)
        positions = torch.full((batch_size,), -1, dtype=torch.long, device=input_ids.device)
        
        if self.emb_token_id is None:
            return positions
        
        emb_mask = (input_ids == self.emb_token_id)
        for i in range(batch_size):
            if emb_mask[i].any():
                pos_indices = emb_mask[i].nonzero(as_tuple=True)[0]
                if len(pos_indices) > 0:
                    positions[i] = pos_indices[-1]
        
        return positions
    
    def _extract_embeddings(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.size()
        
        emb_positions = self._find_emb_positions(input_ids)
        valid_mask = (emb_positions != -1)
        safe_positions = emb_positions.clone()
        safe_positions[~valid_mask] = seq_len - 1
        
        idx = safe_positions.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_dim)
        embeddings = torch.gather(hidden_states, dim=1, index=idx).squeeze(1)
        
        embeddings = F.normalize(embeddings, dim=-1)
        
        return embeddings
    
    def _compute_embeddings_for_texts(
        self,
        texts: List[str],
        model,
    ) -> Optional[torch.Tensor]:
        if not texts or self.rollout_tokenizer is None or self.emb_token_id is None:
            return None
        
        device = get_device_id()
        
        encoded = self.rollout_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            add_special_tokens=False,
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        batch_size = input_ids.size(0)
        max_seq_len = input_ids.size(1)
        
        last_valid_positions = (attention_mask * torch.arange(max_seq_len, device=device).unsqueeze(0)).max(dim=1)[0]
        emb_in_valid = ((input_ids == self.emb_token_id) & (attention_mask > 0)).any(dim=1)
        needs_emb = ~emb_in_valid & (last_valid_positions >= 0)
        
        if needs_emb.any():
            for i in needs_emb.nonzero(as_tuple=True)[0]:
                i = i.item()
                last_pos = last_valid_positions[i].item()
                if last_pos + 1 < max_seq_len:
                    input_ids[i, last_pos + 1] = self.emb_token_id
                    attention_mask[i, last_pos + 1] = 1
                else:
                    input_ids[i, last_pos] = self.emb_token_id
        
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = output.hidden_states[-1]
        
        embeddings = self._extract_embeddings(last_hidden, input_ids)
        
        return embeddings
    
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, prompts: DataProto):
        import asyncio
        from verl.utils.profiler import simple_timer
        from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
        from verl.utils.device import get_device_name, get_torch_device
        
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        meta_info = {
            "eos_token_id": self.model_config.generation_config.eos_token_id
            if self.model_config.generation_config is not None
            else self.model_config.tokenizer.eos_token_id,
            "pad_token_id": self.model_config.generation_config.pad_token_id
            if self.model_config.generation_config is not None
            else self.model_config.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        # Compute retrieval scores if gap_config is present
        batch_size = len(output) if hasattr(output, '__len__') else 0
        if "retrieval_scores" not in output.non_tensor_batch:
            if batch_size > 0:
                output.non_tensor_batch["retrieval_scores"] = np.array([None] * batch_size, dtype=object)
            else:
                output.non_tensor_batch["retrieval_scores"] = np.array([], dtype=object)
        else:
            existing_scores = output.non_tensor_batch["retrieval_scores"]
            if not isinstance(existing_scores, np.ndarray):
                if batch_size > 0:
                    output.non_tensor_batch["retrieval_scores"] = np.array([None] * batch_size, dtype=object)
                else:
                    output.non_tensor_batch["retrieval_scores"] = np.array([], dtype=object)
            elif existing_scores.ndim == 0:
                if batch_size > 0:
                    output.non_tensor_batch["retrieval_scores"] = np.array([None] * batch_size, dtype=object)
                else:
                    output.non_tensor_batch["retrieval_scores"] = np.array([], dtype=object)
        
        gap_config = getattr(self, 'gap_config', None)
        if gap_config is not None and hasattr(self, 'rollout_tokenizer') and self.emb_token_id is not None:
            try:
                # Get model for embedding computation
                model = getattr(self, 'reward_ref_module_fsdp', None)
                if model is None:
                    model = getattr(self, 'actor_module_fsdp', None)
                    if model is None:
                        model = getattr(self, 'actor_module', None)
                    print_rank_0("[GapGRPO] Warning: reward_ref_module not found, falling back to actor_module for embeddings")
                else:
                    print_rank_0("[GapGRPO] Using reward_ref_module for stable embedding computation")
                
                if model is not None:
                    model.eval()
                    
                    all_ori_queries = []
                    all_rewritten_queries = []
                    all_passages_list = []  # List of lists
                    valid_indices = []
                    
                    skipped_reasons = {"no_ground_truth": 0, "no_original_query": 0, "no_passages": 0, "valid": 0}
                    
                    think_open = gap_config.get("think_open", "<think>")
                    think_close = gap_config.get("think_close", "</think>")
                    mask_hist = gap_config.get("mask_hist", True)
                    
                    for i in range(batch_size):
                        data_item = output[i]
                        
                        prompt_ids = data_item.batch["prompts"]
                        response_ids = data_item.batch["responses"]
                        
                        prompt_str = self.rollout_tokenizer.decode(prompt_ids, skip_special_tokens=False)
                        response_str = self.rollout_tokenizer.decode(response_ids, skip_special_tokens=False)
                        pattern = rf"{re.escape(think_open)}(.+?){re.escape(think_close)}\s*{re.escape(self.emb_token)}"
                        match = re.search(pattern, response_str, re.DOTALL)
                        
                        if match:
                            rewritten_query = match.group(1).strip()
                        else:
                            rewritten_query = response_str.strip()
                        
                        ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", {})
                        
                        if not ground_truth:
                            skipped_reasons["no_ground_truth"] += 1
                            continue
                            
                        original_query = ground_truth.get("query", "")
                        if isinstance(original_query, (list, tuple)):
                            original_query = str(original_query[-1]) if original_query else ""
                        else:
                            original_query = str(original_query)
                        
                        if not original_query:
                            skipped_reasons["no_original_query"] += 1
                            continue
                        
                        clean_query = original_query.split("geohash")[0].strip()
                        
                        # mask_hist=True: use query_prompt_no_hist (without history)
                        # mask_hist=False: use prompt_str (full prompt with history)
                        if mask_hist:
                            query_prompt_for_emb = ground_truth.get("query_prompt_no_hist", prompt_str)
                        else:
                            query_prompt_for_emb = prompt_str
                        
                        pos_passages = ground_truth.get("pos", [])
                        if isinstance(pos_passages, str): pos_passages = [pos_passages]
                        neg_passages = ground_truth.get("neg", [])
                        if isinstance(neg_passages, str): neg_passages = [neg_passages]
                        all_passages = (pos_passages if isinstance(pos_passages, list) else []) + \
                                      (neg_passages if isinstance(neg_passages, list) else [])
                        
                        if not all_passages:
                            skipped_reasons["no_passages"] += 1
                            continue
                        
                        ori_text = f"{query_prompt_for_emb}{think_open}{clean_query}{think_close}{self.emb_token}"
                        rew_text = f"{query_prompt_for_emb}{think_open}{rewritten_query}{think_close}{self.emb_token}"
                        
                        all_ori_queries.append(ori_text)
                        all_rewritten_queries.append(rew_text)
                        all_passages_list.append((all_passages, len(pos_passages)))
                        valid_indices.append(i)
                        skipped_reasons["valid"] += 1

                    if skipped_reasons.get("no_ground_truth", 0) > 0 or skipped_reasons.get("no_original_query", 0) > 0 or skipped_reasons.get("no_passages", 0) > 0:
                        print_rank_0(f"[GapGRPOV3RolloutRefWorker] Warning: Sample collection issues - {skipped_reasons}, valid_indices: {len(valid_indices)}")
                    
                    # Compute embeddings and similarity scores for valid samples
                    if valid_indices:
                        print_rank_0(f"[GapGRPOV5] ========== Starting reward_ref_module inference (valid_samples={len(valid_indices)}/{batch_size}) ==========")
                        # Compute original query embeddings
                        print_rank_0(f"[GapGRPOV5] Computing original query embeddings using reward_ref_module (batch_size={len(all_ori_queries)})")
                        ori_embs = self._compute_embeddings_for_texts(all_ori_queries, model)
                        print_rank_0(f"[GapGRPOV5] ✓ Original query embeddings computed: shape={ori_embs.shape if ori_embs is not None else 'None'}")
                        
                        # Compute rewritten query embeddings
                        print_rank_0(f"[GapGRPOV5] Computing rewritten query embeddings using reward_ref_module (batch_size={len(all_rewritten_queries)})")
                        rewritten_embs = self._compute_embeddings_for_texts(all_rewritten_queries, model)
                        print_rank_0(f"[GapGRPOV5] ✓ Rewritten query embeddings computed: shape={rewritten_embs.shape if rewritten_embs is not None else 'None'}")
                        
                        # Compute passage embeddings (flatten all passages)
                        passage_start_indices = [0]
                        all_passages_flat = []
                        for passages, _num_pos in all_passages_list:
                            all_passages_flat.extend(passages)
                            passage_start_indices.append(len(all_passages_flat))
                        
                        print_rank_0(f"[GapGRPOV5] Computing passage embeddings using reward_ref_module (batch_size={len(all_passages_flat)})")
                        passage_embs = self._compute_embeddings_for_texts(all_passages_flat, model)
                        print_rank_0(f"[GapGRPOV5] ✓ Passage embeddings computed: shape={passage_embs.shape if passage_embs is not None else 'None'}")
                        
                        if ori_embs is not None and rewritten_embs is not None and passage_embs is not None:
                            # Process each valid sample
                            for idx, i in enumerate(valid_indices):
                                # Get passage embeddings for this sample
                                start_idx = passage_start_indices[idx]
                                end_idx = passage_start_indices[idx + 1]
                                sample_passage_embs = passage_embs[start_idx:end_idx]
                                
                                # Get query embeddings
                                ori_emb = ori_embs[idx:idx+1]
                                rewritten_emb = rewritten_embs[idx:idx+1]
                                
                                # Normalize embeddings
                                ori_emb_norm = F.normalize(ori_emb, dim=-1)
                                rewritten_emb_norm = F.normalize(rewritten_emb, dim=-1)
                                sample_passage_embs_norm = F.normalize(sample_passage_embs, dim=-1)
                                
                                # Compute similarity scores
                                temperature = gap_config.get("infonce_temperature", 0.02)
                                ori_scores = torch.matmul(ori_emb_norm, sample_passage_embs_norm.T) / temperature
                                rewritten_scores = torch.matmul(rewritten_emb_norm, sample_passage_embs_norm.T) / temperature
                                
                                # Store retrieval scores in a SEPARATE key to avoid union_numpy_dict AssertionError
                                # We cannot modify extra_info here because union_numpy_dict requires exact match
                                # Instead, store in a new key that will be merged after union
                                # Convert to float32 before numpy conversion (BFloat16 not supported by numpy)
                                retrieval_data = {
                                    "ori_query_embedding": ori_emb[0].float().cpu().numpy(),
                                    "rewritten_query_embedding": rewritten_emb[0].float().cpu().numpy(),
                                    "passage_embeddings": sample_passage_embs.float().cpu().numpy(),
                                    "ori_sim_scores": ori_scores[0].float().cpu().numpy(),
                                    "rewritten_sim_scores": rewritten_scores[0].float().cpu().numpy(),
                                    "num_pos_passages": all_passages_list[idx][1] if idx < len(all_passages_list) else 0,
                                }
                                output.non_tensor_batch["retrieval_scores"][i] = retrieval_data
                            
                            print_rank_0(f"[GapGRPOV5] ========== Reward_ref_module inference completed successfully ({len(valid_indices)} samples processed) ==========")

                    keys_to_delete = ["extra_info", "pos_text", "neg_texts", "query_prompt_no_hist"]
                    for key in keys_to_delete:
                        if key in output.non_tensor_batch:
                            del output.non_tensor_batch[key]
                                
            except Exception as e:
                import traceback
                print_rank_0(f"[GapGRPOV5] ❌ Error: Reward_ref_module inference failed: {e}")
                print_rank_0(f"[GapGRPOV5] Traceback:\n{traceback.format_exc()}")

                batch_size = len(output) if hasattr(output, '__len__') else 0
                if "retrieval_scores" not in output.non_tensor_batch:
                    # Ensure retrieval_scores is always at least 1D array (even if batch_size == 0)
                    # This prevents IndexError in chunk() when np.array_split tries to access shape[0]
                    if batch_size > 0:
                        output.non_tensor_batch["retrieval_scores"] = np.array([None] * batch_size, dtype=object)
                    else:
                        # Create empty 1D array instead of 0D scalar
                        output.non_tensor_batch["retrieval_scores"] = np.array([], dtype=object)
                else:
                    # Ensure existing retrieval_scores is at least 1D (fix if corrupted)
                    existing_scores = output.non_tensor_batch["retrieval_scores"]
                    if not isinstance(existing_scores, np.ndarray) or existing_scores.ndim == 0:
                        if batch_size > 0:
                            output.non_tensor_batch["retrieval_scores"] = np.array([None] * batch_size, dtype=object)
                        else:
                            output.non_tensor_batch["retrieval_scores"] = np.array([], dtype=object)
                
                keys_to_delete = ["extra_info", "pos_text", "neg_texts", "query_prompt_no_hist"]
                for key in keys_to_delete:
                    if key in output.non_tensor_batch:
                        del output.non_tensor_batch[key]
        
        keys_to_delete = ["extra_info", "pos_text", "neg_texts", "query_prompt_no_hist"]
        for key in keys_to_delete:
            if key in output.non_tensor_batch:
                del output.non_tensor_batch[key]

        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        if hasattr(output, 'non_tensor_batch') and output.non_tensor_batch:
            current_batch_size = len(output) if hasattr(output, '__len__') else 0
            for key, val in list(output.non_tensor_batch.items()):
                if isinstance(val, np.ndarray):
                    if val.ndim == 0:
                        if current_batch_size > 0:
                            output.non_tensor_batch[key] = np.array([val.item()] * current_batch_size, dtype=val.dtype)
                        else:
                            output.non_tensor_batch[key] = np.array([], dtype=val.dtype)
                elif not isinstance(val, np.ndarray):
                    if current_batch_size > 0:
                        output.non_tensor_batch[key] = np.array([val] * current_batch_size, dtype=object)
                    else:
                        output.non_tensor_batch[key] = np.array([], dtype=object)

        get_torch_device().empty_cache()
        return output
