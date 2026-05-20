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
Custom PPO Trainer for GAP-GRPO v3.

Extends RayPPOTrainer to ensure reward_model is passed to workers for embedding computation.
"""

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl import DataProto


class GapGRPOV3Trainer(RayPPOTrainer):
    """
    Custom trainer for GAP-GRPO v3 that ensures reward_model is available to workers.
    
    The key difference from RayPPOTrainer is that _get_gen_batch does NOT pop
    reward_model from gen_batch, allowing workers to access ground_truth for
    embedding computation during rollout.
    """
    
    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """
        Override to ensure reward_model and related keys are passed to workers.
        
        Unlike the base implementation which keeps reward_model in the original batch,
        we include it in gen_batch so workers can compute embeddings using ground_truth.
        """
        # Keys that workers need for embedding computation
        worker_required_keys = {"reward_model", "extra_info", "pos_text", "neg_texts", "query_prompt_no_hist"}
        
        # Keys that should stay in both batch and gen_batch
        shared_keys = worker_required_keys & set(batch.non_tensor_batch.keys())
        
        # Keys to pop for generation (everything except shared_keys and uid/data_source)
        preserve_keys = shared_keys | {"uid", "data_source"}
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - preserve_keys
        
        # Pop tensor batch keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )
        
        # 🔑 CRITICAL: Manually add shared_keys to gen_batch
        # Since pop doesn't include them, we need to copy them over
        for key in shared_keys:
            if key in batch.non_tensor_batch:
                gen_batch.non_tensor_batch[key] = batch.non_tensor_batch[key]
        
        # For agent loop, we need reward model keys to compute score
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        
        return gen_batch

