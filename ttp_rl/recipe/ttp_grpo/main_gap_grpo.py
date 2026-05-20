#!/usr/bin/env python3
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
Main entry point for GAP-GRPO training.

This script:
1. Sets up the reranker model path from environment variable
2. Runs the standard verl PPO training with GAP-GRPO config

Usage:
    python -m recipe.ttp_grpo.main_gap_grpo [hydra overrides...]
"""

import os
import sys
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import print_rank_0 for consistent logging
from verl.utils.logger import print_rank_0

# Ensure recipe is in Python path
recipe_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if recipe_path not in sys.path:
    sys.path.insert(0, recipe_path)

# Set reranker model path from environment (for reward computation)
# Note: v5 uses custom reward_manager with optional reranker reward
try:
    from recipe.ttp_grpo.reward_function import set_reranker_model_path
    reranker_path = os.environ.get("RERANKER_MODEL", None)
    if reranker_path:
        set_reranker_model_path(reranker_path)
        print_rank_0(f"[GAP-GRPO v5] Reranker model path set to: {reranker_path}")
except ImportError:
    # v5 uses reranker for optional reranker reward, skip if not available
    pass

# Now import hydra and verl
import hydra
import ray
from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.ray_trainer import Role


# Debugging patches removed for cleaner logs


@hydra.main(config_path="config", config_name="gap_grpo_trainer", version_base=None)
def main(config):
    """Main entry point for GAP-GRPO training with Hydra configuration management."""
    run_gap_grpo(config)


def run_gap_grpo(config):
    """Run GAP-GRPO training with custom Worker."""
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from omegaconf import OmegaConf
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
    
    # Create custom TaskRunner
    runner = GapGRPOTaskRunner.remote()
    ray.get(runner.run.remote(config))
    
    # [Optional] get the path of the timeline trace file from the configuration
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class GapGRPOTaskRunner:
    """Custom TaskRunner for GAP-GRPO v4 that uses GapGRPOV3RolloutRefWorker.
    
    Note: Cannot inherit from TaskRunner (Ray doesn't support actor inheritance).
    Instead, we implement the same interface methods.
    """
    
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}
    
    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker using GapGRPOV3RolloutRefWorker (v4 uses v3 worker for compatibility)."""
        from verl.single_controller.ray import RayWorkerGroup
        from omegaconf import OmegaConf, open_dict
        
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from recipe.ttp_grpo.fsdp_workers import GapGRPOV3RolloutRefWorker
            
            # Ensure gap_config is included in actor_rollout_ref config
            # This allows the worker to access gap_config via self.config.gap_config
            if hasattr(config, 'gap_config') and config.gap_config is not None:
                # Copy gap_config to actor_rollout_ref so worker can access it
                if not hasattr(config.actor_rollout_ref, 'gap_config'):
                    with open_dict(config.actor_rollout_ref):
                        config.actor_rollout_ref.gap_config = config.gap_config
            
            # Support async rollout mode (use default AsyncActorRolloutRefWorker for async mode)
            # Note: GapGRPOV3RolloutRefWorker currently only supports sync mode
            if config.actor_rollout_ref.rollout.mode == "async":
                from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
                actor_rollout_cls = AsyncActorRolloutRefWorker
            else:
                actor_rollout_cls = GapGRPOV3RolloutRefWorker
            
            ray_worker_group_cls = RayWorkerGroup
            
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not supported")
        
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        
        return actor_rollout_cls, ray_worker_group_cls
    
    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker
        else:
            raise NotImplementedError
        
        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
    
    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager
        
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # TODO Here you can use the new registration method to support dynamic registration of roles
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")
            
            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
        
        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id
        
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager
    
    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    from verl.workers.fsdp_workers import RewardModelWorker
                elif config.reward_model.strategy == "megatron":
                    from verl.workers.megatron_workers import RewardModelWorker
                else:
                    raise NotImplementedError
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import RewardModelWorker
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")
            
            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"
    
    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"
    
    def run(self, config):
        """Execute the main PPO training workflow.
        
        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.
        
        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Import all necessary modules
        import os
        import socket
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        from verl.trainer.ppo.reward import load_reward_manager
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        # Explicitly import reward_manager to trigger registration
        from recipe.ttp_grpo import reward_manager
        from verl.utils.config import validate_config
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        # Import custom collate_fn from dataset module (handles extra_info correctly)
        from recipe.ttp_grpo.dataset import collate_fn
        
        OmegaConf.resolve(config)
        
        # Add workers
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)
        
        # Validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )
        
        # Load model
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        
        # Load tokenizer and processor
        from verl.utils import hf_processor, hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        
        # Load reward functions
        # Auto-populate num_generations from rollout.n if not specified in reward_kwargs
        reward_kwargs = dict(config.reward_model.get("reward_kwargs", {}))
        if "num_generations" not in reward_kwargs:
            reward_kwargs["num_generations"] = config.actor_rollout_ref.rollout.n
            print_rank_0(f"[GapGRPOTaskRunner] Auto-set reward_kwargs.num_generations = {reward_kwargs['num_generations']} from rollout.n")
        
        # Validate reward_manager configuration
        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name != "gap_grpo":
            print_rank_0(f"[GapGRPOTaskRunner] ⚠️ Warning: reward_manager={reward_manager_name}, expected 'gap_grpo'")
            print_rank_0(f"[GapGRPOTaskRunner] This may cause incorrect reward computation! Please set reward_model.reward_manager=gap_grpo")
        
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **reward_kwargs
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **reward_kwargs
        )
        
        # Initialize resource pool
        resource_pool_manager = self.init_resource_pool_mgr(config)
        
        # Create datasets
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        print_rank_0(f"[GAP-GRPO] Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        
        # Create trainer (use custom GapGRPOV3Trainer to pass reward_model to workers)
        from recipe.ttp_grpo.gap_grpo_trainer import GapGRPOV3Trainer
        trainer = GapGRPOV3Trainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        
        # Initialize workers
        print_rank_0("[GAP-GRPO] Initializing workers...")
        trainer.init_workers()
        print_rank_0("[GAP-GRPO] Workers ready")
        
        # Start training
        print_rank_0("[GAP-GRPO] Starting training...")
        trainer.fit()
        print_rank_0("[GAP-GRPO] Training completed")


if __name__ == "__main__":
    main()
