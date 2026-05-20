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
GAP-GRPO: Generative Augmented Passage Retrieval with GRPO Training

This recipe implements GRPO training for a model that can do both 
generation (query rewriting) and embedding (retrieval) tasks.

Key components:
- GapGRPODataset: Dataset class for gap-r1 format data
- GapGRPOActor: Extended actor with embedding extraction and InfoNCE loss
- compute_score: Reward function (format + rewrite gain + retrieval)
- set_reranker_model_path: Set global reranker model path
- patch_fsdp_workers: Patch verl workers to use GapGRPOActor
"""

from .dataset import GapGRPODataset
from .reward_function import (
    compute_score,
    compute_score_batch,
    set_reranker_model_path,
    get_reranker_model_path,
)
from .gap_actor import GapGRPOActor

__all__ = [
    "GapGRPODataset",
    "GapGRPOActor", 
    "compute_score",
    "compute_score_batch",
    "set_reranker_model_path",
    "get_reranker_model_path",
]

