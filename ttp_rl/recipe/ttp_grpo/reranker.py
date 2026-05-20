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

import os
import torch
import ray
from typing import List, Optional

# Global variable for reranker model path
_RERANKER_MODEL_PATH = None

def set_reranker_model_path(path: str):
    """Set the reranker model path globally."""
    global _RERANKER_MODEL_PATH
    _RERANKER_MODEL_PATH = path

def get_reranker_model_path() -> str:
    """Get the reranker model path."""
    return _RERANKER_MODEL_PATH or "BAAI/bge-reranker-base"

def preprocess_query_for_reranker(query: str) -> str:
    """
    Preprocess query for reranker scoring.
    
    Rules:
    1. Remove instruction prefix (keep only raw query)
    2. Apply query.split('geohash')[0].strip() to remove geohash suffix
    """
    if isinstance(query, (list, tuple)):
        query = str(query[-1]) if query else ""
    else:
        query = str(query)
    
    query = query.split('geohash')[0].strip()
    return query

class RewriteRewardModel:
    """
    A wrapper for BERT-based reward model to compute query-document relevance.
    """
    
    _instance = None
    _instance_path = None
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        lazy_move_to_device: bool = True,
    ):
        if model_name_or_path is None:
            model_name_or_path = get_reranker_model_path()
        
        self.model_path = model_name_or_path
        self.max_length = max_length
        self.batch_size = batch_size
        env_lazy = os.environ.get("RERANKER_LAZY_MOVE")
        if env_lazy is not None:
            self.lazy_move_to_device = env_lazy not in {"0", "false", "False"}
        else:
            self.lazy_move_to_device = lazy_move_to_device
        
        self.device = self._resolve_preferred_device(device)
        self.current_device = "cpu"
        self._device_log = set()
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            
            # Always load on CPU first with float32 (will be converted on first use if lazy)
            # This ensures compatibility with Ray's CUDA_VISIBLE_DEVICES management
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float32,  # Load as float32, convert to fp16 when moving to GPU
            )
            
            # Keep on CPU initially (current_device = "cpu")
            self.model.to(self.current_device)
            
            # If not lazy and GPU is available NOW (after potential CUDA_VISIBLE_DEVICES unset)
            # move to GPU immediately with fp16
            if not self.lazy_move_to_device and self.device.startswith("cuda") and torch.cuda.is_available():
                self.model.to(self.device, dtype=torch.float16)
                self.current_device = self.device
            
            self.model.eval()
            self.initialized = True
            print(
                f"[RewriteRewardModel] Loaded {model_name_or_path} "
                f"(preferred_device={self.device}, current_device={self.current_device}, lazy={self.lazy_move_to_device})"
            )
        except Exception as e:
            print(f"[RewriteRewardModel] Warning: Failed to load model {model_name_or_path}: {e}")
            print("[RewriteRewardModel] Using fallback similarity computation")
            self.initialized = False
    
    @classmethod
    def get_instance(cls, model_name_or_path: Optional[str] = None, **kwargs) -> "RewriteRewardModel":
        if model_name_or_path is None:
            model_name_or_path = get_reranker_model_path()
        
        if cls._instance is None or cls._instance_path != model_name_or_path:
            cls._instance = cls(model_name_or_path=model_name_or_path, **kwargs)
            cls._instance_path = model_name_or_path
        return cls._instance

    @staticmethod
    def _get_env_device() -> Optional[str]:
        env_device = os.environ.get("RERANKER_DEVICE")
        if env_device:
            return env_device
        return None

    def _resolve_preferred_device(self, explicit_device: Optional[str]) -> str:
        if explicit_device:
            return explicit_device

        env_device = self._get_env_device()
        if env_device:
            return env_device

        if torch.cuda.is_available():
            local_rank = os.environ.get("LOCAL_RANK")
            if local_rank is not None:
                return f"cuda:{local_rank}"
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible_devices:
                first_gpu = visible_devices.split(",")[0].strip()
                if first_gpu:
                    return f"cuda:{first_gpu}" if first_gpu.isdigit() else "cuda"
            return "cuda"

        return "cpu"

    def _should_use_gpu(self) -> bool:
        return self.device.startswith("cuda") and torch.cuda.is_available()

    def _move_model_to(self, target_device: str, dtype: Optional[torch.dtype] = None):
        """Move model to target device with specified dtype."""
        # Get actual model dtype (check first parameter)
        try:
            current_dtype = next(self.model.parameters()).dtype
        except:
            current_dtype = None
        
        # Skip if already on target device with target dtype
        if self.current_device == target_device and current_dtype == dtype:
            return
        
        if dtype is None:
            dtype = torch.float16 if target_device.startswith("cuda") else torch.float32
        
        # Move model
        self.model.to(target_device, dtype=dtype)
        self.current_device = target_device

    def _activate_preferred_device(self) -> str:
        """Activate the preferred device for inference."""
        if self._should_use_gpu():
            # Move to GPU with fp16
            self._move_model_to(self.device, dtype=torch.float16)
            return self.device
        # Stay on CPU with fp32
        self._move_model_to("cpu", dtype=torch.float32)
        return "cpu"

    def _release_gpu_if_needed(self):
        """Release GPU memory if using lazy move strategy.
        
        Note: For singleton Ray actors, it's usually better to keep the model on GPU
        after first use. Set RERANKER_ALWAYS_RELEASE=1 to force release after each batch.
        """
        # Check if we should always release GPU
        always_release = os.environ.get("RERANKER_ALWAYS_RELEASE", "0") not in {"0", "false", "False"}
        
        if always_release and self.lazy_move_to_device and self.current_device.startswith("cuda"):
            # Move back to CPU and keep fp16 to avoid conversion overhead
            self.model.to("cpu", dtype=torch.float16)
            self.current_device = "cpu"
            torch.cuda.empty_cache()
    
    @torch.no_grad()
    def compute_score(self, query: str, document: str) -> float:
        scores = self.compute_batch_scores([query], [document])
        return scores[0]
    
    @torch.no_grad()
    def compute_batch_scores(
        self,
        queries: List[str],
        documents: List[str],
    ) -> List[float]:
        import time
        t_batch_start = time.time()
        
        if not queries or not documents:
            return []
        
        if not self.initialized:
            scores = []
            for q, d in zip(queries, documents):
                q_words = set(q.lower().split())
                d_words = set(d.lower().split())
                if not q_words or not d_words:
                    scores.append(0.0)
                else:
                    overlap = len(q_words & d_words)
                    scores.append(overlap / max(len(q_words), 1))
            return scores
        
        all_scores = []
        
        # Activate device once before all batches (move model to GPU if needed)
        run_device = self._activate_preferred_device()
        
        try:
            for batch_idx, i in enumerate(range(0, len(queries), self.batch_size)):
                batch_queries = queries[i:i + self.batch_size]
                batch_docs = documents[i:i + self.batch_size]
                
                inputs = self.tokenizer(
                    batch_queries,
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                
                inputs = inputs.to(run_device)
                outputs = self.model(**inputs)
                
                if outputs.logits.shape[-1] == 1:
                    batch_scores = outputs.logits[:, 0].float().tolist()
                else:
                    probs = torch.softmax(outputs.logits.float(), dim=-1)
                    batch_scores = probs[:, -1].tolist()
                
                all_scores.extend(batch_scores)
        finally:
            # Release GPU after all batches are done (if lazy mode)
            if run_device.startswith("cuda"):
                self._release_gpu_if_needed()
        
        return all_scores

@ray.remote(num_gpus=0)
class RayRewriteRewardModel:
    """Ray Actor for reranker model."""
    
    def __init__(self, model_name_or_path: str, batch_size: int = 32):
        import os
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        # Unhide GPUs (Ray sets CUDA_VISIBLE_DEVICES="" when num_gpus=0)
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            del os.environ["CUDA_VISIBLE_DEVICES"]
        
        # Load model directly on GPU with fp16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16
        ).to("cuda:0").eval()
        
        self.batch_size = batch_size
        self.max_length = 512
        self.call_count = 0
        
        print(f"[Reranker] Ready on cuda:0 (fp16)")

    def compute_batch_scores(self, queries: List[str], documents: List[str]) -> List[float]:
        """Compute reranker scores for query-document pairs."""
        import time
        import torch
        
        self.call_count += 1
        if not queries:
            return []
        
        t_start = time.time()
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(queries), self.batch_size):
                inputs = self.tokenizer(
                    queries[i:i + self.batch_size],
                    documents[i:i + self.batch_size],
                    padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
                ).to("cuda:0")
                
                logits = self._model(**inputs).logits
                scores = logits[:, 0].float().tolist() if logits.shape[-1] == 1 else torch.softmax(logits.float(), dim=-1)[:, -1].tolist()
                all_scores.extend(scores)
        
        if self.call_count <= 3:
            print(f"[Reranker] #{self.call_count}: {len(queries)} pairs in {time.time()-t_start:.2f}s")
        
        return all_scores

def get_ray_reranker_actor(model_path: str) -> ray.actor.ActorHandle:
    """Get or create the singleton Ray actor for reranker."""
    actor_name = "GapGrpoReranker"
    try:
        return ray.get_actor(actor_name)
    except ValueError:
        num_gpus = float(os.environ.get("RERANKER_GPU_COUNT", "0"))
        print(f"[get_ray_reranker_actor] Creating new actor '{actor_name}' with {num_gpus} GPUs")
        
        try:
            return RayRewriteRewardModel.options(
                name=actor_name,
                num_gpus=num_gpus,
                lifetime=None 
            ).remote(model_name_or_path=model_path)
        except ValueError:
            # Race condition handling
            try:
                return ray.get_actor(actor_name)
            except Exception as e:
                print(f"[get_ray_reranker_actor] Critical error getting actor: {e}")
                raise e

