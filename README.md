# TTP: Think-to-Personalize

Official implementation of **"Think-to-Personalize: Unifying Reasoning and Retrieval for User-Centric Personalized Dense Retrieval"**.

TTP is a novel framework that unifies explicit user-centric intent reasoning with dense retrieval. By reasoning over the user's historical purchase sequence, TTP explicitly deduces latent personalized needs and generates an intent-enhanced query, which is then encoded into a unified dense embedding.

## Framework Overview

TTP employs a two-stage training paradigm built on top of an LLM-based embedding model:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TTP Training Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stage 1: Cold-Start SFT                                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Input: [Instruction] [User History] [Query]                   │ │
│  │  Output: <think> Intent-Enhanced Query </think> <embed>        │ │
│  │  Loss: L = λ_gen · L_gen + L_cl (InfoNCE)                     │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  Stage 2: GRPO Reinforcement Learning                               │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Policy Optimization: GRPO (Group Relative Policy Opt.)        │ │
│  │  Reward: R = λ_fmt·R_format + λ_len·R_length + λ_ret·R_retr   │ │
│  │  + Dynamic Positive Selection for InfoNCE                      │ │
│  │  Loss: L = L_GRPO + λ_cl · L_cl                               │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  (Optional) Distill Retriever for Online Deployment                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Distill TTP (3B) → Encoder-only Bi-encoder (305M)            │ │
│  │  Three pairs: ⟨q, p+⟩, ⟨q_r, p+⟩, ⟨q, q_r⟩ (1:1:1)          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
ttp/
├── README.md
├── .gitignore
├── scripts/                    # Training launch scripts
│   ├── run_sft.sh              # Stage 1: Cold-Start SFT
│   ├── run_grpo.sh             # Stage 2: GRPO RL Alignment
│   ├── run_distill_retriever.sh# Distill Retriever (deployment)
│   └── accelerate_config.yaml  # Distributed training config
├── ttp_sft/                    # Stage 1: SFT training code
│   ├── model.py                # LLM-based embedding model with generation + contrastive
│   ├── data.py                 # Dataset with reasoning-enhanced format
│   ├── trainer.py              # Joint NTP + InfoNCE training loop
│   ├── base_model.py           # Base model architecture
│   ├── arguments.py            # Training arguments
│   └── run.py                  # Entry point
├── ttp_rl/                     # Stage 2: GRPO RL training code (based on VeRL)
│   ├── recipe/ttp_grpo/        # TTP-specific GRPO recipe
│   │   ├── main_gap_grpo.py    # Entry point
│   │   ├── gap_actor.py        # Custom actor with InfoNCE + Dynamic Positive Selection
│   │   ├── reward_function.py  # Format + Length + Retrieval reward
│   │   ├── reward_manager.py   # Reward orchestration with frozen ref model
│   │   ├── gap_grpo_trainer.py # Custom GRPO trainer
│   │   ├── fsdp_workers.py     # FSDP distributed workers
│   │   ├── dataset.py          # RL dataset processing
│   │   ├── reranker.py         # Reranker-based reward model
│   │   ├── utils.py            # Retrieval reward computation
│   │   └── config/gap_grpo_trainer.yaml  # Hydra config
│   └── verl/                   # VeRL framework (RL infrastructure)
└── distill_retriever/          # Distill Retriever training code
    ├── modeling.py             # Bi-encoder architecture
    ├── dataset.py              # Distillation dataset
    ├── trainer.py              # Contrastive distillation trainer
    └── arguments.py            # Training arguments
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- 8x NVIDIA A100 80GB GPUs
- Key dependencies:
  - `transformers`, `accelerate` (for SFT and Distillation)
  - `vllm` (for RL rollout generation)
  - `ray` (for distributed RL training)
  - `hydra-core` (for RL config management)

Install dependencies:

```bash
# For SFT and Distill Retriever
pip install transformers accelerate peft deepspeed

# For GRPO RL (install verl)
cd ttp_rl
pip install -e .
```

## Training

### Data Preparation

TTP requires reasoning-enhanced training data constructed via a teacher model (Qwen3-32B):

1. Filter user history based on purchase records (past year)
2. Use bge-reranker-v2-m3 to select top-N (N=10) relevant history items per query
3. Prompt Qwen3-32B to generate K=5 candidate intent-enhanced queries
4. Select the best rewrite by relevance gain over the original query
5. Retain samples with positive gain; also keep a proportion of neutral samples

### Stage 1: Cold-Start SFT

Establishes the model's capability to follow the reasoning-and-retrieval format while learning retrieval-oriented representations.

```bash
bash scripts/run_sft.sh
```

Key hyperparameters (from paper Section 4.1.3):

| Parameter | Value |
|-----------|-------|
| Backbone | Qwen2.5-3B-Instruct |
| LoRA rank / alpha | 16 / 32 |
| Learning rate | 1e-4 |
| Batch size (per device) | 64 |
| Generation loss weight (λ_gen) | 0.5 |
| InfoNCE temperature (τ) | 0.02 |
| Max sequence length | 512 |

### Stage 2: GRPO Reinforcement Learning

Aligns the reasoning generation with retrieval performance using GRPO, a retrieval-aware composite reward, and dynamic positive selection.

```bash
bash scripts/run_grpo.sh
```

Key hyperparameters (from paper Section 4.1.3):

| Parameter | Value |
|-----------|-------|
| Group size (G) | 8 |
| Training epochs | 3 |
| Learning rate | 2e-7 |
| Batch size (total) | 128 |
| Format reward weight (λ_fmt) | 0.5 |
| Length penalty weight (λ_len) | 0.5, threshold L=64 |
| Retrieval reward weight (λ_ret) | 2.0 |
| Positive gain weight (α) | 2.0 |
| Margin gain weight (β) | 1.0 |
| Retrieval reward clip | [-1, 1] |
| InfoNCE loss weight (λ_cl) | 0.1 |
| KL penalty coefficient | 0.001 |

### (Optional) Distill Retriever

Distills the personalized rewriting capability into a lightweight encoder-only bi-encoder for real-time online serving.

```bash
bash scripts/run_distill_retriever.sh
```

Key hyperparameters:

| Parameter | Value |
|-----------|-------|
| Student model | Encoder-only (305M) |
| Loss pairs (equal weight 1:1:1) | ⟨q, p+⟩, ⟨q_r, p+⟩, ⟨q, q_r⟩ |
| Temperature (τ) | 0.02 |
| In-batch negatives | Enabled |

## Key Design Choices

**Frozen SFT Model as Reward & Reference:** The SFT model serves dual roles — as the KL reference for policy constraint, and as the retrieval reward model. Freezing it provides stable semantic anchoring and avoids self-referential bias (reward hacking).

**Dynamic Positive Selection:** Instead of using all rollout outputs for contrastive learning, we select the best rewrite (by retrieval reward) from the GRPO group as the positive query embedding. If no rewrite improves over the original query, we fall back to the original query embedding.

**Composite Reward:** The retrieval reward combines positive score gain (ΔS_pos) and margin gain (ΔS_margin) to encourage both absolute relevance improvement and better discrimination against negatives.

## Citation

```bibtex
@inproceedings{ttp2025,
  title={Think-to-Personalize: Unifying Reasoning and Retrieval for User-Centric Personalized Dense Retrieval},
  author={Anonymous},
  booktitle={Proceedings of CIKM},
  year={2025}
}
```

## Acknowledgments

The RL training infrastructure is built upon [VeRL](https://github.com/volcengine/verl).
