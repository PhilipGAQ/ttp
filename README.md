# TTP: Think-to-Personalize

Official implementation of **"Think-to-Personalize: Unifying Reasoning and Retrieval for User-Centric Personalized Dense Retrieval"** (CIKM 2025).

## Repository Structure

```
ttp/
├── requirements/
│   ├── sft.txt                 # Stage 1 & Distill Retriever dependencies
│   └── rl.txt                  # Stage 2 (GRPO RL) dependencies
├── scripts/
│   ├── run_sft.sh              # Stage 1: Cold-Start SFT
│   ├── run_grpo.sh             # Stage 2: GRPO RL
│   ├── run_distill_retriever.sh# (Optional) Distill Retriever
│   └── accelerate_config.yaml  # 8-GPU accelerate config
├── ttp_sft/                    # Stage 1 training code
├── ttp_rl/                     # Stage 2 RL training code (VeRL-based)
│   ├── recipe/ttp_grpo/        # TTP-specific GRPO recipe
│   └── verl/                   # VeRL framework
└── distill_retriever/          # Distill Retriever training code
```

## Components

**`ttp_sft/`** — Joint generation + contrastive SFT. The model learns to generate `<think>...</think><emb>`, where the `<emb>` token's hidden state serves as the retrieval embedding. Loss: NTP + InfoNCE with in-batch negatives.

**`ttp_rl/recipe/ttp_grpo/`** — GRPO RL alignment. Composite reward (format + length + retrieval) drives the policy to generate better intent-enhanced queries. Includes Dynamic Positive Selection for InfoNCE and a frozen SFT model as both KL reference and retrieval reward model.

**`distill_retriever/`** — Distills the 3B TTP model into a lightweight bi-encoder (305M) for online deployment.

## Environment Setup

The two training stages use **different environments**:

| | Stage 1 (SFT & Distill) | Stage 2 (GRPO RL) |
|---|---|---|
| PyTorch | 2.6.0 | 2.8.0 |
| Transformers | 4.51.2 | 4.57.1 |
| vLLM | 0.8.5 | 0.10.2 |

```bash
# Stage 1 environment
conda create -n ttp_sft python=3.10 && conda activate ttp_sft
pip install -r requirements/sft.txt

# Stage 2 environment
conda create -n ttp_rl python=3.10 && conda activate ttp_rl
pip install -r requirements/rl.txt
cd ttp_rl && pip install -e .
```

Hardware: 8× NVIDIA A100 80GB GPUs.

## Training

### Stage 1: Cold-Start SFT

Edit paths in `scripts/run_sft.sh`, then:

```bash
bash scripts/run_sft.sh
```

### Stage 2: GRPO Reinforcement Learning

Edit paths in `scripts/run_grpo.sh` (set `MODEL_PATH` to Stage 1 output), then:

```bash
bash scripts/run_grpo.sh
```

### (Optional) Distill Retriever

```bash
bash scripts/run_distill_retriever.sh
```

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
