# TTP: Think-to-Personalize

Official implementation of **"Think-to-Personalize: Unifying Reasoning and Retrieval for User-Centric Personalized Dense Retrieval"**.

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

**`distill_retriever/`** — Distills the TTP model into a lightweight bi-encoder for online deployment.

## Environment Setup

Note that we validate the implementation of stage 1 and stage 2 on different environments.

```
requirements/sft.txt
requirements/rl.txt
```

## Data Construction

Training data is constructed via a teacher model (Qwen3-32B) following a filter → rewrite → select pipeline:

1. **History filtering**: For each user query, select the top-10 most relevant items from the user's purchase history using bge-reranker-v2-m3.
2. **Candidate generation**: Prompt Qwen3-32B with the user history and query to generate K=5 candidate intent-enhanced rewrites.
3. **Quality selection**: Score each rewrite by retrieval relevance gain (cosine similarity improvement over the original query against the ground-truth positive item). Keep the best rewrite per query.
4. **Data split**: Samples with positive gain are used for both SFT and RL. The top-30% highest-gain samples are additionally used as the RL training set.

Each training sample contains: `query`, `pos` (positive item), `neg` (hard negatives), and `query_gen` (the teacher-generated rewrite as the SFT target).

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
