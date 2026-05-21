# TTP: Think-to-Personalize

Official implementation of **"Think-to-Personalize: Unifying Reasoning and Retrieval for User-Centric Personalized Dense Retrieval"**.

## Repository Structure

```
ttp/
├── process/
│   ├── kuaisearch.py           # KuaiSearch data processing
│   └── personalwab.py          # PersonalWAB data processing
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

## Data Preparation

- `process/kuaisearch.py` — Builds train/test splits from KuaiSearch
- `process/personalwab.py` — Builds train/test splits from PersonalWAB

The `query_gen` field (generation supervision target) is synthesized by prompting an LLM to produce intent-enhanced query rewrites given the user history and original query.

## Data Format

### Stage 1 (SFT) & Stage 2 (GRPO RL)

```jsonl
{
  "id": "sample_001",
  "query_gen": "rewritten personalized query",
  "hist_list": [],
  "query": ["Given a user history sequence and an E-commerce query, analyze the user's intent and rewrite a personalized query to retrieve relevant products. Please output in the format: <think>Rewrite Query</think><emb>.", "original query text"],
  "pos": [],
  "neg": []
}
```

- `id`: sample identifier
- `query_gen`: generation target, synthesized by LLM
- `hist_list`: user history sequence
- `query`: `[instruction, original_query]`
- `pos`: positive passage(s)
- `neg`: hard negative passage(s)

### Distill Retriever

```jsonl
{
  "query": "original query text",
  "query_gen": "TTP-generated intent-enhanced query",
  "pos": [],
  "neg": []
}
```

- `query`: original query text
- `query_gen`: TTP-generated intent-enhanced query
- `pos`: positive passage(s)
- `neg`: hard negative passage(s)

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
