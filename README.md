# DuReader-OpenAI

Evaluate OpenAI-compatible chat models on the 200-sample DuReader slice in `data/dureader.jsonl` and save per-sample QA metrics to CSV.

## Overview

`main.py` follows the same evaluation flow as `../2WikiMultihopQA-OpenAI/`:

- load the evaluation dataset
- build one prompt per sample
- send async batched chat-completion requests
- compute `F1`, `Precision`, `Recall`, and `ROUGE-L`
- save per-sample results plus a final average row to CSV

The DuReader adaptation changes only:

- dataset loading: JSONL from `data/dureader.jsonl`
- prompt building: use `prompt/template.txt`
- ground truth handling: evaluate each prediction against all answers in `answers` and keep the best score

## Requirements

- Python `>= 3.11`
- `uv`
- An OpenAI-compatible API endpoint
- Environment variables:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`

Install dependencies with:

```bash
uv sync
```

## Quick Start

Run the default evaluation:

```bash
uv run python main.py
```

Run a local sanity check without API calls:

```bash
uv run python main.py --debug-mode --max-samples 5
```

Write results to a custom CSV:

```bash
uv run python main.py \
  --model-name gpt-5.4 \
  --save-results-path results/dureader/gpt-5.4.csv
```

## Prompt Template

The prompt is loaded from `prompt/template.txt` and formatted with:

- `{context}`: the sample context
- `{input}`: the sample question

Current template:

```text
请基于给定的文章回答下述问题。
文章:{context}
请基于上述文章回答下面的问题。
问题:{input}
回答:
```

## CLI Options

Key flags:

```text
--model-name
--eval-dataset-path
--prompt-template-path
--save-results-path
--enable-thinking
--max-completion-tokens
--max-samples
--request-batch-size
--debug-mode
```

Use `uv run python main.py --help` for the full argparse output.

## Output Format

Each CSV contains one row per evaluated sample:

`index, question, gold, response, f1, precision, recall, rl`

The `gold` column stores the full answer list as JSON. Metrics are computed against all candidate gold answers and use the best-matching one for each sample.

## Repository Layout

- `main.py`: evaluation entry point
- `data/dureader.jsonl`: DuReader evaluation dataset
- `prompt/template.txt`: prompt template
- `results/`: CSV outputs
