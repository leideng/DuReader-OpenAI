# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DuReader-OpenAI evaluates OpenAI-compatible chat models on a 200-sample slice of the DuReader Chinese question answering dataset. It computes F1, Precision, Recall, and ROUGE-L metrics and saves results to CSV.

## Commands

Install dependencies:
```bash
uv sync
```

Run full evaluation (default 200 samples):
```bash
uv run python main.py
```

Run debug mode (no API calls, uses gold answers as mock responses):
```bash
uv run python main.py --debug-mode --max-samples 5
```

Run with custom model and output path:
```bash
uv run python main.py --model-name my-model --save-results-path results/custom.csv
```

Show all CLI options:
```bash
uv run python main.py --help
```

## Environment Variables Required

- `OPENAI_API_KEY` - API key for OpenAI-compatible endpoint
- `OPENAI_BASE_URL` - Base URL for OpenAI-compatible endpoint

## Architecture

The evaluation flow in `main.py`:

1. **Dataset Loading**: Loads JSONL samples from `data/dureader.jsonl` via `load_jsonl_dataset()`
2. **Prompt Building**: Loads template from `prompt/template.txt` and formats with `{context}` and `{input}` (question)
3. **Async Inference**: Sends batched concurrent chat completion requests using AsyncOpenAI client with semaphore-controlled concurrency
4. **Metric Calculation**:
   - Normalizes answers (handles Chinese CJK character tokenization)
   - Computes F1/Precision/Recall and ROUGE-L (custom LCS implementation)
   - Evaluates prediction against **all** gold answers and keeps the best-scoring one
5. **Output**: Writes per-sample results to CSV with a final average metrics row

## Key Files

- `main.py`: Entry point and all evaluation logic
- `data/dureader.jsonl`: Evaluation dataset (200 samples)
- `prompt/template.txt`: Chinese QA prompt template
- `results/`: Output directory for CSV result files

## Key CLI Options

- `--model-name`: Model identifier for API requests (default: gpt-5.4)
- `--max-samples`: Limit number of evaluated samples (default: 200)
- `--request-batch-size`: Concurrent request batch size (default: 10)
- `--enable-thinking`: Enable thinking mode for models that support it
- `--debug-mode`: Skip API calls for quick testing
- `--save-results-path`: Custom output CSV path
