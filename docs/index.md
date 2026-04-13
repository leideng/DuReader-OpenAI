# DuReader-OpenAI

This project evaluates OpenAI-compatible chat models on the 200-example DuReader benchmark slice in `data/dureader.jsonl`.

`main.py` mirrors the evaluation flow from `../2WikiMultihopQA-OpenAI/`:

- load the dataset
- format prompts from `prompt/template.txt`
- call the OpenAI-compatible chat API in async batches
- compute `F1`, `Precision`, `Recall`, and `ROUGE-L`
- save detailed CSV results under `results/dureader/`

## Run

```bash
uv sync
uv run python main.py
```

For a no-API sanity check:

```bash
uv run python main.py --debug-mode --max-samples 5
```

## Notes

- `data/dureader.jsonl` is used directly as ground truth with no fixing or filtering.
- Each sample may contain multiple acceptable answers. Evaluation scores a model response against all of them and keeps the best score.
- The prompt template is externalized in `prompt/template.txt` so you can adjust wording without changing the evaluator.
