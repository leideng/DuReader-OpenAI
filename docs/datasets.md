# Dataset

The evaluation uses a fixed 200-sample slice of the DuReader dataset stored at `data/dureader.jsonl`.

## Format

Each line is a JSON object with:

```json
{
  "context": "Full passage context from Wikipedia/search results...",
  "input": "The question to answer",
  "answers": [
    "First acceptable answer",
    "Second acceptable answer",
    "..."
  ]
}
```

- **context**: The source text the model must base its answer on
- **input**: The question
- **answers**: One or more gold-standard acceptable answers (multiple answers are allowed because questions can have multiple valid phrasings)

## Sample Entry

Here's a concrete example:

```json
{
  "input": "热诚传说结局",
  "answers": ["《热诚传说》动画的结局是百合。"],
  "context": "文章:文章1\n标题：对结局好失望_热诚传说x吧_百度贴吧\n..."
}
```

## Statistics

- Total samples: **200**
- Average context length: ~several thousand characters per sample
- Multiple answers per question: Yes, some questions have multiple valid phrasings
- Language: All questions and context are **Chinese**

## Evaluation Strategy

When multiple gold answers are available, the evaluation:
1. Computes metrics against **every** gold answer independently
2. Keeps the **best** (highest F1) score for the prediction

This accounts for the fact that any of the listed answers can be correct.
