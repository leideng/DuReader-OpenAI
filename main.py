import argparse
import asyncio
import collections
import csv
import json
import os
import re
import string
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI, BadRequestError


DEFAULT_MODEL_NAME = "gpt-5.4"
DEFAULT_EVAL_DATASET_PATH = "data/dureader.jsonl"
DEFAULT_PROMPT_TEMPLATE_PATH = "prompt/template.txt"
DEFAULT_MAX_COMPLETION_TOKENS = 256
DEFAULT_MAX_SAMPLES = 200
DEFAULT_REQUEST_BATCH_SIZE = 10
DEFAULT_ENABLE_THINKING = False
DEFAULT_DEBUG_MODE = False
DEFAULT_SAVE_RESULTS_PATH_TEMPLATE = "results/dureader/{model_name}.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the DuReader dataset."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Model name for chat completions (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--eval-dataset-path",
        type=str,
        default=DEFAULT_EVAL_DATASET_PATH,
        help=(
            "Path to the evaluation dataset JSONL file "
            f"(default: {DEFAULT_EVAL_DATASET_PATH})."
        ),
    )
    parser.add_argument(
        "--prompt-template-path",
        type=str,
        default=DEFAULT_PROMPT_TEMPLATE_PATH,
        help=(
            "Path to the prompt template file "
            f"(default: {DEFAULT_PROMPT_TEMPLATE_PATH})."
        ),
    )
    parser.add_argument(
        "--save-results-path",
        type=str,
        default=None,
        help=(
            "Path to output CSV with evaluation results "
            f"(default: {DEFAULT_SAVE_RESULTS_PATH_TEMPLATE})."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=DEFAULT_ENABLE_THINKING,
        help=(
            "Whether to enable model thinking mode. "
            f"Use --enable-thinking to enable (default: {DEFAULT_ENABLE_THINKING})."
        ),
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help=(
            "Maximum number of completion tokens per request "
            f"(default: {DEFAULT_MAX_COMPLETION_TOKENS})."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Maximum number of samples to evaluate (default: {DEFAULT_MAX_SAMPLES}).",
    )
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=DEFAULT_REQUEST_BATCH_SIZE,
        help=(
            "Concurrent request batch size for async inference "
            f"(default: {DEFAULT_REQUEST_BATCH_SIZE})."
        ),
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        default=DEFAULT_DEBUG_MODE,
        help=(
            "Run without API requests by returning mock responses. "
            f"Use --debug-mode to enable (default: {DEFAULT_DEBUG_MODE})."
        ),
    )
    args = parser.parse_args()
    if args.save_results_path is None:
        args.save_results_path = DEFAULT_SAVE_RESULTS_PATH_TEMPLATE.format(
            model_name=args.model_name
        )
    return args


def get_required_env(name):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise ValueError(f"Required environment variable `{name}` is missing or empty.")
    return value


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value):
        return " ".join(value.split())

    def remove_punc(value):
        # Remove ASCII and common Chinese punctuation
        exclude = set(string.punctuation + "。？！，、；：「」『』《》（）【】")
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value):
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def contains_cjk(text):
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def tokenize_for_metrics(text):
    normalized = normalize_answer(text)
    if contains_cjk(normalized):
        # For Chinese, tokenize character-by-character regardless of spaces
        return [ch for ch in normalized if not ch.isspace()]
    return normalized.split()


def compute_f1(pred, gold):
    if pred is None or len(pred) == 0:
        print(f"WARNING: pred is None or empty for response: {pred}")
        return 0.0, 0.0, 0.0

    if gold is None or len(gold) == 0:
        print(f"WARNING: gold is None or empty for response: {gold}")
        return 0.0, 0.0, 0.0

    prediction_tokens = tokenize_for_metrics(pred)
    ground_truth_tokens = tokenize_for_metrics(gold)
    common = collections.Counter(prediction_tokens) & collections.Counter(
        ground_truth_tokens
    )
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def compute_rl(pred, gold):
    pred_tokens = tokenize_for_metrics(pred)
    gold_tokens = tokenize_for_metrics(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0

    dp = [[0] * (len(gold_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i, pred_token in enumerate(pred_tokens, start=1):
        for j, gold_token in enumerate(gold_tokens, start=1):
            if pred_token == gold_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def compute_best_metrics(prediction, gold_answers):
    best = None
    for gold in gold_answers:
        f1, precision, recall = compute_f1(prediction, gold)
        rl = compute_rl(prediction, gold)
        candidate = {
            "gold": gold,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "rl": rl,
        }
        if best is None or candidate["f1"] > best["f1"] or (
            candidate["f1"] == best["f1"] and candidate["rl"] > best["rl"]
        ):
            best = candidate
    return best


def load_jsonl_dataset(dataset_path):
    samples = []
    with open(dataset_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {dataset_path}: {exc}"
                ) from exc
            samples.append(sample)
    return samples


def load_prompt_template(template_path):
    with open(template_path, encoding="utf-8") as f:
        return f.read()


def build_prompt(template, sample):
    return template.format(context=sample["context"], input=sample["input"])


async def get_response_async(
    client, prompt, model_name, enable_thinking, max_completion_tokens
):
    request_kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": max_completion_tokens,
    }
    if not enable_thinking:
        request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        #request_kwargs["extra_body"] = {"reasoning_effort": "none"}

    try:
        completion = await client.chat.completions.create(**request_kwargs)
        print(completion)
        print("response:", completion.choices[0].message.content)
    except BadRequestError as exc:
        message = str(exc)
        raise("message")
    return completion.choices[0].message.content


async def get_responses_batched_async(
    client,
    prompts,
    model_name,
    enable_thinking,
    max_completion_tokens,
    max_concurrency=4,
):
    semaphore = asyncio.Semaphore(max_concurrency)
    responses = [None] * len(prompts)

    async def _run_one(idx, prompt):
        async with semaphore:
            responses[idx] = await get_response_async(
                client,
                prompt,
                model_name,
                enable_thinking,
                max_completion_tokens,
            )

    await asyncio.gather(*[_run_one(idx, prompt) for idx, prompt in enumerate(prompts)])
    return responses


async def main():
    args = parse_args()
    aclient = None
    if not args.debug_mode:
        api_key = get_required_env("OPENAI_API_KEY")
        base_url = get_required_env("OPENAI_BASE_URL")
        aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)

    try:
        eval_dataset = load_jsonl_dataset(args.eval_dataset_path)
        print(
            "Dataset loaded successfully with "
            f"{len(eval_dataset)} samples from {args.eval_dataset_path}"
        )
    except Exception as exc:
        print(f"Error loading dataset: {exc}")
        raise SystemExit(1) from exc

    try:
        prompt_template = load_prompt_template(args.prompt_template_path)
        print(f"Prompt template loaded from {args.prompt_template_path}")
    except Exception as exc:
        print(f"Error loading prompt template: {exc}")
        raise SystemExit(1) from exc

    save_results_path = Path(args.save_results_path)
    save_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV output
    with open(save_results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "question",
                "gold",
                "response",
                "f1",
                "precision",
                "recall",
                "rl",
            ]
        )

    # Also write Markdown output for better GitHub rendering
    md_path = save_results_path.with_suffix('.md')
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Results: {args.model_name}\n\n")
        f.write("| Index | Question | Gold | Response | F1 | Precision | Recall | ROUGE-L |\n")
        f.write("|-------|----------|------|----------|----|-----------|--------|---------|\n")

    f1_list = []
    rl_list = []
    precision_list = []
    recall_list = []

    selected_samples = eval_dataset[: args.max_samples]
    for batch_start in range(0, len(selected_samples), args.request_batch_size):
        batch = selected_samples[batch_start : batch_start + args.request_batch_size]

        prompts = []
        questions = []
        answers_list = []
        sample_indices = []

        for i, sample in enumerate(batch):
            idx = batch_start + i
            print(
                "=" * 20
                + f"Processing sample {idx + 1} of {len(selected_samples)}"
                + "=" * 20
            )

            question = sample["input"]
            gold_answers = sample["answers"]
            prompt = build_prompt(prompt_template, sample)

            print(f"Question: {question}")
            print(f"Gold answers: {gold_answers}")
            print(f"Context length: {len(sample['context'])} characters")
            print(f"Prompt(short): {prompt[:500]}\n......\n{prompt[-500:]}")

            prompts.append(prompt)
            questions.append(question)
            answers_list.append(gold_answers)
            sample_indices.append(idx + 1)

        if args.debug_mode:
            responses = [answers[0] for answers in answers_list]
        else:
            responses = await get_responses_batched_async(
                aclient,
                prompts,
                model_name=args.model_name,
                enable_thinking=args.enable_thinking,
                max_completion_tokens=args.max_completion_tokens,
                max_concurrency=args.request_batch_size,
            )

        for i, response in enumerate(responses):
            question = questions[i]
            gold_answers = answers_list[i]
            sample_idx = sample_indices[i]

            print(f"Response: {response}")

            best_metrics = compute_best_metrics(response, gold_answers)
            print(
                f"Best match gold: {best_metrics['gold']}\n"
                f"F1: {best_metrics['f1']}, "
                f"Precision: {best_metrics['precision']}, "
                f"Recall: {best_metrics['recall']}, "
                f"RL: {best_metrics['rl']}"
            )

            f1_list.append(best_metrics["f1"])
            precision_list.append(best_metrics["precision"])
            recall_list.append(best_metrics["recall"])
            rl_list.append(best_metrics["rl"])

            # Write to CSV
            with open(save_results_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        sample_idx,
                        question,
                        json.dumps(gold_answers, ensure_ascii=False, indent=2),
                        response,
                        best_metrics["f1"],
                        best_metrics["precision"],
                        best_metrics["recall"],
                        best_metrics["rl"],
                    ]
                )

            # Write to Markdown
            def escape_md(text):
                return text.replace("|", "\\|").replace("\n", "<br>")

            md_path = save_results_path.with_suffix('.md')
            with open(md_path, "a", encoding="utf-8") as f:
                gold_str = "; ".join(f"- {escape_md(g)}" for g in gold_answers)
                f.write(
                    f"| {sample_idx} | {escape_md(question)} | {gold_str} | {escape_md(response)} | "
                    f"{best_metrics['f1']:.4f} | {best_metrics['precision']:.4f} | "
                    f"{best_metrics['recall']:.4f} | {best_metrics['rl']:.4f} |\n"
                )

    print("---------------Result Summary---------------------")
    avg_f1 = np.mean(f1_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_rl = np.mean(rl_list)
    with open(save_results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "", "", "", avg_f1, avg_precision, avg_recall, avg_rl])

    # Add summary to Markdown
    md_path = save_results_path.with_suffix('.md')
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(f"\n## Summary\n\n")
        f.write(f"- **Number of samples evaluated**: {len(f1_list)}\n")
        f.write(f"- **Average F1**: {avg_f1:.4f}\n")
        f.write(f"- **Average Precision**: {avg_precision:.4f}\n")
        f.write(f"- **Average Recall**: {avg_recall:.4f}\n")
        f.write(f"- **Average ROUGE-L**: {avg_rl:.4f}\n")

    print(f"F1: {avg_f1}")
    print(f"Precision: {avg_precision}")
    print(f"Recall: {avg_recall}")
    print(f"RL: {avg_rl}")
    print("----------------------------------------------------")


if __name__ == "__main__":
    asyncio.run(main())
