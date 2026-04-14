#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path


def escape_md(text):
    return text.replace("|", "\\|").replace("\n", "<br>")


def main(csv_path):
    csv_path = Path(csv_path)
    md_path = csv_path.with_suffix('.md')

    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 8:
                continue
            if row[0] == '':  # summary row
                continue
            rows.append(row)

    # Compute averages
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_rl = 0.0

    with open(md_path, 'w', encoding='utf-8') as f:
        model_name = csv_path.stem
        f.write(f"# Evaluation Results: {model_name}\n\n")
        f.write("| Index | Question | Gold Answers | Model Response | F1 | Precision | Recall | ROUGE-L |\n")
        f.write("|-------|----------|--------------|----------------|----|-----------|--------|---------|\n")

        for row in rows:
            index, question, gold_json, response, f1, precision, recall, rl = row
            try:
                gold_answers = json.loads(gold_json)
            except json.JSONDecodeError:
                gold_answers = [gold_json]

            gold_str = "<br>".join(f"- {escape_md(g)}" for g in gold_answers)
            f.write(
                f"| {index} | {escape_md(question)} | {gold_str} | {escape_md(response)} | "
                f"{float(f1):.4f} | {float(precision):.4f} | {float(recall):.4f} | {float(rl):.4f} |\n"
            )
            total_f1 += float(f1)
            total_precision += float(precision)
            total_recall += float(recall)
            total_rl += float(rl)

    # Add summary
    n = len(rows)
    avg_f1 = total_f1 / n
    avg_precision = total_precision / n
    avg_recall = total_recall / n
    avg_rl = total_rl / n

    with open(md_path, 'a', encoding='utf-8') as f:
        f.write(f"\n## Summary\n\n")
        f.write(f"- **Number of samples evaluated**: {n}\n")
        f.write(f"- **Average F1**: {avg_f1:.4f}\n")
        f.write(f"- **Average Precision**: {avg_precision:.4f}\n")
        f.write(f"- **Average Recall**: {avg_recall:.4f}\n")
        f.write(f"- **Average ROUGE-L**: {avg_rl:.4f}\n")

    print(f"Generated {md_path} from {csv_path}")
    print(f"- {n} samples processed")
    print(f"- Average F1: {avg_f1:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_md_from_csv.py <path/to/results.csv>")
        sys.exit(1)
    main(sys.argv[1])
