"""
Enhanced evaluation script for GraphRAG-Biomedical.

Metrics:
  - Exact Match (EM)
  - Token-level F1
  - HITS@k  (k=1,3,5)
  - Average reasoning steps (parsed from log files)
  - Per-question breakdown to a CSV

Usage:
  python eval_enhanced.py \
      --result_file path/to/results.jsonl \
      --logs_dir   path/to/logs \
      --dataset    biomedical \
      --output     path/to/eval_report.csv
"""

import json
import re
import os
import csv
import argparse
import string
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--result_file", type=str, required=True)
parser.add_argument("--logs_dir",    type=str, default=None,
                    help="Directory containing per-question .txt logs (for step counting)")
parser.add_argument("--dataset",     type=str, default="biomedical")
parser.add_argument("--output",      type=str, default=None,
                    help="CSV file to write per-question results (optional)")
args = parser.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the|usd)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match(pred, gt) -> bool:
    return normalize(str(pred)) == normalize(str(gt))


def token_f1(pred, gt) -> float:
    pred_toks = normalize(str(pred)).split()
    gt_toks   = normalize(str(gt)).split()
    common    = Counter(pred_toks) & Counter(gt_toks)
    n_same    = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks) if pred_toks else 0.0
    recall    = n_same / len(gt_toks)   if gt_toks   else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hits_at_k(pred, gt, k: int) -> bool:
    """
    If pred is a list, check whether gt appears within the first k items.
    If pred is a scalar, treat as k=1 exact match regardless of k.
    """
    if isinstance(pred, list):
        candidates = [normalize(str(p)) for p in pred[:k]]
        return normalize(str(gt)) in candidates
    return exact_match(pred, gt)


def parse_step_count(log_path: str):
    """Count Thought steps in a log file as a proxy for reasoning depth."""
    if not log_path or not os.path.exists(log_path):
        return None
    with open(log_path, encoding="utf-8") as f:
        content = f.read()
    return len(re.findall(r"Thought \d+:", content))


def parse_neighborsearch_uses(log_path: str):
    """Count how many times NeighborSearch was invoked in a log."""
    if not log_path or not os.path.exists(log_path):
        return None
    with open(log_path, encoding="utf-8") as f:
        content = f.read()
    return len(re.findall(r"NeighborSearch\[", content))


# ── load results ─────────────────────────────────────────────────────────────

records = []
with open(args.result_file, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

if not records:
    raise ValueError(f"No records found in {args.result_file}")

# ── per-question metrics ──────────────────────────────────────────────────────

rows = []
em_scores   = []
f1_scores   = []
h1_scores, h3_scores, h5_scores = [], [], []
step_counts = []
ns_counts   = []

for i, rec in enumerate(records):
    pred = rec.get("model_answer", "")
    gt   = rec.get("gt_answer",    "")
    q    = rec.get("question",     "")
    qid  = rec.get("qid", f"q{i:04d}")

    if pred is None:
        pred = ""

    em = exact_match(pred, gt)
    f1 = token_f1(pred, gt)
    h1 = hits_at_k(pred, gt, 1)
    h3 = hits_at_k(pred, gt, 3)
    h5 = hits_at_k(pred, gt, 5)

    log_path = os.path.join(args.logs_dir, f"{qid}.txt") if args.logs_dir else None
    steps    = parse_step_count(log_path)
    ns_uses  = parse_neighborsearch_uses(log_path)

    em_scores.append(int(em))
    f1_scores.append(f1)
    h1_scores.append(int(h1))
    h3_scores.append(int(h3))
    h5_scores.append(int(h5))
    if steps is not None:
        step_counts.append(steps)
    if ns_uses is not None:
        ns_counts.append(ns_uses)

    rows.append({
        "qid":             qid,
        "question":        q,
        "gt_answer":       gt,
        "model_answer":    pred,
        "em":              int(em),
        "f1":              round(f1, 4),
        "hits@1":          int(h1),
        "hits@3":          int(h3),
        "hits@5":          int(h5),
        "steps":           steps,
        "neighborsearch":  ns_uses,
    })

# ── aggregate ─────────────────────────────────────────────────────────────────

n = len(records)
print(f"\n{'='*60}")
print(f"Dataset : {args.dataset}")
print(f"Samples : {n}")
print(f"{'='*60}")
print(f"Exact Match   : {sum(em_scores)/n*100:.2f}%")
print(f"Token F1      : {sum(f1_scores)/n*100:.2f}%")
print(f"HITS@1        : {sum(h1_scores)/n*100:.2f}%")
print(f"HITS@3        : {sum(h3_scores)/n*100:.2f}%")
print(f"HITS@5        : {sum(h5_scores)/n*100:.2f}%")

if step_counts:
    avg_steps = sum(step_counts) / len(step_counts)
    print(f"Avg Steps     : {avg_steps:.2f}  (over {len(step_counts)} logged questions)")
    correct_steps   = [s for s, e in zip(step_counts, em_scores) if e == 1]
    incorrect_steps = [s for s, e in zip(step_counts, em_scores) if e == 0]
    if correct_steps:
        print(f"  Correct   avg steps : {sum(correct_steps)/len(correct_steps):.2f}")
    if incorrect_steps:
        print(f"  Incorrect avg steps : {sum(incorrect_steps)/len(incorrect_steps):.2f}")

if ns_counts:
    total_ns = sum(ns_counts)
    print(f"NeighborSearch calls : {total_ns} total  "
          f"({total_ns/len(ns_counts):.2f} avg per question)")

print(f"{'='*60}\n")

# ── optional CSV output ───────────────────────────────────────────────────────

if args.output:
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fieldnames = ["qid", "question", "gt_answer", "model_answer",
                  "em", "f1", "hits@1", "hits@3", "hits@5", "steps", "neighborsearch"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-question results saved to: {args.output}")
