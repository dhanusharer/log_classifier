"""
CLI entrypoint — works with the exact synthetic_logs.csv schema.

Two modes:
  --mode train    : reads a labelled CSV (with target_label), trains the model,
                    runs eval and prints an accuracy report
  --mode classify : reads any CSV with a log_message column (no target_label needed),
                    classifies every row, writes enriched output CSV

Usage examples:
  # Train on the canonical dataset
  python -m src.run_batch --mode train --input data/synthetic_logs.csv

  # Classify a new unlabelled file from the client
  python -m src.run_batch --mode classify --input new_logs.csv --output output/results.csv

  # Classify AND evaluate (when target_label is present)
  python -m src.run_batch --mode classify --input data/eval_data.csv --evaluate
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
audit_handler = logging.FileHandler("output/audit.log")
audit_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logging.getLogger("audit").addHandler(audit_handler)


def mode_train(args) -> None:
    """Train the model on a labelled CSV and report accuracy on the eval split."""
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from src.config import load_config
    from src.data_loader import load_labelled_csv
    from src.pipeline import build_pipeline

    cfg = load_config()
    input_path = Path(args.input)

    print(f"\n📂  Loading labelled data from: {input_path}")
    texts, labels, sources = load_labelled_csv(input_path)
    print(f"    {len(texts)} rows loaded")

    # Split into train / eval
    from collections import Counter
    label_counts = Counter(labels)
    # Only stratify on labels with ≥ 2 samples
    stratify_labels = [l if label_counts[l] >= 2 else "__other__" for l in labels]

    train_idx, eval_idx = train_test_split(
        range(len(texts)), test_size=0.2,
        stratify=stratify_labels, random_state=42,
    )
    train_texts  = [texts[i]   for i in train_idx]
    train_labels = [labels[i]  for i in train_idx]
    eval_texts   = [texts[i]   for i in eval_idx]
    eval_labels  = [labels[i]  for i in eval_idx]
    eval_sources = [sources[i] for i in eval_idx]

    # Save training split
    import csv
    with open(cfg.training_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["log_message", "target_label", "source"])
        w.writeheader()
        for t, l, s in zip(train_texts, train_labels, [sources[i] for i in train_idx]):
            w.writerow({"log_message": t, "target_label": l, "source": s})

    print(f"\n🔧  Training on {len(train_texts)} rows ...")
    pipeline = build_pipeline(train_texts, train_labels, config=cfg)

    # Evaluate
    print(f"\n📊  Evaluating on {len(eval_texts)} rows ...")
    results = pipeline.classify_batch(eval_texts, eval_sources)

    correct = sum(r.predicted_label == true for r, true in zip(results, eval_labels))
    acc = correct / len(results)

    print(f"\n{'═'*60}")
    print(f"  Overall accuracy : {acc:.2%}  ({correct}/{len(results)})")
    print(f"{'═'*60}")
    print(f"\n  {'Label':<24} {'Acc':>6}  {'n':>4}  {'Method'}")
    print("  " + "─" * 52)

    from collections import defaultdict
    per_label: dict[str, list] = defaultdict(list)
    per_label_method: dict[str, list] = defaultdict(list)
    for r, true in zip(results, eval_labels):
        per_label[true].append(r.predicted_label == true)
        per_label_method[true].append(r.method_used.value)

    for label in sorted(per_label):
        hits = per_label[label]
        method = Counter(per_label_method[label]).most_common(1)[0][0]
        print(f"  {label:<24} {sum(hits)/len(hits):>5.1%}  {len(hits):>4}  {method}")

    print(f"\n  Training CSV saved → {cfg.training_csv}")
    print(f"  Model ready for classification.\n")


def mode_classify(args) -> None:
    """Classify every row in the input CSV and write an enriched output CSV."""
    import pandas as pd
    from src.config import load_config
    from src.data_loader import load_inference_csv, load_labelled_csv, save_classified_csv
    from src.pipeline import build_pipeline

    cfg = load_config()
    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else cfg.output_csv

    # Load the trained model from the saved training CSV
    if not cfg.training_csv.exists():
        print("❌  No training data found. Run --mode train first.")
        sys.exit(1)

    texts, labels, _ = load_labelled_csv(cfg.training_csv)
    pipeline = build_pipeline(texts, labels, config=cfg)

    print(f"\n📂  Classifying: {input_path}")
    df = load_inference_csv(input_path)
    print(f"    {len(df)} rows to classify ...")

    results = pipeline.classify_batch(
        df["log_message"].tolist(),
        df["source"].fillna("").tolist(),
    )

    out = save_classified_csv(df, results, output_path)

    # Summary
    label_counts  = Counter(r.predicted_label  for r in results)
    method_counts = Counter(r.method_used.value for r in results)
    classified = sum(1 for r in results if r.predicted_label != "UNCLASSIFIED")
    flagged    = pipeline.flagged_for_labelling(results)

    print(f"\n{'═'*60}")
    print(f"  Total rows      : {len(results)}")
    print(f"  Classified      : {classified}")
    print(f"  Unclassified    : {len(results) - classified}")
    print(f"  Flagged review  : {len(flagged)}")
    print(f"{'═'*60}")
    print(f"\n  Label distribution:")
    for label, count in label_counts.most_common():
        bar = "█" * int(count / max(label_counts.values()) * 30)
        print(f"  {label:<24} {count:>5}  {bar}")

    print(f"\n  Method breakdown:")
    for method, count in method_counts.most_common():
        print(f"  {method:<15} {count:>5}")

    # Optional accuracy evaluation if target_label present
    if args.evaluate and "target_label" in pd.read_csv(input_path, nrows=0).columns:
        truth_df = pd.read_csv(input_path)
        truth = truth_df["target_label"].tolist()
        correct = sum(r.predicted_label == t for r, t in zip(results, truth))
        print(f"\n  Accuracy (vs target_label): {correct/len(results):.2%}  ({correct}/{len(results)})")

    print(f"\n  ✅  Output saved → {out}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log classifier CLI — synthetic_logs.csv schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train", "classify"], required=True,
                        help="train: fit model on labelled CSV | classify: run inference on CSV")
    parser.add_argument("--input",    required=True, help="Path to input CSV")
    parser.add_argument("--output",   default=None,  help="Output CSV path (classify mode)")
    parser.add_argument("--evaluate", action="store_true",
                        help="If target_label is present, print accuracy (classify mode)")
    args = parser.parse_args()

    if args.mode == "train":
        mode_train(args)
    else:
        mode_classify(args)


if __name__ == "__main__":
    main()
