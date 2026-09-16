#!/usr/bin/env python3
"""Merge per-benchmark baseline metrics into results/baseline/metrics.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    results = root / "results" / "baseline"
    per = {}
    for b in range(1, 6):
        path = results / f"iccad{b}" / "metrics.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                per[f"iccad{b}"] = json.load(f)

    if not per:
        print("No per-benchmark metrics found.", file=sys.stderr)
        sys.exit(1)

    bal = {k: v["balanced_accuracy"] for k, v in per.items()}
    avg = sum(bal.values()) / len(bal)
    summary = {
        "benchmarks": list(per.keys()),
        "balanced_accuracy_per_benchmark": bal,
        "average_balanced_accuracy": avg,
        "per_benchmark": per,
        "positive_class_for_reported_metrics": "HS (folder *_hs, label 0)",
        "note": (
            "Final-epoch test-set metrics with HS as positive class. "
            "Notebook bar chart used best-epoch Keras history balanced accuracy."
        ),
    }
    out = results / "metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    labels = [k.replace("iccad", "") for k in per.keys()]
    values = [bal[k] for k in per.keys()]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color="maroon", width=0.4)
    plt.xlabel("ICCAD-12 benchmarks", fontweight="bold", fontsize=12)
    plt.ylabel("Validation Balanced Accuracy", fontweight="bold", fontsize=12)
    plt.title("Custom CNN baseline balanced accuracy")
    plt.tight_layout()
    plt.savefig(results / "balanced_accuracy_bar.png", dpi=150)
    plt.close()

    print(f"Wrote {out}")
    print(f"Average balanced accuracy ({len(per)} benchmarks) = {avg:.4f}")
    for k, v in bal.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
