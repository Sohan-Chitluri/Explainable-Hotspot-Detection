#!/usr/bin/env python3
"""Workstream B1: Decision Threshold Calibration and Analysis for Via-3.

Integrity Safeguards:
- Decision threshold tau* is calibrated STRICTLY on validation predictions.
- Reuses saved validation probabilities/predictions; does not retrain models.
- Generates precision-recall trade-off curves, confusion matrices, and tables.
- Evaluates test predictions only at the frozen calibrated threshold.
- Supports --dry-run mode using synthetic or fixture validation probabilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from src.experiment_contract import (
    compute_contract_metrics,
    select_threshold_from_val,
)


def plot_threshold_curves(sweep_table: list[dict], out_path: Path, title: str, selected_tau: float):
    """Plot Precision, Recall, and F1 across decision thresholds."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    taus = [r["threshold"] for r in sweep_table]
    precisions = [r["precision"] for r in sweep_table]
    recalls = [r["recall"] for r in sweep_table]
    f1s = [r["f1_score"] for r in sweep_table]

    plt.figure(figsize=(7, 4.5))
    plt.plot(taus, recalls, "g-", label="HS Recall (Sensitivity)", linewidth=2)
    plt.plot(taus, precisions, "b--", label="HS Precision", linewidth=2)
    plt.plot(taus, f1s, "r:", label="HS F1 Score", linewidth=2.5)
    plt.axvline(selected_tau, color="black", linestyle="-.", label=f"Selected tau*={selected_tau:.2f}")

    plt.title(title, fontsize=11, fontweight="bold")
    plt.xlabel("Decision Threshold (tau)")
    plt.ylabel("Metric Score")
    plt.ylim(-0.02, 1.02)
    plt.xlim(0.0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr_curve(sweep_table: list[dict], out_path: Path, title: str):
    """Plot Precision vs Recall curve from validation sweep."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recalls = [r["recall"] for r in sweep_table]
    precisions = [r["precision"] for r in sweep_table]

    plt.figure(figsize=(5.5, 4.5))
    plt.plot(recalls, precisions, "b-o", markersize=3, label="Val PR Curve")
    plt.xlabel("Hotspot Recall (Sensitivity)")
    plt.ylabel("Hotspot Precision")
    plt.ylim(-0.02, 1.05)
    plt.xlim(-0.02, 1.02)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.title(title, fontsize=11, fontweight="bold")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: dict, out_path: Path, title: str):
    """Plot confusion matrix with explicit semantic class labels."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array([
        [cm["tn"], cm["fp"]],
        [cm["fn"], cm["tp"]],
    ], dtype=int)

    plt.figure(figsize=(4.8, 4))
    sn.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred NHS (0)", "Pred HS (1)"],
        yticklabels=["Actual NHS (0)", "Actual HS (1)"],
        cbar=False,
    )
    plt.title(title, fontsize=10, fontweight="bold")
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_threshold_analysis(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    output_dir: Path,
    objective: str = "max_f1",
    min_precision: float = 0.50,
    target_recall: float = 0.70,
    test_probs: np.ndarray | None = None,
    test_labels: np.ndarray | None = None,
    experiment_name: str = "via3_baseline",
) -> dict:
    """Run validation threshold sweep and optional frozen test evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Threshold Analysis: {experiment_name}]")
    print(f"Validation cohort: {len(val_labels)} samples ({int(np.sum(val_labels))} HS)")
    print(f"Operating Objective: {objective} (min_precision={min_precision}, target_recall={target_recall})")

    # Strictly calibrate on validation predictions
    calibration = select_threshold_from_val(
        val_labels=val_labels,
        val_probs=val_probs,
        objective=objective,
        min_precision=min_precision,
        target_recall=target_recall,
    )
    tau_star = calibration["selected_threshold"]
    val_m_star = calibration["metrics_at_selected_threshold"]
    val_m_default = compute_contract_metrics(val_labels, val_probs, threshold=0.50)

    print(f"Calibrated Threshold tau* = {tau_star:.4f}")
    print(f"  Val @ 0.50: Recall={val_m_default['recall_sensitivity']:.4f}, Precision={val_m_default['precision']:.4f}, F1={val_m_default['f1_score']:.4f}")
    print(f"  Val @ tau*: Recall={val_m_star['recall_sensitivity']:.4f}, Precision={val_m_star['precision']:.4f}, F1={val_m_star['f1_score']:.4f}")

    # Generate plots and tables
    sweep_csv = output_dir / f"{experiment_name}_threshold_sweep.csv"
    with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(calibration["sweep_table"][0].keys()))
        writer.writeheader()
        writer.writerows(calibration["sweep_table"])

    plot_threshold_curves(
        calibration["sweep_table"],
        output_dir / f"{experiment_name}_threshold_tradeoff.png",
        title=f"{experiment_name}: Threshold Trade-Off Curves",
        selected_tau=tau_star,
    )

    plot_pr_curve(
        calibration["sweep_table"],
        output_dir / f"{experiment_name}_val_pr_curve.png",
        title=f"{experiment_name}: Validation Precision-Recall Curve",
    )

    plot_confusion_matrix(
        val_m_star["confusion_matrix"],
        output_dir / f"{experiment_name}_val_cm_calibrated.png",
        title=f"Validation CM @ tau*={tau_star:.2f}",
    )

    # Held-out test evaluation at frozen tau* (if test data provided)
    test_results = None
    if test_probs is not None and test_labels is not None:
        print("\nEvaluating held-out test predictions at frozen threshold...")
        test_m_default = compute_contract_metrics(test_labels, test_probs, threshold=0.50)
        test_m_star = compute_contract_metrics(test_labels, test_probs, threshold=tau_star)
        test_results = {
            "default_tau_0_50": test_m_default,
            "calibrated_tau_star": test_m_star,
        }
        plot_confusion_matrix(
            test_m_star["confusion_matrix"],
            output_dir / f"{experiment_name}_clean_test_cm_calibrated.png",
            title=f"Clean Test CM @ tau*={tau_star:.2f}",
        )
        print(f"  Test @ 0.50: Recall={test_m_default['recall_sensitivity']:.4f}, Precision={test_m_default['precision']:.4f}, F1={test_m_default['f1_score']:.4f}")
        print(f"  Test @ tau*: Recall={test_m_star['recall_sensitivity']:.4f}, Precision={test_m_star['precision']:.4f}, F1={test_m_star['f1_score']:.4f}")

    results = {
        "experiment_name": experiment_name,
        "calibrated_threshold": tau_star,
        "objective": objective,
        "validation_metrics_default": val_m_default,
        "validation_metrics_calibrated": val_m_star,
        "test_metrics": test_results,
        "sweep_summary_path": str(sweep_csv),
    }

    out_json = output_dir / f"{experiment_name}_threshold_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Workstream B1 Threshold Analysis")
    parser.add_argument("--val-predictions", type=Path, default=None, help="Path to val predictions CSV")
    parser.add_argument("--test-predictions", type=Path, default=None, help="Path to clean test predictions CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("results/via3_experiments/threshold_analysis"))
    parser.add_argument("--objective", type=str, default="max_f1", choices=["max_f1", "max_recall_constrained", "target_recall", "max_balanced_accuracy"])
    parser.add_argument("--min-precision", type=float, default=0.50)
    parser.add_argument("--target-recall", type=float, default=0.70)
    parser.add_argument("--dry-run", action="store_true", help="Run with synthetic validation data fixture")
    args = parser.parse_args()

    if args.dry_run or (args.val_predictions is None and not args.val_predictions):
        print("[DRY-RUN] Generating synthetic validation prediction fixture for verification...")
        rng = np.random.default_rng(42)
        # 2846 validation samples, 87 HS (approx 3% positive)
        n_val = 2846
        n_hs = 87
        val_labels = np.zeros(n_val, dtype=int)
        val_labels[:n_hs] = 1
        rng.shuffle(val_labels)
        val_probs = np.where(val_labels == 1, rng.beta(2, 5, n_val), rng.beta(1, 25, n_val))

        run_threshold_analysis(
            val_probs=val_probs,
            val_labels=val_labels,
            output_dir=args.output_dir,
            objective=args.objective,
            min_precision=args.min_precision,
            target_recall=args.target_recall,
            experiment_name="synthetic_baseline_fixture",
        )
        print("Dry run completed successfully.")
        return

    # Load from CSV if provided
    val_labels, val_probs = [], []
    with open(args.val_predictions, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_labels.append(int(row["ground_truth"]))
            val_probs.append(float(row["hs_probability"]))

    test_labels, test_probs = None, None
    if args.test_predictions and args.test_predictions.exists():
        test_labels, test_probs = [], []
        with open(args.test_predictions, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_labels.append(int(row["ground_truth"]))
                test_probs.append(float(row["hs_probability"]))

    run_threshold_analysis(
        val_probs=np.array(val_probs),
        val_labels=np.array(val_labels),
        output_dir=args.output_dir,
        objective=args.objective,
        min_precision=args.min_precision,
        target_recall=args.target_recall,
        test_probs=np.array(test_probs) if test_probs else None,
        test_labels=np.array(test_labels) if test_labels else None,
        experiment_name="via3_baseline",
    )


if __name__ == "__main__":
    main()
