#!/usr/bin/env python3
"""Workstream H — class-weighted recipe replication audit (CPU-only).

Two strictly separated capabilities:

1. ``audit(root)`` / default CLI — recompute historical classification metrics and
   ``dense_penultimate`` representation statistics for the ICCAD5 imbalance-ablation
   conditions (``baseline_noweight``, ``class_weighted``, ``balanced_sampling``) and the
   five existing seed-replication runs, purely from stored per-sample
   activation/prediction CSVs and stored training ``metrics.json`` files.
   No TensorFlow import, no checkpoint loading, no image I/O, no training, no neural
   inference. Existing artifacts are only ever read.

2. ``--replicate`` — the guarded entry point for the seeds 101/202/303 class-weighted
   retraining protocol (docs/IMPROVEMENT_WEIGHTED_REPLICATION.md §8). It performs
   pre-flight diagnostics and then unconditionally refuses to run. Even the workstream gate
   (``WORKSTREAM_H_GPU_APPROVED=1``) and GPU visibility cannot enable training here. As of
   2026-09-17 the GPU gate is closed (nvidia-smi exit 6; TensorFlow sees ``GPUS []``),
   so this branch always exits non-zero with instructions. No training code is embedded
   here; the training loop must be implemented per the documented protocol and only
   after the lead serializes and approves the GPU job.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REQUIRED_SEEDS = (101, 202, 303)
POSITIVE_LABEL = "HS"  # label 0 in the stored CSVs; HS is the positive class throughout
DOC = "Workstream H: class-weighted recipe audit + guarded seeds 101/202/303 replication runner (CPU-only audit; training blocked)."


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_condition_rows(source: Path, run: str, run_col: str):
    """Load one condition's stored per-sample rows (pure CSV read, no ML execution)."""
    with source.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r[run_col] == run]
    units = np.array([[float(r[f"unit_{i}"]) for i in range(16)] for r in rows], dtype=np.float64)
    if len(rows) != 19368 or len({r["sample_id"] for r in rows}) != 19368:
        raise ValueError("Expected 19368 unique samples per condition")
    if not np.isfinite(units).all() or (units < 0).any():
        raise ValueError("Invalid stored ReLU activations")
    true_hs = np.array([r["true_label"] == POSITIVE_LABEL for r in rows])
    if int(true_hs.sum()) != 41:
        raise ValueError("Expected 41 HS samples")
    pred_positive = np.array([float(r["p_nhs"]) for r in rows]) < 0.5  # p_nhs = P(class 1 = NHS)
    zero = (units == 0.0).all(axis=1)
    stored_pred = np.array([int(r["pred_label"]) for r in rows]) == 0
    stored_zero = np.array([r["is_zero_vector"] == "True" for r in rows])
    return units, true_hs, pred_positive, zero, stored_pred, stored_zero


def _confusion(true_hs: np.ndarray, pred_positive: np.ndarray) -> dict:
    tp = int(np.sum(true_hs & pred_positive))
    fn = int(np.sum(true_hs & ~pred_positive))
    fp = int(np.sum(~true_hs & pred_positive))
    tn = int(np.sum(~true_hs & ~pred_positive))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _binary_metrics(cm: dict) -> dict:
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1_score": f1,
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def _representation(units: np.ndarray, true_hs: np.ndarray, zero: np.ndarray) -> dict:
    hs, nhs = true_hs, ~true_hs  # boolean index masks over HS / NHS rows
    n_hs, n_nhs = int(hs.sum()), int(nhs.sum())
    dead = [int(i) for i in range(units.shape[1]) if (units[:, i] == 0.0).all()]
    aucs = []
    for i in range(units.shape[1]):
        # Mann-Whitney AUC via average ranks (ties handled by rank averaging).
        order = np.argsort(units[:, i], kind="mergesort")
        ranks = np.empty(len(order), dtype=np.float64)
        sorted_vals = units[order, i]
        j = 0
        while j < len(order):
            k = j
            while k + 1 < len(order) and sorted_vals[k + 1] == sorted_vals[j]:
                k += 1
            ranks[order[j:k + 1]] = (j + k + 2) / 2.0  # 1-based average rank
            j = k + 1
        auc = (ranks[hs].sum() - n_hs * (n_hs + 1) / 2.0) / (n_hs * n_nhs)
        aucs.append(float(auc))
    hs_elevated = [i for i, a in enumerate(aucs) if a >= 0.90]
    nhs_elevated = [i for i, a in enumerate(aucs) if a <= 0.10]
    c_hs, c_nhs = units[hs].mean(axis=0), units[nhs].mean(axis=0)
    centroid = float(np.linalg.norm(c_hs - c_nhs))
    within_hs = float(np.linalg.norm(units[hs] - c_hs, axis=1).mean())
    within_nhs = float(np.linalg.norm(units[nhs] - c_nhs, axis=1).mean())
    return {
        "n": int(len(true_hs)),
        "n_hs": n_hs,
        "n_nhs": n_nhs,
        "zero_hs": int(np.sum(zero & hs)),
        "zero_nhs": int(np.sum(zero & nhs)),
        "zero_rate_hs": float(zero[hs].mean()),
        "zero_rate_nhs": float(zero[nhs].mean()),
        "zero_rate_overall": float(zero.mean()),
        "dead_units": len(dead),
        "dead_unit_indices": dead,
        "hs_elevated_units": len(hs_elevated),
        "hs_elevated_unit_indices": hs_elevated,
        "nhs_elevated_units": len(nhs_elevated),
        "max_hs_auc": float(max(aucs)),
        "centroid_distance_HS_NHS": centroid,
        "within_HS_mean_dist_to_centroid": within_hs,
        "within_NHS_mean_dist_to_centroid": within_nhs,
        "centroid_dist_over_within_NHS_ratio": centroid / within_nhs,
        "all_activations_finite": bool(np.isfinite(units).all()),
    }


def _stored_metrics(root: Path, rel: str):
    p = root / rel
    return json.loads(p.read_text()) if p.is_file() else None


def _matches_stored(stored: dict | None, cm: dict, balanced_accuracy: float, f1: float):
    if stored is None:
        return None
    ok_cm = stored.get("confusion_matrix") == cm
    ok_ba = abs(float(stored.get("balanced_accuracy", float("nan"))) - balanced_accuracy) < 1e-12
    ok_f1 = abs(float(stored.get("f1_score", float("nan"))) - f1) < 1e-12
    return bool(ok_cm and ok_ba and ok_f1)


def _audit_one(root: Path, csv_path: Path, run_col: str, run_name: str, family: str,
               seed: int, weighted: bool, balanced: bool, stored_rel: str | None) -> dict:
    units, true_hs, pred, zero, stored_pred, stored_zero = _load_condition_rows(csv_path, run_name, run_col)
    cm = _confusion(true_hs, pred)
    m = _binary_metrics(cm)
    stored = _stored_metrics(root, stored_rel) if stored_rel else None
    return {
        "run": run_name,
        "condition_family": family,
        "source": str(csv_path.relative_to(root)),
        "seed": seed,
        "class_weight_applied": weighted,
        "balanced_sampling_applied": balanced,
        **_representation(units, true_hs, zero),
        "confusion_matrix": cm,
        **m,
        "stored_metrics_file": stored_rel,
        "stored_metrics_match": _matches_stored(stored, cm, m["balanced_accuracy"], m["f1_score"]),
        "stored_zero_flags_match": bool(np.array_equal(zero, stored_zero)),
        "stored_pred_labels_match": bool(np.array_equal(pred, stored_pred)),
    }


ABLATION_STORED = {
    "baseline_noweight": "results/xai/iccad5_imbalance_ablation/training/baseline_noweight/metrics.json",
    "class_weighted": "results/xai/iccad5_checkpoint_fix/training/iccad5/metrics.json",
    "balanced_sampling": "results/xai/iccad5_imbalance_ablation/training/balanced_sampling/metrics.json",
}
ABLATION_FAMILY = {
    "baseline_noweight": "baseline_noweight",
    "class_weighted": "class_weighted",
    "balanced_sampling": "balanced_sampling",
}
SEEDREP_META = {
    "balanced_sampling_seed1": (101, True),
    "balanced_sampling_seed2": (202, True),
    "balanced_sampling_seed3": (303, True),
    "baseline_noweight_seed1": (101, False),
    "baseline_noweight_seed2": (202, False),
}


def audit(root: Path) -> dict:
    """Recompute all eight stored runs on CPU; verify against stored metrics.json."""
    root = Path(root)
    ablation_csv = root / "results/xai/iccad5_imbalance_ablation/sample_activation_comparison.csv"
    seedrep_csv = root / "results/xai/iccad5_seed_replication/sample_activation_comparison.csv"
    if not ablation_csv.is_file():
        raise FileNotFoundError(ablation_csv)
    if not seedrep_csv.is_file():
        raise FileNotFoundError(seedrep_csv)

    runs = []
    for cond in ("baseline_noweight", "class_weighted", "balanced_sampling"):
        runs.append(_audit_one(
            root, ablation_csv, "condition", cond, ABLATION_FAMILY[cond], 42,
            weighted=(cond == "class_weighted"), balanced=(cond == "balanced_sampling"),
            stored_rel=ABLATION_STORED[cond]))
    for run_name, (seed, balanced) in SEEDREP_META.items():
        runs.append(_audit_one(
            root, seedrep_csv, "run_name", run_name,
            "balanced_sampling" if balanced else "baseline_noweight", seed,
            weighted=False, balanced=balanced,
            stored_rel=f"results/xai/iccad5_seed_replication/training/{run_name}/metrics.json"))

    # Checkpoint-selection audit: the production criterion selects on the TEST split.
    stored_new = _stored_metrics(root, "results/xai/iccad5_checkpoint_fix/training/iccad5/metrics.json")
    if stored_new is None:
        raise FileNotFoundError("weighted training metrics.json is required")
    hist = stored_new["history_val_balanced_accuracy"]
    selected = int(stored_new["checkpoint_selection"]["selected_epoch"])
    selection = {
        "criterion": "best validation balanced accuracy (BestBalancedAccuracyCheckpoint, src/metrics.py)",
        "selected_epoch_zero_based": selected,
        "selected_matches_first_argmax": bool(int(np.argmax(hist)) == selected),
        "last_epoch_balanced_accuracy": float(hist[-1]),
        "validation_directory": "iccad-official/iccad5/test",
        "selection_on_test": True,
        "note": ("dataset_analysis (src/data.py) maps the 'validation' generator to the test directory; "
                 "checkpoint selection, per-epoch monitoring, and reported test metrics all evaluate the "
                 "same 19,368-image test split. No held-out validation set exists in this pipeline."),
    }

    required = [
        {"run": f"class_weighted_replication{s}", "seed": s, "status": "BLOCKED_NOT_RUN"}
        for s in REQUIRED_SEEDS
    ]
    return {
        "workstream": "H",
        "generated_by": "scripts/improvements/weighted_replication.py::audit",
        "execution_class": "CPU recomputation of stored artifacts only; no training, no neural inference",
        "checkpoint_provenance": {
            "class_weighted_checkpoint": "models/xai/retrained/xai_cnn_iccad5_best.keras",
            "sha256": _sha256(root / "models/xai/retrained/xai_cnn_iccad5_best.keras"),
            "access": "metadata only; no model load, no inference",
            "activation_csv_sha256": _sha256(ablation_csv),
        },
        "runs": runs,
        "checkpoint_selection": selection,
        "required_replications": required,
        "existing_replication_coverage": sorted(r["run"] for r in runs if r["seed"] != 42),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=DOC)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--audit-out", type=Path, default=None,
                        help="write audit JSON under results/xai/improvements/weighted_replication/")
    parser.add_argument("--replicate", action="store_true",
                        help="guarded seeds 101/202/303 class-weighted retraining (currently blocked)")
    args = parser.parse_args(argv)

    if args.replicate:
        approved = os.environ.get("WORKSTREAM_H_GPU_APPROVED") == "1"
        try:
            gpu_ok = subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
        except OSError:
            gpu_ok = False
        print("REPLICATION BLOCKED: seeds 101/202/303 class-weighted retraining not authorized.", file=sys.stderr)
        print(f"  WORKSTREAM_H_GPU_APPROVED=1 set: {approved}", file=sys.stderr)
        print(f"  nvidia-smi visible: {gpu_ok}", file=sys.stderr)
        print("  Unblocking requires: (1) lead approval + GPU serialization, (2) GPU visibility restored, "
              "(3) implementation of the exact protocol in docs/IMPROVEMENT_WEIGHTED_REPLICATION.md section 8.", file=sys.stderr)
        return 2

    result = audit(args.repo_root)
    out_dir = args.repo_root / "results/xai/improvements/weighted_replication"
    out = args.audit_out or (out_dir / "weighted_replication_audit.json")
    if not out.resolve().is_relative_to(out_dir.resolve()):
        raise ValueError("Output must remain under the owned weighted_replication directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    w = next(r for r in result["runs"] if r["run"] == "class_weighted")
    print(f"Audit written: {out}")
    print(f"class_weighted (stored, recomputed): CM={w['confusion_matrix']} "
          f"BA={w['balanced_accuracy']:.4f} HS zero-vector={w['zero_hs']}/41 dead={w['dead_units']}/16")
    for r in result["required_replications"]:
        print(f"REQUIRED-NOT-RUN: {r['run']} (seed {r['seed']}) — no training authorized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
