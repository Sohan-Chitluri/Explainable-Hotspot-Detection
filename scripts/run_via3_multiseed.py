#!/usr/bin/env python3
"""Workstream B3 / E2: Multi-Seed Robustness Runner for Via-3.

Features:
  - Multi-seed execution across predefined seeds (default: 42, 100, 2024).
  - Evaluates baseline, balanced_sampling, and class_weighted strategies.
  - Frozen Seed 42 Reference Bypass: imports audited Seed 42 reference artifacts
    with SHA-256 integrity verification, strictly preventing Seed 42 retraining.
  - Output namespacing: results isolated under {output_dir}/{strategy}/seed_{seed}/.
  - Safe resume / Overwrite protection: checks artifact existence before initiating training.
  - Computes per-seed metrics and aggregates mean, sample std, and 95% CIs.
  - Pre-training gate validation enforced.
  - Includes --dry-run mode for orchestration and statistical testing without training.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.experiment_contract import (
    CONTRACT_VERSION,
    check_pretraining_gate,
    file_sha256,
    verify_manifest_integrity,
)

SUPPORTED_STRATEGIES = ["baseline", "balanced_sampling", "class_weighted"]

KNOWN_REFERENCE_HASHES = {
    "baseline": [
        "97a9cb8bf776b76b6ed1c0fe6a38b4de31138c97cfb64a1b29ff2f228a852887",
        "b9979e8dbac5d31d8eac8bcd4156874ac7c2fcfa055200232c939922775ef4c3",
    ],
    "balanced_sampling": [
        "551a861f1e3537b93144edf22b5d2bc71456548ca576aea5a3781021fa8ac426",
    ],
    "class_weighted": [
        "63168fd5d9fbd1208cc3b290ca7880919b468c043d249d7a5dca5e53f1567ceb",
    ],
}


def compute_aggregate_statistics(per_seed_metrics: list[dict]) -> dict:
    """Compute mean, sample std, and 95% confidence intervals for all metrics."""
    n = len(per_seed_metrics)
    if n == 0:
        return {}

    metric_keys = [
        "pr_auc", "roc_auc", "recall_sensitivity", "precision", "f1_score", "balanced_accuracy"
    ]
    if all("calibrated_threshold" in m for m in per_seed_metrics):
        metric_keys.append("calibrated_threshold")

    summary = {"num_seeds": n, "metrics": {}}

    # Student's t multiplier for 95% CI (df=1 -> 12.706, df=2 -> 4.303, df=3 -> 3.182, df=4 -> 2.776)
    t_multipliers = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
    t_crit = t_multipliers.get(n - 1, 1.960)

    for k in metric_keys:
        vals = [float(m.get(k, 0.0)) for m in per_seed_metrics]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        margin = t_crit * (std_val / math.sqrt(n)) if n > 1 else 0.0

        summary["metrics"][k] = {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "ci_95": [round(mean_val - margin, 4), round(mean_val + margin, 4)],
            "values": [round(v, 4) for v in vals],
        }

    return summary


def load_reference_seed42(
    strategy: str,
    output_seed_dir: Path,
    models_dir: Path,
    reference_dir: Path = Path("results/via3_experiments/imbalance"),
    reference_models_dir: Path = Path("models/via3_imbalance"),
) -> dict:
    """Load and verify audited Seed 42 reference artifacts without retraining.

    Copies reference predictions and results JSON into output_seed_dir,
    and checkpoint into models_dir if not present.
    Verifies SHA-256 hash against KNOWN_REFERENCE_HASHES.
    """
    if strategy not in KNOWN_REFERENCE_HASHES:
        raise ValueError(f"Unknown strategy '{strategy}' for Seed 42 reference lookup.")

    ref_json = reference_dir / f"{strategy}_results_seed42.json"
    if not ref_json.exists():
        raise FileNotFoundError(f"Reference Seed 42 JSON not found at {ref_json}")

    # Checkpoint candidate search
    ckpt_candidate = reference_models_dir / f"via3_{strategy}_seed42.keras"
    if not ckpt_candidate.exists():
        fallback_ckpt = Path("models/via3/via3_baseline_seed42.keras")
        if strategy == "baseline" and fallback_ckpt.exists():
            ckpt_candidate = fallback_ckpt
        else:
            raise FileNotFoundError(f"Reference Seed 42 checkpoint not found at {ckpt_candidate}")

    actual_hash = file_sha256(ckpt_candidate)
    expected_hashes = KNOWN_REFERENCE_HASHES[strategy]
    if actual_hash not in expected_hashes:
        raise RuntimeError(
            f"Security verification failed: Seed 42 checkpoint {ckpt_candidate} hash {actual_hash} "
            f"does not match expected reference hash(es) {expected_hashes}."
        )

    # Check prediction CSVs
    clean_csv = reference_dir / f"{strategy}_clean_test_predictions.csv"
    val_csv = reference_dir / f"{strategy}_val_predictions.csv"
    official_csv = reference_dir / f"{strategy}_official_test_predictions.csv"

    for p in [clean_csv, val_csv, official_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Reference Seed 42 prediction file missing: {p}")

    with open(ref_json, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    output_seed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    dest_json = output_seed_dir / f"{strategy}_results_seed42.json"
    if not dest_json.exists() or dest_json.resolve() != ref_json.resolve():
        shutil.copy2(ref_json, dest_json)
    for src_csv in [clean_csv, val_csv, official_csv]:
        dest_csv = output_seed_dir / src_csv.name
        if not dest_csv.exists() or dest_csv.resolve() != src_csv.resolve():
            shutil.copy2(src_csv, dest_csv)

    dest_ckpt = models_dir / f"via3_{strategy}_seed42.keras"
    if not dest_ckpt.exists() or dest_ckpt.resolve() != ckpt_candidate.resolve():
        shutil.copy2(ckpt_candidate, dest_ckpt)

    print(f"[REFERENCE LOAD] Seed 42 for strategy '{strategy}' loaded from {ref_json} (Checkpoint hash verified: {actual_hash[:8]}...).")
    return ref_data


def is_seed_run_completed(seed_output_dir: Path, strategy: str, seed: int, ckpt_path: Path) -> bool:
    """Check if a seed run has already completed and produced all required artifacts."""
    result_json = seed_output_dir / f"{strategy}_results_seed{seed}.json"
    clean_csv = seed_output_dir / f"{strategy}_clean_test_predictions.csv"
    val_csv = seed_output_dir / f"{strategy}_val_predictions.csv"
    official_csv = seed_output_dir / f"{strategy}_official_test_predictions.csv"

    if not result_json.exists() or result_json.stat().st_size == 0:
        return False
    if not clean_csv.exists() or clean_csv.stat().st_size == 0:
        return False
    if not val_csv.exists() or val_csv.stat().st_size == 0:
        return False
    if not official_csv.exists() or official_csv.stat().st_size == 0:
        return False
    if not ckpt_path.exists() or ckpt_path.stat().st_size == 0:
        return False

    return True


def run_multiseed_orchestration(
    seeds: list[int],
    manifest_dir: Path,
    output_dir: Path,
    models_dir: Path,
    epochs: int = 15,
    strategy: str = "baseline",
    dry_run: bool = False,
    smoke: bool = False,
    override_gate: bool = False,
    reference_dir: Path = Path("results/via3_experiments/imbalance"),
    reference_models_dir: Path = Path("models/via3_imbalance"),
) -> dict:
    """Orchestrate multi-seed runs and aggregate statistical results."""
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Supported strategies: {SUPPORTED_STRATEGIES}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    strategy_output_dir = output_dir / strategy
    strategy_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Manifest Integrity Check
    manifest_status = verify_manifest_integrity(manifest_dir)
    if not all(manifest_status.values()):
        raise RuntimeError(f"Manifest integrity verification failed: {manifest_status}")

    # 2. Gate Verification Check
    gate_status = check_pretraining_gate()
    if not gate_status["gate_open"] and not dry_run and not override_gate:
        print("\n[CRITICAL ERROR] Pre-training verification gate is CLOSED.")
        print("Full multi-seed training is forbidden until pre-training verification explicitly passes.")
        print("Use --dry-run to test statistical aggregation and orchestration logic.")
        raise PermissionError(f"Pre-training verification gate is CLOSED: {gate_status.get('reason')}")

    if dry_run:
        print(f"[DRY-RUN] Simulating multi-seed orchestration across seeds {seeds} for strategy '{strategy}'...")
        synthetic_seed_results = []
        base_pr = 0.258 if strategy == "baseline" else (0.285 if strategy == "balanced_sampling" else 0.260)
        for i, s in enumerate(seeds):
            m = {
                "seed": s,
                "pr_auc": round(base_pr + (i * 0.005) - 0.005, 4),
                "roc_auc": round(0.740 + (i * 0.004) - 0.004, 4),
                "recall_sensitivity": round(0.032 + (i * 0.002) - 0.002, 4),
                "precision": 1.0,
                "f1_score": round(0.061 + (i * 0.003) - 0.003, 4),
                "balanced_accuracy": round(0.516 + (i * 0.001) - 0.001, 4),
            }
            synthetic_seed_results.append(m)

        agg = compute_aggregate_statistics(synthetic_seed_results)
        manifest = {
            "mode": "dry_run",
            "is_synthetic": True,
            "contract_version": CONTRACT_VERSION,
            "strategy": strategy,
            "seeds": seeds,
            "per_seed_results": synthetic_seed_results,
            "aggregate_statistics": agg,
            "status": "PASS_DRY_RUN",
        }
        # Save dry run manifest to both output_dir and strategy_output_dir
        out_json_root = output_dir / f"multiseed_{strategy}_dry_run_manifest.json"
        with open(out_json_root, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        out_json_strat = strategy_output_dir / f"multiseed_{strategy}_dry_run_manifest.json"
        with open(out_json_strat, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"Dry run aggregate statistics calculated. Manifest saved to {out_json_root}")
        return manifest

    # Real execution
    from scripts.run_via3_imbalance_experiments import train_single_strategy
    from src.openhsd_data import load_via3_manifest_datasets

    per_seed_results = []
    per_seed_full_records = []

    for s in seeds:
        seed_dir = strategy_output_dir / f"seed_{s}"
        ckpt_path = models_dir / f"via3_{strategy}_seed{s}.keras"

        if s == 42:
            print(f"\n[REFERENCE] Seed 42 is the audited reference. Loading without retraining...")
            res = load_reference_seed42(
                strategy=strategy,
                output_seed_dir=seed_dir,
                models_dir=models_dir,
                reference_dir=reference_dir,
                reference_models_dir=reference_models_dir,
            )
        else:
            if is_seed_run_completed(seed_dir, strategy, s, ckpt_path):
                print(f"\n[SAFE RESUME] Seed {s} already completed for strategy '{strategy}'. Loading existing results.")
                res_json = seed_dir / f"{strategy}_results_seed{s}.json"
                with open(res_json, "r", encoding="utf-8") as f:
                    res = json.load(f)
            else:
                print(f"\n[EXECUTION] Launching training for strategy '{strategy}', Seed {s} (Epochs: {epochs})...")
                data = load_via3_manifest_datasets(
                    manifest_dir=manifest_dir,
                    batch_size=32,
                    seed=s,
                    smoke=smoke,
                    smoke_samples=256 if smoke else 0,
                )
                res = train_single_strategy(
                    strategy=strategy,
                    data=data,
                    output_dir=seed_dir,
                    models_dir=models_dir,
                    epochs=epochs,
                    seed=s,
                )

        metric_entry = dict(res["metrics_clean_test_calibrated"])
        metric_entry["seed"] = s
        metric_entry["calibrated_threshold"] = res.get("calibrated_threshold")
        metric_entry["best_epoch"] = res.get("best_epoch")
        metric_entry["checkpoint"] = res.get("checkpoint")
        per_seed_results.append(metric_entry)
        per_seed_full_records.append(res)

    agg = compute_aggregate_statistics(per_seed_results)
    final_output = {
        "contract_version": CONTRACT_VERSION,
        "strategy": strategy,
        "seeds": seeds,
        "per_seed_results": per_seed_results,
        "aggregate_statistics": agg,
        "is_synthetic": False,
        "status": "PASS_MULTISEED",
    }

    # Save to both strategy subdirectory and root output_dir
    out_summary_strat = strategy_output_dir / f"multiseed_{strategy}_summary.json"
    with open(out_summary_strat, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    out_summary_root = output_dir / f"multiseed_{strategy}_summary.json"
    with open(out_summary_root, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n[COMPLETE] Multi-seed orchestration for strategy '{strategy}' across seeds {seeds} finished.")
    print(f"Summary written to {out_summary_strat} and {out_summary_root}")
    return final_output


def main():
    parser = argparse.ArgumentParser(description="Workstream B3 Multi-Seed Variance Runner")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2024])
    parser.add_argument("--strategy", type=str, default="baseline", choices=SUPPORTED_STRATEGIES)
    parser.add_argument("--manifest-dir", type=Path, default=Path("results/dataset_audit"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/via3_experiments/multiseed"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/via3_multiseed"))
    parser.add_argument("--reference-dir", type=Path, default=Path("results/via3_experiments/imbalance"))
    parser.add_argument("--reference-models-dir", type=Path, default=Path("models/via3_imbalance"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--override-gate", action="store_true")
    args = parser.parse_args()

    run_multiseed_orchestration(
        seeds=args.seeds,
        manifest_dir=args.manifest_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        epochs=2 if args.smoke else args.epochs,
        strategy=args.strategy,
        dry_run=args.dry_run,
        smoke=args.smoke,
        override_gate=args.override_gate,
        reference_dir=args.reference_dir,
        reference_models_dir=args.reference_models_dir,
    )


if __name__ == "__main__":
    main()
