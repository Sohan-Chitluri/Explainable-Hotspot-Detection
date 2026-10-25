#!/usr/bin/env python3
"""Workstream E4.2: Multi-Seed XAI Robustness Runner for Via-3.

Executes controlled XAI robustness evaluation across 6 verified checkpoints:
  - Baseline: Seeds 42, 100, 2024
  - Balanced Sampling: Seeds 42, 100, 2024
on the frozen 50-layout diagnostic cohort.

Evaluates:
  - Methods: Grad-CAM, Grad-CAM++, LayerCAM, Saliency, Occlusion, Random
  - Faithfulness: 10-step progressive insertion & deletion AUC (trapezoidal rule)
  - Stability: Spearman rank correlation under additive Gaussian noise (sigma=0.05, T=3 trials)
  - Collapse & Contrast diagnostics: spatial standard deviation & flat-CAM detection
  - Parameter Randomization sanity check (Adebayo check on 5 curated samples)
  - Safe resume / Overwrite protection per checkpoint
  - Isolated outputs under results/via3_experiments/xai_multiseed/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
import tensorflow as tf

from src.experiment_contract import CONTRACT_VERSION, file_sha256
from src.faithfulness import compute_deletion_insertion_curves
from src.model import build_custom_cnn
from src.occlusion import OcclusionSensitivity
from src.openhsd_data import _load_and_preprocess_image
from src.xai_engine import XAIEngine

FROZEN_COHORT_SHA = "1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb"
EVAL_METHODS = ["gradcam", "gradcam_plus_plus", "layercam", "saliency", "occlusion", "random"]
STABILITY_METHODS = ["gradcam", "gradcam_plus_plus", "layercam", "saliency", "occlusion"]

EXPECTED_CHECKPOINTS = {
    ("baseline", 42): "97a9cb8bf776b76b6ed1c0fe6a38b4de31138c97cfb64a1b29ff2f228a852887",
    ("baseline", 100): "f28e4277738794a64d557fd5a5c3eeead32d32f8cbf08642ad135554746d3455",
    ("baseline", 2024): "9347e568cbe59672f295bd3865bc9708d0a279ecd5c4e94ddf54a1b96cf5ae0d",
    ("balanced_sampling", 42): "551a861f1e3537b93144edf22b5d2bc71456548ca576aea5a3781021fa8ac426",
    ("balanced_sampling", 100): "c957304bf07a8aabf9a7861c4b2d5bf1c050c87bba0454ae4cbed9d086ef5815",
    ("balanced_sampling", 2024): "297fa95748adc7a1234467cee898f14872942d5d4b1484e539bd1308f39b89e9",
}


def compute_explanation_stability(
    engine: XAIEngine,
    img_tensor: np.ndarray,
    method: str = "gradcam",
    target_idx: int = 1,
    noise_sigma: float = 0.05,
    num_trials: int = 3,
    seed: int = 42,
) -> float:
    """Compute local attribution stability (Spearman rho) under additive Gaussian noise."""
    rng = np.random.default_rng(seed)

    def get_map(x_tensor):
        if method == "gradcam":
            _, m, _ = engine.explain_gradcam(x_tensor, target_class=target_idx)
        elif method == "gradcam_plus_plus":
            _, m, _ = engine.explain_gradcam_plus_plus(x_tensor, target_class=target_idx)
        elif method == "layercam":
            _, m, _ = engine.explain_layercam(x_tensor, target_class=target_idx)
        elif method == "saliency":
            _, m, _ = engine.explain_gradient_saliency(x_tensor, target_class=target_idx)
        elif method == "occlusion":
            occ = OcclusionSensitivity(engine, window_size=(32, 32), stride=(16, 16), baseline_value=0.0)
            res_occ = occ.explain(x_tensor, target_class=target_idx)
            m = res_occ["norm_map"]
        else:
            return np.zeros((224, 224), dtype=np.float32)
        return m

    base_map = get_map(img_tensor).flatten()
    if np.all(base_map == 0.0) or np.std(base_map) < 1e-7:
        return 0.0

    correlations = []
    for _ in range(num_trials):
        noise = rng.normal(0, noise_sigma, img_tensor.shape).astype(np.float32)
        perturbed_img = np.clip(img_tensor + noise, 0.0, 1.0)
        pert_map = get_map(perturbed_img).flatten()
        if np.std(pert_map) > 1e-7:
            rho, _ = spearmanr(base_map, pert_map)
            if not np.isnan(rho):
                correlations.append(float(rho))

    return float(np.mean(correlations)) if correlations else 0.0


def is_checkpoint_xai_completed(output_seed_dir: Path, expected_samples: int) -> bool:
    """Check if a checkpoint's XAI evaluation has already completed and produced full outputs."""
    faith_csv = output_seed_dir / "faithfulness_metrics.csv"
    stab_csv = output_seed_dir / "stability_metrics.csv"
    coll_csv = output_seed_dir / "collapse_diagnostics.csv"
    sum_json = output_seed_dir / "xai_summary.json"

    if not all(p.is_file() for p in [faith_csv, stab_csv, coll_csv, sum_json]):
        return False

    try:
        df_faith = pd.read_csv(faith_csv)
        df_stab = pd.read_csv(stab_csv)
        df_coll = pd.read_csv(coll_csv)

        expected_faith = expected_samples * len(EVAL_METHODS)
        expected_stab = expected_samples * len(STABILITY_METHODS)
        expected_coll = expected_samples

        if len(df_faith) != expected_faith or len(df_stab) != expected_stab or len(df_coll) != expected_coll:
            return False
        return True
    except Exception:
        return False


def evaluate_single_checkpoint(
    strategy: str,
    seed: int,
    checkpoint_path: Path,
    cohort: List[Dict],
    data_dir: Path,
    output_seed_dir: Path,
    steps: int = 10,
    dry_run: bool = False,
) -> Dict:
    """Run full XAI evaluation for a single model checkpoint across cohort samples."""
    output_seed_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    model_key = f"{strategy}_seed{seed}"
    print("\n" + "=" * 75)
    print(f"EVALUATING XAI FOR: {model_key.upper()} (Checkpoint: {checkpoint_path.name})")
    print("=" * 75)

    if dry_run:
        print(f"[DRY-RUN] Generating synthetic deterministic fixtures for {model_key}...")
        faith_rows = []
        stab_rows = []
        coll_rows = []
        for s in cohort:
            fn = s["filename"]
            c = s["cohort"]
            gt = s["ground_truth"]
            for m in EVAL_METHODS:
                faith_rows.append({
                    "model": model_key,
                    "strategy": strategy,
                    "seed": seed,
                    "filename": fn,
                    "cohort": c,
                    "ground_truth": gt,
                    "method": m,
                    "deletion_auc": round(0.35 + (0.01 * (seed % 10)), 4),
                    "insertion_auc": round(0.55 + (0.01 * (seed % 10)), 4),
                    "status": "SUCCESS",
                })
            for m in STABILITY_METHODS:
                stab_rows.append({
                    "model": model_key,
                    "strategy": strategy,
                    "seed": seed,
                    "filename": fn,
                    "cohort": c,
                    "ground_truth": gt,
                    "method": m,
                    "stability_spearman_rho": round(0.85 + (0.01 * (seed % 10)), 4),
                    "status": "SUCCESS",
                })
            coll_rows.append({
                "model": model_key,
                "strategy": strategy,
                "seed": seed,
                "filename": fn,
                "cohort": c,
                "ground_truth": gt,
                "is_flat_cam": False,
                "cam_std": 0.145,
                "cam_max": 1.0,
                "cam_mean": 0.18,
                "status": "SUCCESS",
            })

        pd.DataFrame(faith_rows).to_csv(output_seed_dir / "faithfulness_metrics.csv", index=False)
        pd.DataFrame(stab_rows).to_csv(output_seed_dir / "stability_metrics.csv", index=False)
        pd.DataFrame(coll_rows).to_csv(output_seed_dir / "collapse_diagnostics.csv", index=False)

        summary = {
            "model_key": model_key,
            "strategy": strategy,
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "samples_evaluated": len(cohort),
            "status": "PASS_DRY_RUN",
            "eval_duration_s": 0.1,
            "method_metrics": {
                m: {
                    "mean_insertion_auc": 0.55,
                    "mean_deletion_auc": 0.35,
                    "mean_stability_rho": 0.85 if m != "random" else None,
                }
                for m in EVAL_METHODS
            },
        }
        with open(output_seed_dir / "xai_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    # Real evaluation
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(str(checkpoint_path), compile=False)

    # Runtime verification of CAM layer
    conv_layer_names = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    assert conv_layer_names[-1] == "conv2d_5", f"Unexpected final conv layer: {conv_layer_names[-1]}"
    target_layer = model.get_layer("conv2d_5")
    assert target_layer.output_shape == (None, 15, 15, 12), f"Unexpected shape: {target_layer.output_shape}"

    engine = XAIEngine(model, target_layer_name="conv2d_5", positive_class_is_1=True)

    test_img_dir = data_dir / "test"
    faithfulness_rows = []
    stability_rows = []
    collapse_rows = []

    for idx_s, item in enumerate(cohort):
        fn = item["filename"]
        cohort_tag = item["cohort"]
        gt = item["ground_truth"]
        img_p = test_img_dir / fn
        assert img_p.is_file(), f"Image not found: {img_p}"

        tensor_hwc = _load_and_preprocess_image(str(img_p))
        tensor = np.expand_dims(tensor_hwc, axis=0)

        # 1. Faithfulness curves (insertion / deletion)
        try:
            curves = compute_deletion_insertion_curves(
                model=model,
                engine=engine,
                img_tensor=tensor,
                target_idx=1,  # Pre-sigmoid Hotspot logit target
                steps=steps,
                methods=EVAL_METHODS,
            )
            for meth in EVAL_METHODS:
                faithfulness_rows.append({
                    "model": model_key,
                    "strategy": strategy,
                    "seed": seed,
                    "filename": fn,
                    "cohort": cohort_tag,
                    "ground_truth": gt,
                    "method": meth,
                    "deletion_auc": round(curves[meth]["deletion_auc"], 4),
                    "insertion_auc": round(curves[meth]["insertion_auc"], 4),
                    "status": "SUCCESS",
                })
        except Exception as e:
            for meth in EVAL_METHODS:
                faithfulness_rows.append({
                    "model": model_key,
                    "strategy": strategy,
                    "seed": seed,
                    "filename": fn,
                    "cohort": cohort_tag,
                    "ground_truth": gt,
                    "method": meth,
                    "deletion_auc": np.nan,
                    "insertion_auc": np.nan,
                    "status": f"FAILED: {e}",
                })

        # 2. Stability under input perturbation
        for meth in STABILITY_METHODS:
            try:
                rho = compute_explanation_stability(
                    engine=engine,
                    img_tensor=tensor,
                    method=meth,
                    target_idx=1,
                    noise_sigma=0.05,
                    num_trials=3,
                    seed=42,
                )
                stability_rows.append({
                    "model": model_key,
                    "strategy": strategy,
                    "seed": seed,
                    "filename": fn,
                    "cohort": cohort_tag,
                    "ground_truth": gt,
                    "method": meth,
                    "stability_spearman_rho": round(rho, 4),
                    "status": "SUCCESS",
                })
            except Exception as e:
                stability_rows.append({
                    "model": model_key,
                    "strategy": strategy,
                    "seed": seed,
                    "filename": fn,
                    "cohort": cohort_tag,
                    "ground_truth": gt,
                    "method": meth,
                    "stability_spearman_rho": np.nan,
                    "status": f"FAILED: {e}",
                })

        # 3. Collapse diagnostics (Flat-CAM check on Grad-CAM)
        try:
            _, cam_g, _ = engine.explain_gradcam(tensor, target_class=1)
            is_flat = bool(np.std(cam_g) < 1e-7 or np.all(cam_g == 0.0))
            collapse_rows.append({
                "model": model_key,
                "strategy": strategy,
                "seed": seed,
                "filename": fn,
                "cohort": cohort_tag,
                "ground_truth": gt,
                "is_flat_cam": is_flat,
                "cam_std": round(float(np.std(cam_g)), 4),
                "cam_max": round(float(np.max(cam_g)), 4),
                "cam_mean": round(float(np.mean(cam_g)), 4),
                "status": "SUCCESS",
            })
        except Exception as e:
            collapse_rows.append({
                "model": model_key,
                "strategy": strategy,
                "seed": seed,
                "filename": fn,
                "cohort": cohort_tag,
                "ground_truth": gt,
                "is_flat_cam": True,
                "cam_std": np.nan,
                "cam_max": np.nan,
                "cam_mean": np.nan,
                "status": f"FAILED: {e}",
            })

        if (idx_s + 1) % 10 == 0 or idx_s == len(cohort) - 1:
            print(f"  Processed {idx_s + 1:2d} / {len(cohort)} samples...")

    # 4. Adebayo parameter randomization check on 5 curated samples
    print(f"  Executing Adebayo parameter randomization sanity check on 5 samples...")
    adebayo_records = []
    try:
        rand_m = tf.keras.models.clone_model(model)
        for layer in rand_m.layers:
            if hasattr(layer, "kernel_initializer"):
                layer.set_weights([np.random.normal(0, 0.05, w.shape) for w in layer.get_weights()])
        rand_engine = XAIEngine(rand_m, target_layer_name="conv2d_5", positive_class_is_1=True)

        for s in cohort[:5]:
            fn = s["filename"]
            img_p = test_img_dir / fn
            tensor_hwc = _load_and_preprocess_image(str(img_p))
            tensor = np.expand_dims(tensor_hwc, axis=0)

            _, orig_cam, _ = engine.explain_gradcam(tensor, target_class=1)
            _, rand_cam, _ = rand_engine.explain_gradcam(tensor, target_class=1)

            rho, _ = spearmanr(orig_cam.flatten(), rand_cam.flatten())
            adebayo_records.append({
                "filename": fn,
                "spearman_rho_vs_randomized": round(float(rho) if not np.isnan(rho) else 0.0, 4),
            })
        del rand_m, rand_engine
    except Exception as e:
        print(f"  Adebayo check encountered error: {e}")

    adebayo_mean_rho = float(np.mean([r["spearman_rho_vs_randomized"] for r in adebayo_records])) if adebayo_records else np.nan

    duration = time.time() - t0
    print(f"Evaluation of {model_key} completed in {duration:.2f}s (Adebayo rho: {adebayo_mean_rho:.4f})")

    # Save per-model CSVs
    df_faith = pd.DataFrame(faithfulness_rows)
    df_stab = pd.DataFrame(stability_rows)
    df_coll = pd.DataFrame(collapse_rows)

    df_faith.to_csv(output_seed_dir / "faithfulness_metrics.csv", index=False)
    df_stab.to_csv(output_seed_dir / "stability_metrics.csv", index=False)
    df_coll.to_csv(output_seed_dir / "collapse_diagnostics.csv", index=False)

    # Compute per-model method summaries
    method_metrics = {}
    for m in EVAL_METHODS:
        sub_f = df_faith[df_faith["method"] == m]
        sub_s = df_stab[df_stab["method"] == m]
        method_metrics[m] = {
            "mean_insertion_auc": round(float(sub_f["insertion_auc"].mean()), 4),
            "std_insertion_auc": round(float(sub_f["insertion_auc"].std(ddof=1)), 4),
            "mean_deletion_auc": round(float(sub_f["deletion_auc"].mean()), 4),
            "std_deletion_auc": round(float(sub_f["deletion_auc"].std(ddof=1)), 4),
            "mean_stability_rho": round(float(sub_s["stability_spearman_rho"].mean()), 4) if not sub_s.empty else None,
            "std_stability_rho": round(float(sub_s["stability_spearman_rho"].std(ddof=1)), 4) if not sub_s.empty else None,
        }

    summary = {
        "model_key": model_key,
        "strategy": strategy,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "samples_evaluated": len(cohort),
        "eval_duration_s": round(duration, 2),
        "status": "SUCCESS",
        "method_metrics": method_metrics,
        "collapse_diagnostics": {
            "flat_cam_rate": round(float(df_coll["is_flat_cam"].mean()), 4),
            "num_flat_samples": int(df_coll["is_flat_cam"].sum()),
            "mean_cam_spatial_std": round(float(df_coll["cam_std"].mean()), 4),
            "std_cam_spatial_std": round(float(df_coll["cam_std"].std(ddof=1)), 4),
        },
        "adebayo_sanity_check": {
            "mean_spearman_rho_vs_randomized": round(adebayo_mean_rho, 4) if not np.isnan(adebayo_mean_rho) else None,
            "passed_sanity_check": bool(abs(adebayo_mean_rho) < 0.20) if not np.isnan(adebayo_mean_rho) else False,
            "sample_records": adebayo_records,
        },
    }

    with open(output_seed_dir / "xai_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    del model, engine
    tf.keras.backend.clear_session()
    return summary


def run_multiseed_xai(
    strategies: List[str] = ["baseline", "balanced_sampling"],
    seeds: List[int] = [42, 100, 2024],
    cohort_manifest: Path = Path("results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json"),
    data_dir: Path = Path("/home/peskybird/Projects/data/openhsd/via-3"),
    models_dir: Path = Path("models/via3_multiseed"),
    output_dir: Path = Path("results/via3_experiments/xai_multiseed"),
    steps: int = 10,
    smoke: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> Dict:
    """Orchestrate sequential multi-seed XAI evaluation and aggregate statistical results."""
    t_global_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"WORKSTREAM E4.2: MULTI-SEED XAI ROBUSTNESS SUITE (Smoke: {smoke}, Dry-Run: {dry_run})")
    print("=" * 80)

    # 1. Preflight Cohort Verification
    assert cohort_manifest.is_file(), f"Cohort manifest not found: {cohort_manifest}"
    cohort_sha = file_sha256(cohort_manifest)
    assert cohort_sha == FROZEN_COHORT_SHA, f"Cohort SHA-256 mismatch! Expected {FROZEN_COHORT_SHA}, got {cohort_sha}"

    with open(cohort_manifest, "r", encoding="utf-8") as f:
        cohort = json.load(f)
    assert len(cohort) == 50, f"Expected 50 cohort items, got {len(cohort)}"
    print(f"[PREFLIGHT 1/4] Frozen cohort manifest verified (SHA: {cohort_sha[:16]}..., 50 layouts).")

    # Freeze a copy in output directory
    cohort_frozen_dir = output_dir / "cohort"
    cohort_frozen_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cohort_manifest, cohort_frozen_dir / "xai_sample_cohort_manifest.json")

    eval_cohort = cohort[:2] if smoke else cohort
    expected_samples = len(eval_cohort)

    # 2. Preflight Checkpoints Verification
    if not dry_run:
        for strat in strategies:
            for s in seeds:
                ckpt_p = models_dir / f"via3_{strat}_seed{s}.keras"
                assert ckpt_p.is_file(), f"Missing checkpoint: {ckpt_p}"
                actual_sha = file_sha256(ckpt_p)
                exp_sha = EXPECTED_CHECKPOINTS.get((strat, s))
                if exp_sha:
                    assert actual_sha == exp_sha, f"Checkpoint hash mismatch on {ckpt_p.name}: {actual_sha} vs {exp_sha}"
        print(f"[PREFLIGHT 2/4] All {len(strategies) * len(seeds)} checkpoints verified against expected hashes.")
    else:
        print("[PREFLIGHT 2/4] Dry-run mode: skipping physical checkpoint verification.")

    # 3. Sequential Execution Loop
    all_summaries = {}
    master_faith_rows = []
    master_stab_rows = []
    master_coll_rows = []

    for strat in strategies:
        for s in seeds:
            ckpt_p = models_dir / f"via3_{strat}_seed{s}.keras"
            seed_out = output_dir / strat / f"seed_{s}"

            if not force and not smoke and is_checkpoint_xai_completed(seed_out, expected_samples):
                print(f"\n[SAFE RESUME] Found completed XAI evaluation for {strat}_seed{s} at {seed_out}. Loading results...")
                with open(seed_out / "xai_summary.json", "r", encoding="utf-8") as f:
                    summ = json.load(f)
            else:
                summ = evaluate_single_checkpoint(
                    strategy=strat,
                    seed=s,
                    checkpoint_path=ckpt_p,
                    cohort=eval_cohort,
                    data_dir=data_dir,
                    output_seed_dir=seed_out,
                    steps=steps,
                    dry_run=dry_run,
                )

            all_summaries[f"{strat}_seed{s}"] = summ

            # Load rows for master compilation
            df_f = pd.read_csv(seed_out / "faithfulness_metrics.csv")
            df_s = pd.read_csv(seed_out / "stability_metrics.csv")
            df_c = pd.read_csv(seed_out / "collapse_diagnostics.csv")
            master_faith_rows.extend(df_f.to_dict(orient="records"))
            master_stab_rows.extend(df_s.to_dict(orient="records"))
            master_coll_rows.extend(df_c.to_dict(orient="records"))

    # 4. Save Master CSVs
    df_all_faith = pd.DataFrame(master_faith_rows)
    df_all_stab = pd.DataFrame(master_stab_rows)
    df_all_coll = pd.DataFrame(master_coll_rows)

    df_all_faith.to_csv(output_dir / "master_faithfulness_metrics.csv", index=False)
    df_all_stab.to_csv(output_dir / "master_stability_metrics.csv", index=False)
    df_all_coll.to_csv(output_dir / "master_collapse_diagnostics.csv", index=False)

    # 5. Multi-Seed Aggregation & Paired Differences Analysis
    aggregation_results = {}
    for strat in strategies:
        strat_agg = {}
        for m in EVAL_METHODS:
            ins_seed_means = [all_summaries[f"{strat}_seed{s}"]["method_metrics"][m]["mean_insertion_auc"] for s in seeds]
            del_seed_means = [all_summaries[f"{strat}_seed{s}"]["method_metrics"][m]["mean_deletion_auc"] for s in seeds]
            stab_seed_means = [
                all_summaries[f"{strat}_seed{s}"]["method_metrics"][m]["mean_stability_rho"]
                for s in seeds
                if all_summaries[f"{strat}_seed{s}"]["method_metrics"][m]["mean_stability_rho"] is not None
            ]

            strat_agg[m] = {
                "insertion_auc": {
                    "mean": round(float(np.mean(ins_seed_means)), 4),
                    "std": round(float(np.std(ins_seed_means, ddof=1)), 4),
                    "seed_values": ins_seed_means,
                },
                "deletion_auc": {
                    "mean": round(float(np.mean(del_seed_means)), 4),
                    "std": round(float(np.std(del_seed_means, ddof=1)), 4),
                    "seed_values": del_seed_means,
                },
            }
            if stab_seed_means:
                strat_agg[m]["stability_rho"] = {
                    "mean": round(float(np.mean(stab_seed_means)), 4),
                    "std": round(float(np.std(stab_seed_means, ddof=1)), 4),
                    "seed_values": stab_seed_means,
                }
        aggregation_results[strat] = strat_agg

    # Paired Strategy Comparison across Seeds
    paired_comparison = {}
    if "baseline" in strategies and "balanced_sampling" in strategies:
        can_pair = all(
            f"balanced_sampling_seed{s}" in all_summaries and f"baseline_seed{s}" in all_summaries
            for s in seeds
        )
        if can_pair:
            for m in EVAL_METHODS:
                diff_ins = [
                    all_summaries[f"balanced_sampling_seed{s}"]["method_metrics"][m]["mean_insertion_auc"]
                    - all_summaries[f"baseline_seed{s}"]["method_metrics"][m]["mean_insertion_auc"]
                    for s in seeds
                ]
                diff_del = [
                    all_summaries[f"balanced_sampling_seed{s}"]["method_metrics"][m]["mean_deletion_auc"]
                    - all_summaries[f"baseline_seed{s}"]["method_metrics"][m]["mean_deletion_auc"]
                    for s in seeds
                ]
                paired_comparison[m] = {
                    "delta_insertion_auc": {
                        "mean": round(float(np.mean(diff_ins)), 4),
                        "std": round(float(np.std(diff_ins, ddof=1)), 4),
                        "seed_deltas": [round(d, 4) for d in diff_ins],
                        "consistent_positive": bool(all(d > 0 for d in diff_ins)),
                    },
                    "delta_deletion_auc": {
                        "mean": round(float(np.mean(diff_del)), 4),
                        "std": round(float(np.std(diff_del, ddof=1)), 4),
                        "seed_deltas": [round(d, 4) for d in diff_del],
                        "consistent_positive": bool(all(d > 0 for d in diff_del)),
                    },
                }
                if m in STABILITY_METHODS:
                    diff_stab = [
                        all_summaries[f"balanced_sampling_seed{s}"]["method_metrics"][m]["mean_stability_rho"]
                        - all_summaries[f"baseline_seed{s}"]["method_metrics"][m]["mean_stability_rho"]
                        for s in seeds
                    ]
                    paired_comparison[m]["delta_stability_rho"] = {
                        "mean": round(float(np.mean(diff_stab)), 4),
                        "std": round(float(np.std(diff_stab, ddof=1)), 4),
                        "seed_deltas": [round(d, 4) for d in diff_stab],
                        "consistent_positive": bool(all(d > 0 for d in diff_stab)),
                    }

    # Master Summary Document
    master_summary = {
        "contract_version": CONTRACT_VERSION,
        "evaluation_name": "Workstream E4.2 Multi-Seed XAI Robustness Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_evaluations_planned": len(strategies) * len(seeds) * len(eval_cohort) * len(EVAL_METHODS),
        "total_evaluations_completed": len(df_all_faith[df_all_faith["status"] == "SUCCESS"]),
        "total_evaluations_failed": len(df_all_faith[df_all_faith["status"] != "SUCCESS"]),
        "models_evaluated": list(all_summaries.keys()),
        "strategies": strategies,
        "seeds": seeds,
        "diagnostic_cohort_samples": len(eval_cohort),
        "methods_evaluated": EVAL_METHODS,
        "stability_methods": STABILITY_METHODS,
        "per_model_summaries": all_summaries,
        "cross_seed_aggregations": aggregation_results,
        "paired_strategy_differences": paired_comparison,
        "total_suite_duration_s": round(time.time() - t_global_start, 2),
        "status": "SUCCESS" if len(df_all_faith[df_all_faith["status"] == "SUCCESS"]) == len(df_all_faith) else "PARTIAL_FAILURE",
    }

    sum_out = output_dir / "multiseed_xai_comparison_summary.json"
    with open(sum_out, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    # 6. Reproducibility Manifest
    rep_manifest = {
        "contract_version": CONTRACT_VERSION,
        "cohort_manifest_sha256": cohort_sha,
        "checkpoints": {f"{strat}_seed{s}": EXPECTED_CHECKPOINTS.get((strat, s)) for strat in strategies for s in seeds},
        "python_version": sys.version,
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "output_files": {
            "master_faithfulness_metrics_csv": {
                "path": str(output_dir / "master_faithfulness_metrics.csv"),
                "sha256": file_sha256(output_dir / "master_faithfulness_metrics.csv"),
                "rows": len(df_all_faith),
            },
            "master_stability_metrics_csv": {
                "path": str(output_dir / "master_stability_metrics.csv"),
                "sha256": file_sha256(output_dir / "master_stability_metrics.csv"),
                "rows": len(df_all_stab),
            },
            "master_collapse_diagnostics_csv": {
                "path": str(output_dir / "master_collapse_diagnostics.csv"),
                "sha256": file_sha256(output_dir / "master_collapse_diagnostics.csv"),
                "rows": len(df_all_coll),
            },
            "multiseed_xai_comparison_summary_json": {
                "path": str(sum_out),
                "sha256": file_sha256(sum_out),
            },
        },
        "random_seeds": {
            "evaluation_seeds": seeds,
            "random_baseline_seed": 42,
            "stability_perturbation_seed": 42,
        },
    }
    with open(output_dir / "reproducibility_manifest.json", "w", encoding="utf-8") as f:
        json.dump(rep_manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("WORKSTREAM E4.2: MULTI-SEED XAI EVALUATION COMPLETED SUCCESSFULLY")
    print(f"Total Evaluations Completed: {master_summary['total_evaluations_completed']} / {master_summary['total_evaluations_planned']}")
    print(f"Summary JSON: {sum_out}")
    print(f"Reproducibility Manifest: {output_dir / 'reproducibility_manifest.json'}")
    print("=" * 80)
    return master_summary


def main():
    parser = argparse.ArgumentParser(description="Workstream E4.2 Multi-Seed XAI Robustness Runner")
    parser.add_argument("--strategies", nargs="+", type=str, default=["baseline", "balanced_sampling"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2024])
    parser.add_argument("--cohort-manifest", type=Path, default=Path("results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("/home/peskybird/Projects/data/openhsd/via-3"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/via3_multiseed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/via3_experiments/xai_multiseed"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="Run 2-sample evaluation for execution pipeline validation")
    parser.add_argument("--dry-run", action="store_true", help="Run synthetic pipeline verification without model loading")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if completed artifacts exist")
    args = parser.parse_args()

    run_multiseed_xai(
        strategies=args.strategies,
        seeds=args.seeds,
        cohort_manifest=args.cohort_manifest,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        steps=args.steps,
        smoke=args.smoke,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
