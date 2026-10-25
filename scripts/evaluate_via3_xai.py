#!/usr/bin/env python3
"""Workstream C: Comparative XAI Faithfulness and Stability Evaluation Suite for Via-3.

Evaluates and compares explanations from:
  1. Baseline unweighted custom CNN (models/via3/via3_baseline_seed42.keras)
  2. Balanced sampling custom CNN (models/via3_imbalance/via3_balanced_sampling_seed42.keras)
  3. Class-weighted custom CNN (optional control: models/via3_imbalance/via3_class_weighted_seed42.keras)

Methods evaluated:
  1. Grad-CAM (Selvaraju et al., 2017)
  2. Grad-CAM++ (Chattopadhay et al., 2018)
  3. LayerCAM (Jiang et al., 2021)
  4. Gradient Saliency (Simonyan et al., 2013)
  5. Occlusion Sensitivity (Zeiler & Fergus, 2014)
  6. Random-Region Baseline (sanity check)

Evaluation Procedures:
  - Deletion curves & Area-Under-Deletion (AUD)
  - Insertion curves & Area-Under-Insertion (AUI)
  - Explanation stability under controlled input perturbations (Spearman rho)
  - Model parameter randomization sanity checks (Adebayo et al., 2018)
  - Representation collapse & flat attribution map diagnostics
  - Paired statistical tests (Wilcoxon signed-rank & paired t-test)
  - Stratified cohort analysis (Shared TP, Shared FN, Balanced-Won, Baseline-Won, Shared TN, Shared FP)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
import tensorflow as tf

from src.experiment_contract import CONTRACT_VERSION, file_sha256
from src.faithfulness import compute_deletion_insertion_curves, run_model_parameter_randomization_check
from src.model import build_custom_cnn
from src.occlusion import OcclusionSensitivity
from src.openhsd_data import _load_and_preprocess_image
from src.xai_engine import XAIEngine, get_pre_sigmoid_logit


def compute_explanation_stability(
    engine: XAIEngine,
    img_tensor: np.ndarray,
    method: str = "gradcam",
    target_idx: int = 1,
    noise_sigma: float = 0.05,
    num_trials: int = 3,
    seed: int = 42,
) -> float:
    """Compute stability (Spearman rank correlation) under controlled input perturbations."""
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
    for t in range(num_trials):
        noise = rng.normal(0, noise_sigma, img_tensor.shape).astype(np.float32)
        perturbed_img = np.clip(img_tensor + noise, 0.0, 1.0)
        pert_map = get_map(perturbed_img).flatten()
        if np.std(pert_map) > 1e-7:
            rho, _ = spearmanr(base_map, pert_map)
            if not np.isnan(rho):
                correlations.append(float(rho))

    return float(np.mean(correlations)) if correlations else 0.0


def run_xai_evaluation(
    checkpoint_path: Path,
    output_dir: Path,
    sample_records: list[dict],
    steps: int = 10,
    dry_run: bool = False,
) -> dict:
    """Run single-model XAI evaluation across methods and qualitative quadrants (for backward compatibility)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"WORKSTREAM C: XAI FAITHFULNESS & STABILITY (Dry-Run: {dry_run})")
    print("=" * 80)

    if dry_run or not checkpoint_path.exists():
        print("[DRY-RUN] Using initialized model instance to verify XAI evaluation interface...")
        model = build_custom_cnn(input_shape=(224, 224, 3))
    else:
        print(f"Loading verified model checkpoint from {checkpoint_path}...")
        model = tf.keras.models.load_model(str(checkpoint_path), compile=False)

    engine = XAIEngine(model, positive_class_is_1=True)
    methods = ["gradcam", "layercam", "saliency", "random"]

    faithfulness_records = []
    stability_records = []

    print(f"\nEvaluating {len(sample_records)} representative samples across {methods}...")
    for item in sample_records:
        quadrant = item["quadrant"]
        img_arr = item["tensor"]
        img_name = item.get("filename", "sample")

        curves = compute_deletion_insertion_curves(
            model=model,
            engine=engine,
            img_tensor=img_arr,
            target_idx=1,
            steps=steps,
            methods=methods,
        )

        for m in methods:
            faithfulness_records.append({
                "filename": img_name,
                "quadrant": quadrant,
                "method": m,
                "deletion_auc": curves[m]["deletion_auc"],
                "insertion_auc": curves[m]["insertion_auc"],
            })

            if m != "random":
                stab_rho = compute_explanation_stability(
                    engine=engine,
                    img_tensor=img_arr,
                    method=m,
                    target_idx=1,
                )
                stability_records.append({
                    "filename": img_name,
                    "quadrant": quadrant,
                    "method": m,
                    "stability_spearman_rho": stab_rho,
                })

    summary = {}
    for m in methods:
        del_aucs = [r["deletion_auc"] for r in faithfulness_records if r["method"] == m]
        ins_aucs = [r["insertion_auc"] for r in faithfulness_records if r["method"] == m]
        summary[m] = {
            "mean_deletion_auc": round(float(np.mean(del_aucs)), 4),
            "mean_insertion_auc": round(float(np.mean(ins_aucs)), 4),
        }
        stabs = [r["stability_spearman_rho"] for r in stability_records if r["method"] == m]
        if stabs:
            summary[m]["mean_stability_rho"] = round(float(np.mean(stabs)), 4)

    faith_csv = output_dir / "faithfulness_metrics.csv"
    with open(faith_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "quadrant", "method", "deletion_auc", "insertion_auc"])
        writer.writeheader()
        writer.writerows(faithfulness_records)

    stab_csv = output_dir / "stability_metrics.csv"
    with open(stab_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "quadrant", "method", "stability_spearman_rho"])
        writer.writeheader()
        writer.writerows(stability_records)

    eval_manifest = {
        "mode": "dry_run" if dry_run else "production",
        "contract_version": CONTRACT_VERSION,
        "checkpoint": str(checkpoint_path) if checkpoint_path.exists() else "uninitialized_mock",
        "methods_evaluated": methods,
        "quadrants_represented": list(set(r["quadrant"] for r in sample_records)),
        "samples_evaluated": len(sample_records),
        "method_summary": summary,
        "status": "PASS_DRY_RUN" if dry_run else "SUCCESS",
    }

    out_json = output_dir / "xai_evaluation_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_manifest, f, indent=2)

    print(f"\nXAI Evaluation completed. Outputs saved to {output_dir}")
    print(f"Summary: {summary}")
    return eval_manifest


def run_xai_comparison(
    baseline_checkpoint: Path,
    balanced_checkpoint: Path,
    output_dir: Path,
    data_dir: Path = Path("/home/peskybird/Projects/data/openhsd/via-3"),
    imbalance_dir: Path = Path("results/via3_experiments/imbalance"),
    class_weighted_checkpoint: Optional[Path] = None,
    methods: Optional[List[str]] = None,
    samples_per_cohort: int = 10,
    steps: int = 10,
    seed: int = 42,
) -> dict:
    """Run full comparative XAI evaluation between baseline, balanced-sampling, and class-weighted models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("WORKSTREAM C: COMPARATIVE XAI FAITHFULNESS & STABILITY BENCHMARK")
    print("=" * 80)

    # 1. Preflight Verification
    assert baseline_checkpoint.is_file(), f"Baseline checkpoint not found: {baseline_checkpoint}"
    assert balanced_checkpoint.is_file(), f"Balanced checkpoint not found: {balanced_checkpoint}"

    base_sha = file_sha256(baseline_checkpoint)
    bal_sha = file_sha256(balanced_checkpoint)
    cw_sha = file_sha256(class_weighted_checkpoint) if class_weighted_checkpoint and class_weighted_checkpoint.is_file() else None

    print(f"Baseline Checkpoint:         {baseline_checkpoint} (SHA: {base_sha[:16]}...)")
    print(f"Balanced Sampling Checkpoint:{balanced_checkpoint} (SHA: {bal_sha[:16]}...)")
    if cw_sha:
        print(f"Class-Weighted Checkpoint:   {class_weighted_checkpoint} (SHA: {cw_sha[:16]}...)")

    # Load prediction CSVs from B2
    base_pred_csv = imbalance_dir / "baseline_clean_test_predictions.csv"
    bal_pred_csv = imbalance_dir / "balanced_sampling_clean_test_predictions.csv"
    assert base_pred_csv.is_file(), f"Missing B2 baseline test predictions: {base_pred_csv}"
    assert bal_pred_csv.is_file(), f"Missing B2 balanced test predictions: {bal_pred_csv}"

    df_base = pd.read_csv(base_pred_csv)
    df_bal = pd.read_csv(bal_pred_csv)
    assert len(df_base) == len(df_bal) == 12803, "Clean test set size mismatch!"

    # Frozen validation operating thresholds
    tau_star_base = 0.08
    tau_star_bal = 0.76

    y_true = df_base["ground_truth"].values
    pred_base = (df_base["hs_probability"].values >= tau_star_base).astype(int)
    pred_bal = (df_bal["hs_probability"].values >= tau_star_bal).astype(int)
    filenames = df_base["filename"].values
    test_img_dir = data_dir / "test"

    # Construct stratified cohorts
    rng = np.random.default_rng(seed)
    cohort_indices = {
        "shared_tp": np.where((y_true == 1) & (pred_base == 1) & (pred_bal == 1))[0],
        "shared_fn": np.where((y_true == 1) & (pred_base == 0) & (pred_bal == 0))[0],
        "balanced_won": np.where((y_true == 1) & (pred_base == 0) & (pred_bal == 1))[0],
        "baseline_won": np.where((y_true == 1) & (pred_base == 1) & (pred_bal == 0))[0],
        "shared_tn": np.where((y_true == 0) & (pred_base == 0) & (pred_bal == 0))[0],
        "shared_fp": np.where((y_true == 0) & (pred_base == 1) & (pred_bal == 1))[0],
    }

    selected_samples = []
    sample_manifest = []

    for c_name, idx_pool in cohort_indices.items():
        k = min(samples_per_cohort if ("tp" in c_name or "fn" in c_name or "won" in c_name) else 5, len(idx_pool))
        chosen_indices = rng.choice(idx_pool, size=k, replace=False)
        print(f"Cohort '{c_name}': {len(idx_pool)} available -> sampled {k}")
        for idx in chosen_indices:
            fn = filenames[idx]
            img_path = test_img_dir / fn
            assert img_path.is_file(), f"Image not found: {img_path}"
            tensor_hwc = _load_and_preprocess_image(str(img_path))
            tensor = np.expand_dims(tensor_hwc, axis=0)
            record = {
                "cohort": c_name,
                "filename": fn,
                "index": int(idx),
                "ground_truth": int(y_true[idx]),
                "baseline_prob": float(df_base["hs_probability"].values[idx]),
                "balanced_prob": float(df_bal["hs_probability"].values[idx]),
                "tensor": tensor,
                "raw_rgb": (tensor_hwc * 255.0).astype(np.uint8),
            }
            selected_samples.append(record)
            sample_manifest.append({
                "cohort": c_name,
                "filename": fn,
                "index": int(idx),
                "ground_truth": int(y_true[idx]),
                "baseline_prob": float(df_base["hs_probability"].values[idx]),
                "balanced_prob": float(df_bal["hs_probability"].values[idx]),
            })

    manifest_out = output_dir / "xai_sample_cohort_manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(sample_manifest, f, indent=2)
    print(f"\nCohort sample manifest saved to {manifest_out} ({len(selected_samples)} samples total)")

    # 2. Load Models and Build Engines
    print("\nLoading models and constructing XAI engines...")
    m_base = tf.keras.models.load_model(str(baseline_checkpoint), compile=False)
    m_bal = tf.keras.models.load_model(str(balanced_checkpoint), compile=False)
    engine_base = XAIEngine(m_base, positive_class_is_1=True, threshold=tau_star_base)
    engine_bal = XAIEngine(m_bal, positive_class_is_1=True, threshold=tau_star_bal)

    models_dict = {
        "baseline": {"model": m_base, "engine": engine_base, "tau": tau_star_base, "ckpt": str(baseline_checkpoint)},
        "balanced_sampling": {"model": m_bal, "engine": engine_bal, "tau": tau_star_bal, "ckpt": str(balanced_checkpoint)},
    }

    if class_weighted_checkpoint and class_weighted_checkpoint.is_file():
        m_cw = tf.keras.models.load_model(str(class_weighted_checkpoint), compile=False)
        engine_cw = XAIEngine(m_cw, positive_class_is_1=True, threshold=0.50)
        models_dict["class_weighted"] = {"model": m_cw, "engine": engine_cw, "tau": 0.50, "ckpt": str(class_weighted_checkpoint)}

    eval_methods = methods or ["gradcam", "gradcam_plus_plus", "layercam", "saliency", "occlusion", "random"]
    print(f"Attribution methods to evaluate: {eval_methods}")

    # 3. Comparative Evaluation Loop
    faithfulness_rows = []
    stability_rows = []
    collapse_rows = []

    print(f"\nEvaluating explanations across {len(models_dict)} models and {len(selected_samples)} samples...")
    t0 = time.time()

    for idx_s, sample in enumerate(selected_samples):
        fn = sample["filename"]
        cohort = sample["cohort"]
        tensor = sample["tensor"]
        gt = sample["ground_truth"]

        if (idx_s + 1) % 10 == 0 or idx_s == 0:
            print(f"  Sample {idx_s + 1}/{len(selected_samples)}: {fn} ({cohort})")

        for m_name, m_cfg in models_dict.items():
            model = m_cfg["model"]
            engine = m_cfg["engine"]

            # Compute faithfulness deletion/insertion curves
            curves = compute_deletion_insertion_curves(
                model=model,
                engine=engine,
                img_tensor=tensor,
                target_idx=1,  # Target Hotspot logit
                steps=steps,
                methods=eval_methods,
            )

            for meth in eval_methods:
                del_auc = curves[meth]["deletion_auc"]
                ins_auc = curves[meth]["insertion_auc"]
                faithfulness_rows.append({
                    "model": m_name,
                    "filename": fn,
                    "cohort": cohort,
                    "ground_truth": gt,
                    "method": meth,
                    "deletion_auc": round(del_auc, 4),
                    "insertion_auc": round(ins_auc, 4),
                })

                # Stability
                if meth != "random":
                    stab_rho = compute_explanation_stability(
                        engine=engine,
                        img_tensor=tensor,
                        method=meth,
                        target_idx=1,
                    )
                    stability_rows.append({
                        "model": m_name,
                        "filename": fn,
                        "cohort": cohort,
                        "ground_truth": gt,
                        "method": meth,
                        "stability_spearman_rho": round(stab_rho, 4),
                    })

            # Check flat attribution / collapse on primary Grad-CAM
            _, cam_g, _ = engine.explain_gradcam(tensor, target_class=1)
            is_flat = bool(np.std(cam_g) < 1e-7 or np.all(cam_g == 0.0))
            collapse_rows.append({
                "model": m_name,
                "filename": fn,
                "cohort": cohort,
                "ground_truth": gt,
                "is_flat_cam": is_flat,
                "cam_std": float(np.std(cam_g)),
                "cam_max": float(np.max(cam_g)),
                "cam_mean": float(np.mean(cam_g)),
            })

    eval_duration = time.time() - t0
    print(f"Explanation evaluation completed in {eval_duration:.2f}s")

    # 4. Save CSV Records
    df_faith = pd.DataFrame(faithfulness_rows)
    df_stab = pd.DataFrame(stability_rows)
    df_coll = pd.DataFrame(collapse_rows)

    faith_csv = output_dir / "faithfulness_metrics.csv"
    stab_csv = output_dir / "stability_metrics.csv"
    coll_csv = output_dir / "collapse_diagnostics.csv"

    df_faith.to_csv(faith_csv, index=False)
    df_stab.to_csv(stab_csv, index=False)
    df_coll.to_csv(coll_csv, index=False)

    # 5. Statistical Hypothesis Testing: Baseline vs. Balanced Sampling (Paired on identical samples)
    stat_comparisons = {}
    for meth in eval_methods:
        sub_base = df_faith[(df_faith["model"] == "baseline") & (df_faith["method"] == meth)].sort_values("filename")
        sub_bal = df_faith[(df_faith["model"] == "balanced_sampling") & (df_faith["method"] == meth)].sort_values("filename")

        del_base = sub_base["deletion_auc"].values
        del_bal = sub_bal["deletion_auc"].values
        ins_base = sub_base["insertion_auc"].values
        ins_bal = sub_bal["insertion_auc"].values

        delta_del = del_bal - del_base
        delta_ins = ins_bal - ins_base

        # Wilcoxon test for paired differences
        try:
            w_del = wilcoxon(del_base, del_bal)
            p_del_w = float(w_del.pvalue)
        except Exception:
            p_del_w = 1.0

        try:
            w_ins = wilcoxon(ins_base, ins_bal)
            p_ins_w = float(w_ins.pvalue)
        except Exception:
            p_ins_w = 1.0

        # Paired t-test
        t_del = ttest_rel(del_bal, del_base)
        t_ins = ttest_rel(ins_bal, ins_base)

        meth_stat = {
            "mean_del_auc_baseline": round(float(np.mean(del_base)), 4),
            "mean_del_auc_balanced": round(float(np.mean(del_bal)), 4),
            "mean_delta_del_auc": round(float(np.mean(delta_del)), 4),
            "wilcoxon_p_del": round(p_del_w, 4),
            "paired_t_p_del": round(float(t_del.pvalue), 4),
            "mean_ins_auc_baseline": round(float(np.mean(ins_base)), 4),
            "mean_ins_auc_balanced": round(float(np.mean(ins_bal)), 4),
            "mean_delta_ins_auc": round(float(np.mean(delta_ins)), 4),
            "wilcoxon_p_ins": round(p_ins_w, 4),
            "paired_t_p_ins": round(float(t_ins.pvalue), 4),
        }

        if meth != "random":
            s_base = df_stab[(df_stab["model"] == "baseline") & (df_stab["method"] == meth)].sort_values("filename")["stability_spearman_rho"].values
            s_bal = df_stab[(df_stab["model"] == "balanced_sampling") & (df_stab["method"] == meth)].sort_values("filename")["stability_spearman_rho"].values
            delta_s = s_bal - s_base
            try:
                w_s = wilcoxon(s_base, s_bal)
                p_s_w = float(w_s.pvalue)
            except Exception:
                p_s_w = 1.0
            t_s = ttest_rel(s_bal, s_base)
            meth_stat.update({
                "mean_stability_baseline": round(float(np.mean(s_base)), 4),
                "mean_stability_balanced": round(float(np.mean(s_bal)), 4),
                "mean_delta_stability": round(float(np.mean(delta_s)), 4),
                "wilcoxon_p_stability": round(p_s_w, 4),
                "paired_t_p_stability": round(float(t_s.pvalue), 4),
            })

        stat_comparisons[meth] = meth_stat

    # Collapse rates across models
    collapse_summary = {}
    for m_name in models_dict:
        sub_c = df_coll[df_coll["model"] == m_name]
        flat_rate = float(sub_c["is_flat_cam"].mean())
        mean_std = float(sub_c["cam_std"].mean())
        collapse_summary[m_name] = {
            "flat_cam_rate": round(flat_rate, 4),
            "mean_cam_spatial_std": round(mean_std, 4),
            "num_flat_samples": int(sub_c["is_flat_cam"].sum()),
            "total_evaluated": len(sub_c),
        }

    # 6. Adebayo Model Parameter Randomization Check (5 samples)
    print("\nExecuting Adebayo parameter randomization check on 5 curated samples...")
    adebayo_records = []
    for s in selected_samples[:5]:
        fn = s["filename"]
        tensor = s["tensor"]
        # Baseline model check
        rand_m = tf.keras.models.clone_model(m_base)
        for layer in rand_m.layers:
            if hasattr(layer, "kernel_initializer"):
                layer.set_weights([np.random.normal(0, 0.05, w.shape) for w in layer.get_weights()])
        rand_eng = XAIEngine(rand_m, positive_class_is_1=True)

        _, orig_cam, _ = engine_base.explain_gradcam(tensor, target_class=1)
        _, rand_cam, _ = rand_eng.explain_gradcam(tensor, target_class=1)

        rho, _ = spearmanr(orig_cam.flatten(), rand_cam.flatten())
        adebayo_records.append({
            "filename": fn,
            "spearman_rho_vs_randomized": round(float(rho) if not np.isnan(rho) else 0.0, 4),
        })

    adebayo_mean_rho = float(np.mean([r["spearman_rho_vs_randomized"] for r in adebayo_records]))
    print(f"Adebayo check mean rank correlation vs randomized model: {adebayo_mean_rho:.4f}")

    # 7. Generate Representative Heatmap Figures
    print("\nGenerating representative qualitative comparison figures...")
    sample_subsets = [
        ("shared_tp", "Shared True Positive"),
        ("balanced_won", "Balanced-Won Discordant (HS captured only by Balanced)"),
        ("baseline_won", "Baseline-Won Discordant (HS captured only by Baseline)"),
        ("shared_fn", "Shared False Negative (HS missed by both)"),
    ]

    for cohort_tag, cohort_title in sample_subsets:
        reps = [s for s in selected_samples if s["cohort"] == cohort_tag]
        if not reps:
            continue
        rep = reps[0]
        fn = rep["filename"]
        tensor = rep["tensor"]
        raw_rgb = rep["raw_rgb"]

        # Explain with Grad-CAM and Occlusion
        _, cam_base, _ = engine_base.explain_gradcam(tensor, target_class=1)
        _, cam_bal, _ = engine_bal.explain_gradcam(tensor, target_class=1)

        occ_base = OcclusionSensitivity(engine_base, window_size=(32, 32), stride=(16, 16))
        res_occ_base = occ_base.explain(tensor, target_class=1)
        occ_map_base = res_occ_base["norm_map"]

        occ_bal = OcclusionSensitivity(engine_bal, window_size=(32, 32), stride=(16, 16))
        res_occ_bal = occ_bal.explain(tensor, target_class=1)
        occ_map_bal = res_occ_bal["norm_map"]

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle(f"{cohort_title}\nFile: {fn} (Base P(HS): {rep['baseline_prob']:.4f}, Bal P(HS): {rep['balanced_prob']:.4f})", fontsize=11)

        axes[0].imshow(raw_rgb)
        axes[0].set_title("Input Layout (224x224)")
        axes[0].axis("off")

        axes[1].imshow(raw_rgb)
        axes[1].imshow(cam_base, cmap="jet", alpha=0.45)
        axes[1].set_title(f"Baseline Grad-CAM\n(P(HS)={rep['baseline_prob']:.3f})")
        axes[1].axis("off")

        axes[2].imshow(raw_rgb)
        axes[2].imshow(cam_bal, cmap="jet", alpha=0.45)
        axes[2].set_title(f"Balanced Grad-CAM\n(P(HS)={rep['balanced_prob']:.3f})")
        axes[2].axis("off")

        axes[3].imshow(raw_rgb)
        axes[3].imshow(occ_map_base, cmap="jet", alpha=0.45)
        axes[3].set_title("Baseline Occlusion (32x32)")
        axes[3].axis("off")

        axes[4].imshow(raw_rgb)
        axes[4].imshow(occ_map_bal, cmap="jet", alpha=0.45)
        axes[4].set_title("Balanced Occlusion (32x32)")
        axes[4].axis("off")

        plt.tight_layout()
        fig_path = figures_dir / f"xai_comparison_{cohort_tag}_{fn}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)

    print(f"Comparison figures saved to {figures_dir}")

    # 8. Produce Final Comparison Summary JSON
    comparison_summary = {
        "contract_version": CONTRACT_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples_evaluated": len(selected_samples),
        "cohort_breakdown": {c: len([s for s in selected_samples if s["cohort"] == c]) for c in cohort_indices},
        "methods_evaluated": eval_methods,
        "models_evaluated": list(models_dict.keys()),
        "thresholds_applied": {
            "baseline": tau_star_base,
            "balanced_sampling": tau_star_bal,
            "class_weighted": 0.50 if "class_weighted" in models_dict else None,
        },
        "collapse_diagnostics": collapse_summary,
        "adebayo_sanity_check": {
            "mean_spearman_rho_vs_randomized": round(adebayo_mean_rho, 4),
            "passed_sanity_check": bool(abs(adebayo_mean_rho) < 0.20),
        },
        "statistical_comparison_baseline_vs_balanced": stat_comparisons,
        "status": "SUCCESS",
    }

    summary_out = output_dir / "xai_comparison_summary.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    print(f"\nComparative XAI Benchmark completed successfully! Summary: {summary_out}")
    return comparison_summary


def main():
    parser = argparse.ArgumentParser(description="Workstream C XAI Evaluation Runner")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/via3/via3_baseline_seed42.keras"))
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path("models/via3/via3_baseline_seed42.keras"))
    parser.add_argument("--balanced-checkpoint", type=Path, default=Path("models/via3_imbalance/via3_balanced_sampling_seed42.keras"))
    parser.add_argument("--class-weighted-checkpoint", type=Path, default=Path("models/via3_imbalance/via3_class_weighted_seed42.keras"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/via3_experiments/xai_comparison"))
    parser.add_argument("--data-dir", type=Path, default=Path("/home/peskybird/Projects/data/openhsd/via-3"))
    parser.add_argument("--imbalance-dir", type=Path, default=Path("results/via3_experiments/imbalance"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        # Standard synthetic dry run for unit test compatibility
        rng = np.random.default_rng(42)
        sample_records = [
            {"filename": "fixture_tp.png", "quadrant": "TP", "tensor": rng.uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)},
            {"filename": "fixture_fn.png", "quadrant": "FN", "tensor": rng.uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)},
            {"filename": "fixture_tn.png", "quadrant": "TN", "tensor": rng.uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)},
            {"filename": "fixture_fp.png", "quadrant": "FP", "tensor": rng.uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)},
        ]
        run_xai_evaluation(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            sample_records=sample_records,
            steps=args.steps,
            dry_run=True,
        )
    else:
        run_xai_comparison(
            baseline_checkpoint=args.baseline_checkpoint,
            balanced_checkpoint=args.balanced_checkpoint,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            imbalance_dir=args.imbalance_dir,
            class_weighted_checkpoint=args.class_weighted_checkpoint,
            steps=args.steps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
