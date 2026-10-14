"""
Automated unit & regression tests asserting numerical consistency across:
- Primary repository artifacts (metrics.json, CSVs, manifests)
- Master research report (docs/FINAL_RESEARCH_REPORT.md)
- LaTeX manuscript (latex_report/document.tex)
"""

import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).parent.parent


def test_table_i_raw_dataset_counts():
    """Verify Table I raw file counts and imbalance ratios against iccad-official/."""
    expected_counts = {
        "iccad1": {"tr_hs": 99, "tr_nhs": 340, "te_hs": 226, "te_nhs": 4679},
        "iccad2": {"tr_hs": 174, "tr_nhs": 5285, "te_hs": 498, "te_nhs": 41298},
        "iccad3": {"tr_hs": 909, "tr_nhs": 4643, "te_hs": 1808, "te_nhs": 46333},
        "iccad4": {"tr_hs": 95, "tr_nhs": 4452, "te_hs": 177, "te_nhs": 31890},
        "iccad5": {"tr_hs": 26, "tr_nhs": 2716, "te_hs": 41, "te_nhs": 19327},
    }

    for b, counts in expected_counts.items():
        tr_hs = len(os.listdir(REPO_ROOT / f"iccad-official/{b}/train/train_hs"))
        tr_nhs = len(os.listdir(REPO_ROOT / f"iccad-official/{b}/train/train_nhs"))
        te_hs = len(os.listdir(REPO_ROOT / f"iccad-official/{b}/test/test_hs"))
        te_nhs = len(os.listdir(REPO_ROOT / f"iccad-official/{b}/test/test_nhs"))

        assert tr_hs == counts["tr_hs"], f"{b} train HS count mismatch"
        assert tr_nhs == counts["tr_nhs"], f"{b} train NHS count mismatch"
        assert te_hs == counts["te_hs"], f"{b} test HS count mismatch"
        assert te_nhs == counts["te_nhs"], f"{b} test NHS count mismatch"

        tr_ratio = tr_nhs / tr_hs
        te_ratio = te_nhs / te_hs
        assert abs(tr_ratio - (counts["tr_nhs"] / counts["tr_hs"])) < 1e-4
        assert abs(te_ratio - (counts["te_nhs"] / counts["te_hs"])) < 1e-4


def test_table_iv_baseline_vs_xai_recomputation():
    """Recompute Precision, Recall, Specificity, F1, Balanced Accuracy from confusion matrices."""
    base_file = REPO_ROOT / "results/baseline/metrics.json"
    xai_file = REPO_ROOT / "results/xai_training/retrained_clean/metrics.json"

    assert base_file.exists()
    assert xai_file.exists()

    with open(base_file) as f:
        base_data = json.load(f)
    with open(xai_file) as f:
        xai_data = json.load(f)

    benchmarks = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]

    for b in benchmarks:
        b_cm = base_data["per_benchmark"][b]["confusion_matrix"]
        b_prec = b_cm["tp"] / (b_cm["tp"] + b_cm["fp"])
        b_rec = b_cm["tp"] / (b_cm["tp"] + b_cm["fn"])
        b_spec = b_cm["tn"] / (b_cm["tn"] + b_cm["fp"])
        b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec)
        b_ba = (b_rec + b_spec) / 2.0

        assert abs(b_prec - base_data["per_benchmark"][b]["precision"]) < 1e-4
        assert abs(b_rec - base_data["per_benchmark"][b]["recall_sensitivity"]) < 1e-4
        assert abs(b_f1 - base_data["per_benchmark"][b]["f1_score"]) < 1e-4
        assert abs(b_ba - base_data["per_benchmark"][b]["balanced_accuracy"]) < 1e-4

        x_cm = xai_data["per_benchmark_xai"][b]["confusion_matrix"]
        x_prec = x_cm["tp"] / (x_cm["tp"] + x_cm["fp"])
        x_rec = x_cm["tp"] / (x_cm["tp"] + x_cm["fn"])
        x_spec = x_cm["tn"] / (x_cm["tn"] + x_cm["fp"])
        x_f1 = 2 * x_prec * x_rec / (x_prec + x_rec)
        x_ba = (x_rec + x_spec) / 2.0

        assert abs(x_prec - xai_data["per_benchmark_xai"][b]["precision"]) < 1e-4
        assert abs(x_rec - xai_data["per_benchmark_xai"][b]["recall_sensitivity"]) < 1e-4
        assert abs(x_f1 - xai_data["per_benchmark_xai"][b]["f1_score"]) < 1e-4
        assert abs(x_ba - xai_data["per_benchmark_xai"][b]["balanced_accuracy"]) < 1e-4


def test_table_v_leakage_deltas():
    """Verify Table V leaked-vs-clean metrics and deltas."""
    leak_file = REPO_ROOT / "results/xai_training/leaky_vs_clean_comparison.json"
    with open(leak_file) as f:
        leak_data = json.load(f)

    comp = leak_data["comparison_table"]
    expected_deltas = {
        "iccad1": -0.0192,
        "iccad2": 0.0526,
        "iccad3": -0.0132,
        "iccad4": 0.0104,
        "iccad5": -0.0532,
    }

    for b, delta_exp in expected_deltas.items():
        delta_real = comp[b]["delta"]
        assert abs(delta_real - delta_exp) < 0.001


def test_table_vii_smoke_test_harmonic_mean():
    """Verify exact arithmetic of Table VII confusion matrix."""
    tp, tn, fp, fn = 233, 4743, 1, 23
    total = tp + tn + fp + fn
    assert total == 5000

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    spec = tn / (tn + fp)
    ba = (rec + spec) / 2.0
    f1 = 2 * prec * rec / (prec + rec)

    assert abs(prec - 0.995726) < 1e-4
    assert abs(rec - 0.910156) < 1e-4
    assert abs(spec - 0.999789) < 1e-4
    assert abs(ba - 0.954973) < 1e-4
    assert abs(f1 - 0.951020) < 1e-4


def test_table_viii_iccad5_ablation_collapse_rates():
    """Verify ICCAD5 class-imbalance ablation results and zero-vector rates."""
    classif_file = REPO_ROOT / "results/xai/iccad5_imbalance_ablation/classification_comparison.csv"
    zerovec_file = REPO_ROOT / "results/xai/iccad5_imbalance_ablation/zero_vector_comparison.csv"

    df_c = pd.read_csv(classif_file)
    df_z = pd.read_csv(zerovec_file)

    # Class-weighted HS collapse is 35/41 = 85.366%
    cw_hs = df_z[(df_z["condition"] == "class_weighted") & (df_z["group"] == "HS")].iloc[0]
    assert cw_hs["zero_vector_count"] == 35
    assert cw_hs["n"] == 41
    assert abs(cw_hs["zero_vector_pct"] - 85.366) < 0.01

    # Balanced sampling has 0 zero vectors
    bs_all = df_z[(df_z["condition"] == "balanced_sampling") & (df_z["group"] == "ALL")].iloc[0]
    assert bs_all["zero_vector_count"] == 0
    assert bs_all["zero_vector_pct"] == 0.0


def test_table_x_faithfulness_auc_values():
    """Verify Table X deletion and insertion AUC against stored results."""
    faith_file = REPO_ROOT / "results/xai_faithfulness/faithfulness_and_stability_results.json"
    with open(faith_file) as f:
        data = json.load(f)["faithfulness"]

    expected = {
        "iccad1": {"gradcam": (0.536, 0.562), "gradcam_plus_plus": (0.541, 0.528), "layercam": (0.521, 0.580)},
        "iccad2": {"gradcam": (0.117, 0.536), "gradcam_plus_plus": (0.214, 0.136), "layercam": (0.065, 0.606)},
        "iccad3": {"gradcam": (0.349, 0.826), "gradcam_plus_plus": (0.482, 0.842), "layercam": (0.197, 0.949)},
        "iccad4": {"gradcam": (0.342, 0.688), "gradcam_plus_plus": (0.421, 0.658), "layercam": (0.317, 0.946)},
        "iccad5": {"gradcam": (0.195, 0.271), "gradcam_plus_plus": (0.209, 0.264), "layercam": (0.228, 0.284)},
    }

    for b, methods in expected.items():
        summary = data[b]["summary"]
        for m, (exp_del, exp_ins) in methods.items():
            assert abs(summary[m]["mean_deletion_auc"] - exp_del) < 0.005
            assert abs(summary[m]["mean_insertion_auc"] - exp_ins) < 0.005


def test_manuscript_latex_text_consistency():
    """Verify latex_report/document.tex contains corrected numbers and no stale claims."""
    tex_path = REPO_ROOT / "latex_report/document.tex"
    assert tex_path.exists()
    content = tex_path.read_text()

    # Exact parameter count: 128,321
    assert "128,321" in content
    assert "25k" not in content

    # RGB 3-channel input
    assert "224\\times224\\times3" in content

    # Diagnostic smoke test label
    assert "ICCAD1 Synthetic" in content
    assert "diagnostic fixture was used solely for pipeline integration verification" in content

    # 6x expansion caveat on pooled ResNet
    assert "Exact $6\\times$ Expansion" in content
    assert "877,662" in content

    # 2D raster morphology qualification
    assert "2D binary raster morphology" in content
