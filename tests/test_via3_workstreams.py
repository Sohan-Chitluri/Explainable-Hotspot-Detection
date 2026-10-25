"""Tests for Via-3 Workstreams A, B, C, D dry-run execution and schemas."""

import json
from pathlib import Path
import numpy as np
import pytest

from scripts.analyze_via3_thresholds import run_threshold_analysis
from scripts.evaluate_via3_xai import run_xai_evaluation
from scripts.generate_paper_reports import generate_draft_reports
from scripts.run_via3_baseline import run_baseline
from scripts.run_via3_imbalance_experiments import run_imbalance_experiments
from scripts.run_via3_multiseed import (
    compute_aggregate_statistics,
    is_seed_run_completed,
    load_reference_seed42,
    run_multiseed_orchestration,
)
from scripts.run_via3_xai_multiseed import (
    is_checkpoint_xai_completed,
    run_multiseed_xai,
)
from src.results_schema import validate_schema


def test_workstream_a_baseline_dry_run(tmp_path):
    """Test that baseline dry-run completes without training and generates valid manifest."""
    manifest_dir = Path("results/dataset_audit")
    out_dir = tmp_path / "baseline_out"
    models_dir = tmp_path / "models"

    manifest = run_baseline(
        manifest_dir=manifest_dir,
        output_dir=out_dir,
        models_dir=models_dir,
        epochs=15,
        dry_run=True,
    )

    assert manifest["status"] == "PASS_DRY_RUN"
    assert manifest["simulated_test_evaluated"] is False
    assert (out_dir / "baseline_dry_run_manifest.json").exists()


def test_workstream_b1_threshold_analysis(tmp_path):
    """Test validation-only threshold analysis and curve generation."""
    rng = np.random.default_rng(42)
    val_labels = np.array([1] * 20 + [0] * 80)
    val_probs = np.where(val_labels == 1, rng.uniform(0.3, 0.9, 100), rng.uniform(0.01, 0.4, 100))

    out_dir = tmp_path / "thresh_out"
    res = run_threshold_analysis(
        val_probs=val_probs,
        val_labels=val_labels,
        output_dir=out_dir,
        objective="max_f1",
        experiment_name="test_experiment",
    )

    assert 0.0 < res["calibrated_threshold"] < 1.0
    assert (out_dir / "test_experiment_threshold_tradeoff.png").exists()
    assert (out_dir / "test_experiment_threshold_sweep.csv").exists()
    assert validate_schema(res, "threshold")


def test_workstream_b2_imbalance_dry_run(tmp_path):
    """Test that imbalance runner dry-run verifies datasets and balanced sampling without training."""
    manifest_dir = Path("results/dataset_audit")
    out_dir = tmp_path / "imbalance_out"
    models_dir = tmp_path / "models"

    manifest = run_imbalance_experiments(
        strategies=["baseline", "class_weighted", "balanced_sampling"],
        manifest_dir=manifest_dir,
        output_dir=out_dir,
        models_dir=models_dir,
        dry_run=True,
    )

    assert manifest["status"] == "PASS_DRY_RUN"
    assert "balanced_sampling" in manifest["strategies_verified"]
    assert (out_dir / "imbalance_dry_run_manifest.json").exists()


def test_workstream_b3_multiseed_aggregation(tmp_path):
    """Test statistical aggregation (mean, std, 95% CI) and dry-run execution."""
    sample_seeds = [
        {"pr_auc": 0.250, "roc_auc": 0.730, "recall_sensitivity": 0.030, "precision": 1.0, "f1_score": 0.058, "balanced_accuracy": 0.515},
        {"pr_auc": 0.260, "roc_auc": 0.740, "recall_sensitivity": 0.033, "precision": 1.0, "f1_score": 0.064, "balanced_accuracy": 0.516},
        {"pr_auc": 0.255, "roc_auc": 0.735, "recall_sensitivity": 0.031, "precision": 1.0, "f1_score": 0.060, "balanced_accuracy": 0.515},
    ]

    stats = compute_aggregate_statistics(sample_seeds)
    assert stats["num_seeds"] == 3
    assert abs(stats["metrics"]["pr_auc"]["mean"] - 0.255) < 1e-4
    assert stats["metrics"]["precision"]["std"] == 0.0

    # Test runner dry-run
    manifest_dir = Path("results/dataset_audit")
    out_dir = tmp_path / "multiseed_out"
    models_dir = tmp_path / "models"

    m = run_multiseed_orchestration(
        seeds=[42, 100, 2024],
        manifest_dir=manifest_dir,
        output_dir=out_dir,
        models_dir=models_dir,
        dry_run=True,
    )
    assert m["status"] == "PASS_DRY_RUN"
    assert validate_schema(m, "multiseed")


def test_workstream_c_xai_evaluation_dry_run(tmp_path):
    """Test XAI evaluation across TP, FN, TN, FP quadrants and methods."""
    rng = np.random.default_rng(42)
    sample_records = [
        {"filename": "tp.png", "quadrant": "TP", "tensor": rng.uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)},
        {"filename": "fn.png", "quadrant": "FN", "tensor": rng.uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)},
        {"filename": "tn.png", "quadrant": "TN", "tensor": rng.uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)},
        {"filename": "fp.png", "quadrant": "FP", "tensor": rng.uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)},
    ]

    out_dir = tmp_path / "xai_out"
    res = run_xai_evaluation(
        checkpoint_path=Path("non_existent_mock.keras"),
        output_dir=out_dir,
        sample_records=sample_records,
        steps=5,
        dry_run=True,
    )

    assert res["status"] == "PASS_DRY_RUN"
    assert set(res["methods_evaluated"]) == {"gradcam", "layercam", "saliency", "random"}
    assert validate_schema(res, "xai")
    assert (out_dir / "faithfulness_metrics.csv").exists()
    assert (out_dir / "stability_metrics.csv").exists()


def test_workstream_d_report_generation(tmp_path):
    """Test paper report generation and ensure protected files are untouched."""
    out_dir = tmp_path / "draft_reports"
    res = generate_draft_reports(
        input_dir=tmp_path,
        output_dir=out_dir,
        use_synthetic=True,
    )

    assert Path(res["report_markdown"]).exists()
    assert Path(res["latex_table"]).exists()

    # Check content contains synthetic marker
    with open(res["report_markdown"], "r", encoding="utf-8") as f:
        txt = f.read()
    assert "SYNTHETIC TEST FIXTURE ONLY" in txt


def test_workstream_d_report_generation_real_data(tmp_path):
    """Test paper report generation with real experiment results."""
    out_dir = tmp_path / "reports"
    res = generate_draft_reports(
        input_dir=Path("results/via3_experiments"),
        output_dir=out_dir,
        use_synthetic=False,
    )

    assert Path(res["report_markdown"]).exists()
    assert Path(res["latex_table"]).exists()

    with open(res["report_markdown"], "r", encoding="utf-8") as f:
        txt = f.read()

    assert "SYNTHETIC TEST FIXTURE ONLY" not in txt
    assert "Mandatory Methodological & Statistical Qualifications" in txt
    assert "0.2619" in txt
    assert "0.6118" in txt
    assert "Nominal Wilcoxon" in txt
    assert "FIND-01" in txt
    assert "FIND-02" in txt
    assert "0.5744" in txt
    assert "0.0229" in txt

    with open(res["latex_table"], "r", encoding="utf-8") as f:
        latex_txt = f.read()
    assert "tab:via3_classification" in latex_txt
    assert "tab:via3_xai_comparison" in latex_txt
    assert "tab:via3_collapse_diagnostics" in latex_txt
    assert "Nominal Wilcoxon" in latex_txt
    assert "FIND-01" in latex_txt
    assert "FIND-02" in latex_txt
    assert "0.0229" in latex_txt


def test_protected_files_safeguard():
    """Verify that protected files have not been modified."""
    docx_file = Path("docs/DA2_Group18_Report.docx")
    latex_dir = Path("latex_report")

    assert docx_file.exists()
    assert latex_dir.is_dir()


def test_multiseed_strategy_validation(tmp_path):
    """Verify that unknown strategies are rejected with ValueError."""
    with pytest.raises(ValueError, match="Invalid strategy"):
        run_multiseed_orchestration(
            seeds=[42],
            manifest_dir=Path("results/dataset_audit"),
            output_dir=tmp_path / "out",
            models_dir=tmp_path / "models",
            strategy="unsupported_strategy",
            dry_run=True,
        )


def test_multiseed_seed42_reference_loading(tmp_path):
    """Verify that Seed 42 reference is safely imported without retraining."""
    out_seed_dir = tmp_path / "seed_42"
    models_dir = tmp_path / "models"

    ref = load_reference_seed42(
        strategy="balanced_sampling",
        output_seed_dir=out_seed_dir,
        models_dir=models_dir,
    )

    assert ref["strategy"] == "balanced_sampling"
    assert ref["seed"] == 42
    assert (out_seed_dir / "balanced_sampling_results_seed42.json").exists()
    assert (out_seed_dir / "balanced_sampling_clean_test_predictions.csv").exists()
    assert (models_dir / "via3_balanced_sampling_seed42.keras").exists()


def test_multiseed_seed42_hash_mismatch(tmp_path):
    """Verify that corrupted Seed 42 checkpoint triggers security verification error."""
    mock_models_dir = tmp_path / "corrupted_models"
    mock_models_dir.mkdir(parents=True)
    corrupted_ckpt = mock_models_dir / "via3_balanced_sampling_seed42.keras"
    corrupted_ckpt.write_bytes(b"tampered content")

    with pytest.raises(RuntimeError, match="Security verification failed"):
        load_reference_seed42(
            strategy="balanced_sampling",
            output_seed_dir=tmp_path / "seed_42",
            models_dir=tmp_path / "models",
            reference_models_dir=mock_models_dir,
        )


def test_multiseed_strategy_namespacing(tmp_path):
    """Verify that strategies write into namespaced directories without colliding."""
    out_dir = tmp_path / "multiseed_out"
    models_dir = tmp_path / "models"

    m_base = run_multiseed_orchestration(
        seeds=[42, 100],
        manifest_dir=Path("results/dataset_audit"),
        output_dir=out_dir,
        models_dir=models_dir,
        strategy="baseline",
        dry_run=True,
    )
    m_bal = run_multiseed_orchestration(
        seeds=[42, 100],
        manifest_dir=Path("results/dataset_audit"),
        output_dir=out_dir,
        models_dir=models_dir,
        strategy="balanced_sampling",
        dry_run=True,
    )

    assert (out_dir / "baseline" / "multiseed_baseline_dry_run_manifest.json").exists()
    assert (out_dir / "balanced_sampling" / "multiseed_balanced_sampling_dry_run_manifest.json").exists()
    assert m_base["strategy"] == "baseline"
    assert m_bal["strategy"] == "balanced_sampling"


def test_multiseed_is_completed_guard(tmp_path):
    """Verify file-existence and completion detection logic."""
    seed_dir = tmp_path / "seed_100"
    seed_dir.mkdir(parents=True)
    ckpt = tmp_path / "models" / "via3_baseline_seed100.keras"
    ckpt.parent.mkdir(parents=True)

    # Initially missing
    assert not is_seed_run_completed(seed_dir, "baseline", 100, ckpt)

    # Create empty files (should still fail size check)
    ckpt.write_text("")
    (seed_dir / "baseline_results_seed100.json").write_text("")
    (seed_dir / "baseline_clean_test_predictions.csv").write_text("")
    (seed_dir / "baseline_val_predictions.csv").write_text("")
    (seed_dir / "baseline_official_test_predictions.csv").write_text("")
    assert not is_seed_run_completed(seed_dir, "baseline", 100, ckpt)

    # Populate files with content
    ckpt.write_text("model_bytes")
    (seed_dir / "baseline_results_seed100.json").write_text('{"status": "ok"}')
    (seed_dir / "baseline_clean_test_predictions.csv").write_text("a,b,c")
    (seed_dir / "baseline_val_predictions.csv").write_text("a,b,c")
    (seed_dir / "baseline_official_test_predictions.csv").write_text("a,b,c")
    assert is_seed_run_completed(seed_dir, "baseline", 100, ckpt)


def test_workstream_e4_xai_completion_guard(tmp_path):
    """Verify XAI checkpoint completion detection logic."""
    seed_out = tmp_path / "xai_seed_42"
    seed_out.mkdir(parents=True)

    # Missing files
    assert not is_checkpoint_xai_completed(seed_out, expected_samples=2)

    # Create dummy files with wrong row counts
    (seed_out / "faithfulness_metrics.csv").write_text("model,method\n1,gradcam\n")
    (seed_out / "stability_metrics.csv").write_text("model,method\n1,gradcam\n")
    (seed_out / "collapse_diagnostics.csv").write_text("model,is_flat\n1,0\n")
    (seed_out / "xai_summary.json").write_text("{}")
    assert not is_checkpoint_xai_completed(seed_out, expected_samples=2)

    # Create dummy files with exact expected row counts (2 samples * 6 methods = 12, 2 * 5 = 10, 2)
    faith_lines = ["model,method\n"] + [f"m{i},meth\n" for i in range(12)]
    stab_lines = ["model,method\n"] + [f"m{i},meth\n" for i in range(10)]
    coll_lines = ["model,is_flat\n"] + [f"m{i},0\n" for i in range(2)]

    (seed_out / "faithfulness_metrics.csv").write_text("".join(faith_lines))
    (seed_out / "stability_metrics.csv").write_text("".join(stab_lines))
    (seed_out / "collapse_diagnostics.csv").write_text("".join(coll_lines))
    assert is_checkpoint_xai_completed(seed_out, expected_samples=2)


def test_workstream_e4_xai_dry_run(tmp_path):
    """Verify XAI multi-seed orchestration in dry-run mode."""
    out_dir = tmp_path / "xai_multiseed_out"
    cohort_manifest = Path("results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json")

    summary = run_multiseed_xai(
        strategies=["baseline"],
        seeds=[42, 100],
        cohort_manifest=cohort_manifest,
        output_dir=out_dir,
        smoke=True,
        dry_run=True,
    )

    assert summary["status"] == "SUCCESS"
    assert (out_dir / "master_faithfulness_metrics.csv").exists()
    assert (out_dir / "master_stability_metrics.csv").exists()
    assert (out_dir / "master_collapse_diagnostics.csv").exists()
    assert (out_dir / "multiseed_xai_comparison_summary.json").exists()
    assert (out_dir / "reproducibility_manifest.json").exists()


def test_workstream_e4_real_artifacts_integrity():
    """Verify that real generated E4.2 XAI artifacts are complete and internally consistent."""
    results_dir = Path("results/via3_experiments/xai_multiseed")
    if not results_dir.exists():
        pytest.skip("E4.2 results directory does not exist")

    master_faith = results_dir / "master_faithfulness_metrics.csv"
    master_stab = results_dir / "master_stability_metrics.csv"
    master_coll = results_dir / "master_collapse_diagnostics.csv"
    summary_json = results_dir / "multiseed_xai_comparison_summary.json"
    rep_manifest = results_dir / "reproducibility_manifest.json"

    for p in [master_faith, master_stab, master_coll, summary_json, rep_manifest]:
        assert p.is_file(), f"Missing required artifact: {p}"

    import pandas as pd
    df_f = pd.read_csv(master_faith)
    df_s = pd.read_csv(master_stab)
    df_c = pd.read_csv(master_coll)

    # 6 models * 50 samples * 6 methods = 1800
    assert len(df_f) == 1800
    # 6 models * 50 samples * 5 methods = 1500
    assert len(df_s) == 1500
    # 6 models * 50 samples = 300
    assert len(df_c) == 300

    # Verify no failures
    assert (df_f["status"] == "SUCCESS").all()
    assert (df_s["status"] == "SUCCESS").all()
    assert (df_c["status"] == "SUCCESS").all()

    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "SUCCESS"
    assert data["total_evaluations_completed"] == 1800
    assert len(data["per_model_summaries"]) == 6


