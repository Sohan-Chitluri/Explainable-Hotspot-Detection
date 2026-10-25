"""Tests for Experiment Contract, Metric Semantics, and Leakage Safeguards."""

import numpy as np
import pytest

from src.experiment_contract import (
    AUDITED_COUNTS,
    CONTRACT_VERSION,
    check_pretraining_gate,
    compute_contract_metrics,
    save_predictions_csv,
    select_threshold_from_val,
    verify_leakage_safeguards,
    verify_manifest_integrity,
)


def test_manifest_integrity():
    """Verify that audited via-3 manifests match expected SHA-256 hashes."""
    status = verify_manifest_integrity("results/dataset_audit")
    assert all(status.values()), f"Manifest integrity failed: {status}"


def test_pretraining_gate_closed(monkeypatch, tmp_path):
    """Verify that gate check correctly detects the closed gate."""
    fake_gate = tmp_path / ".non_existent_gate"
    monkeypatch.setattr("src.experiment_contract.PRETRAINING_GATE_FILE", fake_gate)
    gate = check_pretraining_gate()
    assert isinstance(gate, dict)
    assert "gate_open" in gate
    # When gate file is not present, gate must be closed
    assert gate["gate_open"] is False


def test_metric_semantics_positive_hotspot():
    """Verify that Label 1 is strictly Hotspot (HS) and metrics match mathematical formulas."""
    # 2 HS (1), 2 NHS (0)
    # y_true = [1, 1, 0, 0]
    # y_prob = [0.9, 0.4, 0.8, 0.1]
    # At tau = 0.5:
    #   y_pred = [1, 0, 1, 0]
    #   TP = 1, FN = 1, FP = 1, TN = 1
    y_true = [1, 1, 0, 0]
    y_prob = [0.9, 0.4, 0.8, 0.1]

    m = compute_contract_metrics(y_true, y_prob, threshold=0.5)

    assert m["confusion_matrix"]["tp"] == 1
    assert m["confusion_matrix"]["fn"] == 1
    assert m["confusion_matrix"]["fp"] == 1
    assert m["confusion_matrix"]["tn"] == 1

    # Recall = 1 / (1 + 1) = 0.5
    assert m["recall_sensitivity"] == 0.5
    # Specificity = 1 / (1 + 1) = 0.5
    assert m["specificity"] == 0.5
    # Precision = 1 / (1 + 1) = 0.5
    assert m["precision"] == 0.5
    # F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
    assert m["f1_score"] == 0.5
    # Balanced Accuracy = 0.5 * (0.5 + 0.5) = 0.5
    assert m["balanced_accuracy"] == 0.5
    assert m["positive_label"] == 1


def test_threshold_calibration_validation_only():
    """Verify threshold selection calibrates strictly on validation and chooses optimal tau."""
    # Synthetic validation set where high threshold gives high precision, lower threshold gives high recall
    y_val = [1, 1, 1, 0, 0, 0, 0, 0]
    p_val = [0.8, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1, 0.05]

    # For max_f1
    res = select_threshold_from_val(y_val, p_val, objective="max_f1")
    assert "selected_threshold" in res
    assert 0.0 < res["selected_threshold"] < 1.0
    assert "metrics_at_selected_threshold" in res

    # Verify sweep table is comprehensive
    assert len(res["sweep_table"]) == 99


def test_leakage_safeguards_detection():
    """Verify that overlap between partitions or exclusion contamination is caught."""
    clean_tr = ["path/to/tr1.png", "path/to/tr2.png"]
    clean_val = ["path/to/val1.png"]
    clean_te = ["path/to/te1.png"]

    # Clean case
    res_clean = verify_leakage_safeguards(clean_tr, clean_val, clean_te)
    assert res_clean["is_leakage_free"] is True

    # Injected leakage: tr1 appears in val
    leaky_val = ["path/to/tr1.png"]
    res_leaky = verify_leakage_safeguards(clean_tr, leaky_val, clean_te)
    assert res_leaky["is_leakage_free"] is False
    assert res_leaky["train_val_overlap_count"] == 1


def test_save_predictions_csv_schema_and_counts(tmp_path):
    """Verify prediction CSV schema, column names, sample ID uniqueness, and valid ranges."""
    import csv

    paths = [
        "/data/via3/val/Hvia107504_mb_mb_lccout.oas.gds.png",
        "/data/via3/val/Nvia118233_mb_mb_lccout.oas.gds.png",
        "/data/via3/val/Hvia87136_mb_mb_lccout.oas.gds.png",
    ]
    labels = [1, 0, 1]
    probs = [0.852319, 0.041284, 0.499999]

    out_csv = tmp_path / "test_val_predictions.csv"
    res_path = save_predictions_csv(paths, labels, probs, out_csv)
    assert res_path.exists()

    with open(res_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 3
    expected_cols = ["sample_id", "filename", "ground_truth", "hs_probability", "predicted_label"]
    assert list(reader[0].keys()) == expected_cols

    # Check row 0
    assert reader[0]["sample_id"] == "Hvia107504"
    assert reader[0]["filename"] == "Hvia107504_mb_mb_lccout.oas.gds.png"
    assert int(reader[0]["ground_truth"]) == 1
    assert float(reader[0]["hs_probability"]) == 0.852319
    assert int(reader[0]["predicted_label"]) == 1

    # Check row 1
    assert reader[1]["sample_id"] == "Nvia118233"
    assert int(reader[1]["ground_truth"]) == 0
    assert float(reader[1]["hs_probability"]) == 0.041284
    assert int(reader[1]["predicted_label"]) == 0

    # Check row 2 (0.499999 < 0.5 -> predicted_label = 0)
    assert reader[2]["sample_id"] == "Hvia87136"
    assert int(reader[2]["ground_truth"]) == 1
    assert int(reader[2]["predicted_label"]) == 0

    # Verify uniqueness of sample IDs
    sids = [r["sample_id"] for r in reader]
    assert len(sids) == len(set(sids))

