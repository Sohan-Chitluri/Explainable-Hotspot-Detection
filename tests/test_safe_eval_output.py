"""
Regression tests for safe output directory creation in research scripts.
"""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from scripts.evaluate_iccad5_research_xai import OUT_XAI_CSV, OUT_FAITH_CSV


def test_evaluate_iccad5_output_paths_configured():
    """Assert output CSV paths in evaluate_iccad5_research_xai are Path objects."""
    assert isinstance(OUT_XAI_CSV, Path)
    assert isinstance(OUT_FAITH_CSV, Path)
    assert OUT_XAI_CSV.name == "xai_collapse_evaluation.csv"
    assert OUT_FAITH_CSV.name == "faithfulness_comparison.csv"


def test_safe_directory_creation_pattern(tmp_path):
    """Verify that writing DataFrame to a deeply nested non-existent directory succeeds with parent.mkdir."""
    nested_dir = tmp_path / "deeply" / "nested" / "results"
    out_file = nested_dir / "test_eval.csv"

    # Ensure parent does not exist yet
    assert not nested_dir.exists()

    # The safe pattern used in evaluate_iccad5_research_xai.py
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"sample": "test1", "score": 0.95}])
    df.to_csv(out_file, index=False)

    assert out_file.exists()
    read_df = pd.read_csv(out_file)
    assert len(read_df) == 1
    assert read_df.iloc[0]["sample"] == "test1"
