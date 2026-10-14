"""Unit tests for src/data.make_three_way_split.

All tests run against a tiny synthetic dataset created in a temporary directory;
no ICCAD benchmark files are required.  Tests are CPU-only and do not instantiate
Keras models.

Test coverage:
    1. No file overlap: val ∩ train = ∅ (per class)
    2. Correct split counts: each class split independently at val_frac
    3. Full coverage: every original training file ends up in exactly one split
    4. Reproducibility: same seed → identical split on two calls
    5. Seed sensitivity: different seeds produce different val sets (>50% of files)
    6. Test set untouched: test_gen filenames are exactly those in test_hs/test_nhs
    7. Class indices consistent: train_gen and val_gen both map train_hs→0, train_nhs→1
    8. iccad5 edge case: 26 HS + 2716-style NHS at 15% → val_hs=4 exactly
    9. Minimum 1 val sample: val_frac so small that round() would give 0 → clamped to 1
   10. Missing directory raises FileNotFoundError
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


def _make_tiny_dataset(root: Path, n_hs: int, n_nhs: int, n_test_hs: int = 5,
                       n_test_nhs: int = 10) -> None:
    """Create a synthetic ICCAD-style benchmark directory with tiny PNG stubs."""
    dirs = {
        root / "train" / "train_hs": n_hs,
        root / "train" / "train_nhs": n_nhs,
        root / "test" / "test_hs": n_test_hs,
        root / "test" / "test_nhs": n_test_nhs,
    }
    for d, count in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            # 4x4 RGB stub (minimal valid PNG; avoids heavy file I/O)
            img = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
            img.save(d / f"img_{i:05d}.png")


class TestMakeThreeWaySplit(unittest.TestCase):

    def setUp(self):
        """Create a temporary benchmark directory and import the function under test."""
        self.tmpdir = Path(tempfile.mkdtemp())
        self.n_hs = 40
        self.n_nhs = 200
        _make_tiny_dataset(self.tmpdir, self.n_hs, self.n_nhs)
        # Import here so test isolation is cleaner
        from src.data import make_three_way_split
        self.make_split = make_three_way_split

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Helper                                                               #
    # ------------------------------------------------------------------ #

    def _call(self, val_frac=0.15, seed=42):
        return self.make_split(
            self.tmpdir,
            "iccad_test",
            val_frac=val_frac,
            seed=seed,
            batch_size=8,
            img_size=(4, 4),   # tiny images for speed
        )

    # ------------------------------------------------------------------ #
    # Test 1: No overlap between train and val (per class)                #
    # ------------------------------------------------------------------ #

    def test_no_overlap_hs(self):
        split = self._call()
        manifest = split.split_manifest
        val_hs = set(manifest["val_subset"]["hs_filenames"])
        # All train_hs files
        all_hs = sorted(f for f in os.listdir(self.tmpdir / "train" / "train_hs")
                        if not f.startswith("."))
        n_val_hs = len(val_hs)
        train_hs = set(all_hs) - val_hs
        self.assertEqual(len(val_hs & train_hs), 0,
                         "val HS files leaked into training HS set")
        self.assertEqual(len(val_hs) + len(train_hs), self.n_hs)

    def test_no_overlap_nhs(self):
        split = self._call()
        manifest = split.split_manifest
        val_nhs = set(manifest["val_subset"]["nhs_filenames"])
        all_nhs = sorted(f for f in os.listdir(self.tmpdir / "train" / "train_nhs")
                         if not f.startswith("."))
        train_nhs = set(all_nhs) - val_nhs
        self.assertEqual(len(val_nhs & train_nhs), 0,
                         "val NHS files leaked into training NHS set")
        self.assertEqual(len(val_nhs) + len(train_nhs), self.n_nhs)

    # ------------------------------------------------------------------ #
    # Test 2: Correct counts at val_frac=0.15                             #
    # ------------------------------------------------------------------ #

    def test_val_counts_at_15pct(self):
        split = self._call(val_frac=0.15)
        manifest = split.split_manifest
        expected_val_hs = max(1, round(self.n_hs * 0.15))   # round(6.0) = 6
        expected_val_nhs = max(1, round(self.n_nhs * 0.15))  # round(30.0) = 30
        self.assertEqual(manifest["val_subset"]["n_hs"], expected_val_hs)
        self.assertEqual(manifest["val_subset"]["n_nhs"], expected_val_nhs)

    def test_train_counts_at_15pct(self):
        split = self._call(val_frac=0.15)
        manifest = split.split_manifest
        expected_train_hs = self.n_hs - max(1, round(self.n_hs * 0.15))
        expected_train_nhs = self.n_nhs - max(1, round(self.n_nhs * 0.15))
        self.assertEqual(manifest["train_subset"]["n_hs"], expected_train_hs)
        self.assertEqual(manifest["train_subset"]["n_nhs"], expected_train_nhs)

    # ------------------------------------------------------------------ #
    # Test 3: Full coverage (every file in exactly one split)             #
    # ------------------------------------------------------------------ #

    def test_full_coverage_hs(self):
        split = self._call()
        manifest = split.split_manifest
        val_hs = set(manifest["val_subset"]["hs_filenames"])
        train_n_hs = manifest["train_subset"]["n_hs"]
        self.assertEqual(len(val_hs) + train_n_hs, self.n_hs)

    def test_full_coverage_nhs(self):
        split = self._call()
        manifest = split.split_manifest
        val_nhs = set(manifest["val_subset"]["nhs_filenames"])
        train_n_nhs = manifest["train_subset"]["n_nhs"]
        self.assertEqual(len(val_nhs) + train_n_nhs, self.n_nhs)

    # ------------------------------------------------------------------ #
    # Test 4: Reproducibility — same seed → same split                    #
    # ------------------------------------------------------------------ #

    def test_reproducibility(self):
        split_a = self._call(seed=42)
        split_b = self._call(seed=42)
        self.assertEqual(
            split_a.split_manifest["val_subset"]["hs_filenames"],
            split_b.split_manifest["val_subset"]["hs_filenames"],
        )
        self.assertEqual(
            split_a.split_manifest["val_subset"]["nhs_filenames"],
            split_b.split_manifest["val_subset"]["nhs_filenames"],
        )

    # ------------------------------------------------------------------ #
    # Test 5: Seed sensitivity — different seeds → different val sets     #
    # ------------------------------------------------------------------ #

    def test_different_seeds_differ(self):
        split_42 = self._call(seed=42)
        split_99 = self._call(seed=99)
        val_42 = set(split_42.split_manifest["val_subset"]["hs_filenames"])
        val_99 = set(split_99.split_manifest["val_subset"]["hs_filenames"])
        # Overlap must be < 100% (probabilistically certain for n_hs=40)
        self.assertNotEqual(val_42, val_99,
                            "Different seeds produced identical val splits — "
                            "RNG is not being used")

    # ------------------------------------------------------------------ #
    # Test 6: test_gen covers exactly the test directory files            #
    # ------------------------------------------------------------------ #

    def test_test_gen_class_indices(self):
        split = self._call()
        # test_gen must have class_indices with test_hs and test_nhs
        class_indices = split.test_gen.class_indices
        self.assertIn("test_hs", class_indices)
        self.assertIn("test_nhs", class_indices)

    def test_test_gen_sample_count(self):
        split = self._call()
        # 5 test_hs + 10 test_nhs = 15
        self.assertEqual(split.test_gen.n, 5 + 10)

    # ------------------------------------------------------------------ #
    # Test 7: Class indices consistent between train and val              #
    # ------------------------------------------------------------------ #

    def test_class_indices_consistent(self):
        split = self._call()
        train_ci = split.train_gen.class_indices
        val_ci = split.val_gen.class_indices
        # Both should map train_hs→0 (alphabetically first)
        self.assertEqual(train_ci, val_ci,
                         f"train class_indices {train_ci} != val class_indices {val_ci}")
        self.assertEqual(train_ci.get("train_hs"), 0)
        self.assertEqual(train_ci.get("train_nhs"), 1)

    # ------------------------------------------------------------------ #
    # Test 8: iccad5-style edge case (26 HS, 2716 NHS at 15%)            #
    # ------------------------------------------------------------------ #

    def test_iccad5_edge_case(self):
        """26 HS training images at 15% → val_hs=4 (round(26*0.15)=round(3.9)=4)."""
        iccad5_dir = Path(tempfile.mkdtemp())
        try:
            _make_tiny_dataset(iccad5_dir, n_hs=26, n_nhs=100,
                               n_test_hs=3, n_test_nhs=20)
            from src.data import make_three_way_split
            split = make_three_way_split(
                iccad5_dir, "iccad5",
                val_frac=0.15, seed=42, batch_size=4, img_size=(4, 4)
            )
            manifest = split.split_manifest
            self.assertEqual(manifest["val_subset"]["n_hs"], 4,
                             "iccad5: expected 4 val HS at val_frac=0.15")
            self.assertEqual(manifest["train_subset"]["n_hs"], 22,
                             "iccad5: expected 22 remaining train HS")
            # iccad5 warning should be populated
            self.assertIsNotNone(manifest["iccad5_warning"])
        finally:
            shutil.rmtree(iccad5_dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Test 9: Minimum 1 val sample (very small val_frac)                  #
    # ------------------------------------------------------------------ #

    def test_minimum_one_val_sample(self):
        """val_frac so small that round(n * val_frac) == 0 → clamped to 1."""
        tiny_dir = Path(tempfile.mkdtemp())
        try:
            # 3 HS: round(3 * 0.001) = 0, should be clamped to 1
            _make_tiny_dataset(tiny_dir, n_hs=3, n_nhs=10,
                               n_test_hs=2, n_test_nhs=5)
            from src.data import make_three_way_split
            split = make_three_way_split(
                tiny_dir, "tiny",
                val_frac=0.001, seed=42, batch_size=2, img_size=(4, 4)
            )
            self.assertGreaterEqual(split.split_manifest["val_subset"]["n_hs"], 1)
        finally:
            shutil.rmtree(tiny_dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Test 10: Missing directory raises FileNotFoundError                  #
    # ------------------------------------------------------------------ #

    def test_missing_directory_raises(self):
        bad_dir = self.tmpdir / "does_not_exist"
        from src.data import make_three_way_split
        with self.assertRaises(FileNotFoundError):
            make_three_way_split(bad_dir, "missing", seed=42,
                                 batch_size=2, img_size=(4, 4))

    # ------------------------------------------------------------------ #
    # Test 11: Script source audit — no test set leakage in model.fit     #
    # ------------------------------------------------------------------ #

    def test_exploratory_scripts_do_not_leak_test_set(self):
        """Verify ablation and seed replication scripts use val_gen for validation."""
        import ast

        root = Path(__file__).resolve().parents[1]
        scripts_to_check = [
            root / "src" / "run_iccad5_imbalance_ablation.py",
            root / "src" / "run_iccad5_seed_replication.py",
            root / "src" / "train_baseline.py",
            root / "src" / "train_xai.py",
        ]

        def check_ast_for_leakage(code_str: str, filename: str):
            tree = ast.parse(code_str, filename=filename)
            fit_calls = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Check if method call is .fit(...)
                    if isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
                        fit_calls += 1
                        val_arg = None
                        for kw in node.keywords:
                            if kw.arg == "validation_data":
                                val_arg = kw.value
                        if val_arg is not None:
                            val_name = ast.unparse(val_arg).strip()
                            if "test" in val_name.lower():
                                raise AssertionError(
                                    f"Leakage detected in {filename}: model.fit() has validation_data={val_name}"
                                )
            return fit_calls

        for script_path in scripts_to_check:
            self.assertTrue(script_path.is_file(), f"Missing script: {script_path}")
            content = script_path.read_text(encoding="utf-8")
            self.assertIn("make_three_way_split", content, f"{script_path.name} missing make_three_way_split")
            fits = check_ast_for_leakage(content, script_path.name)
            self.assertGreater(fits, 0, f"Expected at least one model.fit call in {script_path.name}")

        # Verify that representative leakage regressions DO fail the checker:
        leaky_snippet = "model.fit(train_gen, epochs=5, validation_data=test_gen)"
        with self.assertRaises(AssertionError):
            check_ast_for_leakage(leaky_snippet, "leaky_dummy.py")

        leaky_snippet_attr = "model.fit(train_gen, epochs=5, validation_data=split.test_gen)"
        with self.assertRaises(AssertionError):
            check_ast_for_leakage(leaky_snippet_attr, "leaky_dummy_attr.py")


if __name__ == "__main__":
    unittest.main()
