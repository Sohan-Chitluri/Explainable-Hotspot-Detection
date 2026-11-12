# Three-Way Train / Val / Test Split — Reproducibility Record

**Status:** Implemented (2026-09-20)
**Fixes:** [`docs/TEST_VALIDATION_LEAKAGE_AUDIT.md`](TEST_VALIDATION_LEAKAGE_AUDIT.md) — Task A HIGH severity
**Function:** `src/data.py::make_three_way_split`

---

## 1. Motivation

Prior to this change, every training script mapped `validation_dir` to the official
`iccad{N}/test/` folder (see `src/data.py::dataset_analysis`, line 16 before the fix).
The consequences were:

| Script | Severity |
|--------|----------|
| `train_xai.py` | **HIGH** — `BestBalancedAccuracyCheckpoint` selected the saved checkpoint using test-set `val_*` metrics; reported "final" metrics came from the same files |
| `train_baseline.py` | **LOW-MODERATE** — no callback, but per-epoch test curves were visible; `BENCHMARK_EPOCHS` differ per benchmark (suggestive of human tuning) |
| `run_iccad5_imbalance_ablation.py` | **HIGH** — same as `train_xai.py`; still unmitigated (flagged) |
| `run_iccad5_seed_replication.py` | **HIGH** — same as above; still unmitigated (flagged) |

---

## 2. Split Design

### 2.1 Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `val_frac` | **0.15** (15 %) | Within the audit-recommended 15–20 % range; uniform across all 5 benchmarks for auditability |
| `seed` | **42** | Matches `tf.keras.utils.set_random_seed(42)` in `main()` of both training scripts |
| Stratification | **Per-class** | HS and NHS shuffled and split independently so the HS:NHS ratio is preserved in both train and val subsets |
| Method | **`flow_from_dataframe` with numpy index split** | Produces an auditable file list (manifest JSON); no files moved on disk |

### 2.2 Algorithm (pseudocode)

```python
rng = np.random.default_rng(seed)

for class_files in [sorted(hs_files), sorted(nhs_files)]:
    rng.shuffle(idx := np.arange(len(class_files)))
    n_val = max(1, round(len(class_files) * val_frac))
    val_files   = class_files[idx[:n_val]]   # first n_val after shuffle
    train_files = class_files[idx[n_val:]]   # remainder
```

The `sorted()` before `rng.shuffle()` ensures a deterministic starting order that is
independent of filesystem enumeration order, making the split reproducible across
operating systems and Python versions.

### 2.3 Per-benchmark counts

| Benchmark | train_hs (orig) | train_nhs (orig) | val_hs | val_nhs | rem_train_hs | rem_train_nhs |
|-----------|:---------------:|:----------------:|:------:|:-------:|:------------:|:-------------:|
| iccad1    | 99              | 340              | 15     | 51      | 84           | 289           |
| iccad2    | 174             | 5,285            | 26     | 793     | 148          | 4,492         |
| iccad3    | 909             | 4,643            | 136    | 696     | 773          | 3,947         |
| iccad4    | 95              | 4,452            | 14     | 668     | 81           | 3,784         |
| **iccad5**| **26**          | 2,716            | **4**  | 407     | **22**       | 2,309         |

*Counts derived from `round(n * 0.15)` with `max(1, ...)` floor; actual values in each
`split_manifest.json` are authoritative.*

### 2.4 iccad5 limitation

iccad5 has only **26 HS training images** total.  At 15 %, the val carve-out contains
**4 HS images**.  Implications:

- Validation **sensitivity** can only take values in {0, 0.25, 0.50, 0.75, 1.0} — the
  granularity is inherently coarse.
- `BestBalancedAccuracyCheckpoint` will observe noisy balanced-accuracy signals for iccad5;
  the best epoch may be selected by a single HS prediction flipping.
- This is a **dataset limitation**, not a split design deficiency.  Reducing `val_frac`
  further (e.g., to 10 % → 3 val HS) saves only one training sample while worsening the
  signal.  Increasing `val_frac` (e.g., to 20 % → 5 val HS) would leave only 21 training
  HS samples.  15 % is the least-bad choice.
- The `split_manifest.json` for iccad5 includes an `"iccad5_warning"` field documenting
  this limitation.

---

## 3. Implementation

`src/data.py::make_three_way_split` returns a `ThreeWaySplit` named tuple:

```
ThreeWaySplit(
    train_gen,       # flow_from_dataframe — 85% of iccad{N}/train/
    val_gen,         # flow_from_dataframe — 15% carve-out of iccad{N}/train/
    test_gen,        # flow_from_directory — iccad{N}/test/ UNTOUCHED
    counts,          # raw counts from original train + test directories
    split_manifest,  # JSON-serialisable dict — written to split_manifest.json
)
```

**Class label mapping** (consistent with old `flow_from_directory` alphabetical ordering):

| Label (string) | Generator integer class | Meaning |
|----------------|------------------------|---------|
| `"train_hs"`   | 0 (positive)           | Hotspot |
| `"train_nhs"`  | 1 (negative)           | Non-hotspot |

The `evaluate_xai_model` / `evaluate_generator` helpers look up the HS label as
`class_indices.get("test_hs", 0)` — this still works because `test_gen` uses
`flow_from_directory` on `iccad{N}/test/` with `test_hs` / `test_nhs` subdirectories,
so `test_gen.class_indices["test_hs"] == 0` as before.  The train/val generator class
names (`train_hs`, `train_nhs`) are distinct and only accessed by their integer index.

---

## 4. Artefacts produced per benchmark

For each benchmark, the following files are written alongside existing training artefacts:

```
results/{baseline|xai_training}/iccad{N}/
    split_manifest.json      ← exact filenames in val carve-out, per class
    metrics.json             ← updated checkpoint_selection.val_set = "carve-out"
    balanced_accuracy_vs_epoch.png  ← curves now show train vs val CARVE-OUT
```

The `split_manifest.json` structure:

```json
{
  "benchmark": "iccad5",
  "split_design": {
    "val_frac": 0.15,
    "seed": 42,
    "stratified_per_class": true,
    "method": "flow_from_dataframe with numpy-seeded index split"
  },
  "train_subset": {"n_hs": 22, "n_nhs": 2309, "total": 2331},
  "val_subset": {
    "n_hs": 4,
    "n_nhs": 407,
    "total": 411,
    "hs_filenames": ["img_03.png", ...],
    "nhs_filenames": ["img_001.png", ...]
  },
  "test_set": {
    "n_hs": 41, "n_nhs": 19327, "total": 19368,
    "note": "Official ICCAD test split; read exactly once after training..."
  },
  "iccad5_warning": "iccad5 has only 26 HS training images. At val_frac=0.15, val_hs=4..."
}
```

---

## 5. Residual leakage (NOT yet fixed)

The following scripts still use the old `dataset_analysis` / `data_extractor` pair and
therefore still point `validation_data` at `iccad5/test/`.  Each has a
`TODO(leakage-fix)` block in its docstring:

| Script | Severity | Next action |
|--------|----------|-------------|
| `src/run_iccad5_imbalance_ablation.py` | HIGH | Migrate to `make_three_way_split` |
| `src/run_iccad5_seed_replication.py` | HIGH | Migrate to `make_three_way_split` (thread `seed` through split) |

Results currently in `results/xai/iccad5_imbalance_ablation/` and
`results/xai/iccad5_seed_replication/` are checkpoint-selection-biased and must be
re-run after those scripts are migrated.

---

## 6. How to regenerate or verify the split

```bash
# From repo root with .venv activated:
python - <<'EOF'
import json
from pathlib import Path
from src.data import make_three_way_split

for n in range(1, 6):
    split = make_three_way_split(
        Path("iccad-official") / f"iccad{n}",
        f"iccad{n}",
        val_frac=0.15,
        seed=42,
    )
    print(json.dumps(split.split_manifest["val_subset"], indent=2))
EOF
```

The printed `hs_filenames` / `nhs_filenames` lists must match those in
`results/.../split_manifest.json` for any training run to be considered reproducible.

---

## 7. Automated tests

```bash
source .venv/bin/activate
python -m pytest tests/test_three_way_split.py -v
```

Tests cover: no-overlap, correct counts, full coverage, reproducibility, seed
sensitivity, test-gen isolation, class-index consistency, iccad5 edge case (4 val HS),
minimum-1-val clamping, and missing-dir error.

---

*Related documents:*
- [`TEST_VALIDATION_LEAKAGE_AUDIT.md`](TEST_VALIDATION_LEAKAGE_AUDIT.md) — original audit
- [`ICCAD5_CHECKPOINT_FIX.md`](ICCAD5_CHECKPOINT_FIX.md) — motivation for best-epoch selection
- [`ICCAD5_IMBALANCE_ABLATION.md`](ICCAD5_IMBALANCE_ABLATION.md) — pending re-run after migration
- [`ICCAD5_SEED_REPLICATION.md`](ICCAD5_SEED_REPLICATION.md) — pending re-run after migration
