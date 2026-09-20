# Test/Validation Leakage Audit

**Audit date:** 2026-09-20
**Scope:** All training entry points that call `src/data.py::data_extractor` / `dataset_analysis`.
**Status: CONFIRMED — validation_dir is the official ICCAD test split, and it is used for checkpoint selection, not just monitoring.**

---

## 1. Claim

> "validation_dir is mapped to the official ICCAD test set" — if true, model
> selection (checkpoint picking, early stopping) was implicitly tuned against
> test data, invalidating reported accuracy numbers.

**Verdict: CONFIRMED**, with two distinct severities depending on which training script is examined (see §3).

---

## 2. Trace: where `validation_dir` is defined and consumed

### 2.1 Definition — `src/data.py:10-16`

```python
def dataset_analysis(folder: str):
    base_dir = os.path.join(os.path.dirname(folder))
    train_dir = os.path.join(base_dir, "train")
    validation_dir = os.path.join(base_dir, "test")   # <-- literally the ICCAD test/ folder
```

`validation_dir` is not a carve-out of `train/`; it is a direct alias for `iccad-official/iccad{N}/test/`, which is the same physical directory tree (`test_hs/`, `test_nhs/`) that every "final evaluation" call in the repo also reads from. There is no third directory, no `validation_split=` argument, and no `train_test_split` call anywhere in `src/` (confirmed by repo-wide grep — zero hits for `validation_split`, `train_test_split`, `holdout`, `dev_dir`).

### 2.2 Generator construction — `src/data.py:54-75`

```python
def data_extractor(train_dir, validation_dir, batch_size=32):
    ...
    val_data_gen = image_gen_val.flow_from_directory(
        directory=validation_dir,   # == iccad{N}/test/
        shuffle=False,
        target_size=(224, 224),
        class_mode="binary",
    )
    return train_data_gen, val_data_gen
```

`val_data_gen` reads the test split. This same generator object is used two ways in every training script:

1. Passed as `validation_data=val_data_gen` to `model.fit(...)` — Keras evaluates it every epoch and logs `val_*` metrics.
2. Passed again, unchanged, to a post-training "final evaluation" function (`evaluate_generator` / `evaluate_xai_model`) that computes the numbers written to `metrics.json` and reported in `README.md`/`BASELINE.md`.

**These are the same file set.** The "held-out test evaluation" and the "per-epoch validation" are drawing from an identical directory with `shuffle=False`, so they are the same 19,368 (iccad5), 32,067 (iccad4), etc. images every time.

### 2.3 Consumers

| File | Uses `val_data_gen` for `model.fit(validation_data=...)`? | Uses a callback that reads `val_*` logs to change what gets saved? |
|---|---|---|
| `src/train_baseline.py:170-174` | Yes | **No** — no callbacks passed to `.fit()` at all |
| `src/train_xai.py:198-204` | Yes | **Yes** — `BestBalancedAccuracyCheckpoint` (see §3.2) |
| `src/run_iccad5_seed_replication.py:107-138` | Yes | **Yes** — same `BestBalancedAccuracyCheckpoint` |
| `src/run_iccad5_imbalance_ablation.py:96-127` | Yes | **Yes** — same `BestBalancedAccuracyCheckpoint` |

---

## 3. Severity breakdown

### 3.1 `train_baseline.py` — monitoring/observation leakage only (LOW-MODERATE)

`train_one_benchmark` (`src/train_baseline.py:147-174`) calls `model.fit(train_data_gen, epochs=epochs, validation_data=val_data_gen)` with **no callbacks**. There is no `EarlyStopping`, no `ModelCheckpoint`, no `ReduceLROnPlateau`. The model that gets saved (`model.save(model_path)`, line 221) is unconditionally whatever weights exist after the last configured epoch (5 or 10, per `BENCHMARK_EPOCHS`).

- Per-epoch `val_*` metrics computed on the test set **are** logged to Keras `history` and plotted (`plot_balanced_accuracy_curves`), and a `best_history_val_balanced_accuracy` scalar is written into `metrics.json` — but that scalar is never used programmatically to pick or restore any weights. It is descriptive only.
- Leakage here is at the **human/experiment-design level**: whoever set `BENCHMARK_EPOCHS = {1: 5, 2: 5, 3: 10, 4: 10, 5: 10}` could have looked at test-set epoch curves (the notebook this reproduces displays exactly such curves) when choosing those counts, and the differing epoch counts by benchmark are suggestive of that. This is not provably automated tuning against the test set, but the opportunity and a plausible motive both exist, and it cannot be ruled out from the code alone.
- **No automated checkpoint selection or early stopping occurred for the baseline models.** The reported baseline numbers reflect a fixed, pre-declared epoch budget, not a test-optimized stopping point.

### 3.2 `train_xai.py` and all ICCAD5 improvement scripts — direct checkpoint-selection leakage (HIGH)

`src/metrics.py:116-158`, class `BestBalancedAccuracyCheckpoint`:

```python
def on_epoch_end(self, epoch, logs=None):
    tp = logs.get("val_true_positives")   # computed on val_data_gen == test set
    tn = logs.get("val_true_negatives")
    fp = logs.get("val_false_positives")
    fn = logs.get("val_false_negatives")
    ...
    bal_acc = 0.5 * (sensitivity + specificity)
    if bal_acc > self.best_value:
        self.best_value = bal_acc
        self.best_epoch = epoch
        self.best_weights = [w.copy() for w in self.model.get_weights()]

def on_train_end(self, logs=None):
    if self.best_weights is not None:
        self.model.set_weights(self.best_weights)   # restores best-on-test-set weights
```

`src/train_xai.py:197-204` wires this directly into `.fit(..., callbacks=[best_ckpt_cb])`, and the unchanged `model.save(model_path)` call afterward persists **the epoch whose weights scored highest on the test set**, not the final epoch. `final_metrics["checkpoint_selection"]` (lines 238-245) documents this explicitly and transparently in the saved JSON — the project's own metadata admits the selection criterion is "best validation balanced accuracy... from val_true_positives/... " computed on `val_data_gen`.

This is **textbook checkpoint-selection leakage**: the model that gets saved and then evaluated for the "final" reported metrics (`evaluate_xai_model`, same generator, same files) was chosen specifically because it scored well on that exact evaluation set. The reported test accuracy for every XAI-CNN model (all 5 production checkpoints in `models/xai/xai_cnn_iccad{1..5}.keras`, plus every ICCAD5 experimental/retrained variant) is optimistically biased by this selection step — the magnitude of the bias is not quantified anywhere in the repo (no true held-out set exists to measure it against).

The project's own `docs/ICCAD5_CHECKPOINT_FIX.md` independently arrived at half of this finding — it explicitly proposes and implements "best-epoch checkpoint selection by validation balanced accuracy" as an *improvement*, without recognizing that the "validation" set it selects against is the same test set the improvement is then benchmarked on (§7 of that doc reports "OLD vs NEW TEST BALANCED ACCURACY" using the identical generator). That document treats this as a training-quality fix; from a train/test-separation standpoint it is actually the introduction of a more severe, better-documented form of the same leakage that `train_baseline.py` only risked informally.

`src/run_iccad5_seed_replication.py` and `src/run_iccad5_imbalance_ablation.py` both import and use the identical `BestBalancedAccuracyCheckpoint` against the identical `val_data_gen` (`iccad5/test/`), so every seed/imbalance-ablation result in `results/xai/iccad5_seed_replication/` and `results/xai/iccad5_imbalance_ablation/` carries the same leakage.

### 3.3 Was it used for hyperparameter tuning?

No evidence of automated hyperparameter search (grid/random search, Keras Tuner, etc.) was found anywhere in `src/`. `BENCHMARK_EPOCHS`, `BATCH_SIZE`, `OPTIMIZER`, learning rate (Nadam default, unspecified), and architecture are all hardcoded constants shared across benchmarks, not selected per-benchmark by a search loop. The only "tuning" mechanism found is the per-epoch checkpoint selection in §3.2, plus the informal epoch-count-per-benchmark choice noted in §3.1.

### 3.4 Summary table

| Mechanism | Uses test set as validation? | Confirmed automated selection against it? | Affected artifacts |
|---|:---:|:---:|---|
| Early stopping | N/A — not implemented anywhere in the repo | — | — |
| Checkpoint/best-epoch selection | Yes | **Yes** (`train_xai.py` + both ICCAD5 improvement scripts) | All 5 `models/xai/xai_cnn_iccad*.keras`, `models/xai/retrained/*`, `models/xai/experimental/*`, every metrics.json/CSV derived from them, all downstream XAI/Grad-CAM/occlusion/geometry analyses built on those checkpoints |
| Hyperparameter tuning | Per-epoch `val_*` visible during training | Not automated; epoch-count-per-benchmark choice is suggestive but not provably test-tuned | `models/custom_cnn_iccad*.keras` (baseline) |
| Final "held-out" evaluation | Yes — identical directory/generator as above | N/A (this is the reporting step, not a selection step) | Every reported balanced accuracy / F1 / recall / specificity number in `README.md`, `BASELINE.md`, and all `docs/*` reports |

---

## 4. Which reported numbers need re-running on a clean split

**Everything downstream of a `BestBalancedAccuracyCheckpoint`-selected model is affected**, specifically:

- `models/xai/xai_cnn_iccad{1..5}.keras` (production XAI models) and every metric quoted for them in `README.md`'s benchmark table, `results/xai_training/`.
- `models/xai/retrained/xai_cnn_iccad5_best.keras` and all of `docs/ICCAD5_CHECKPOINT_FIX.md`.
- `models/xai/experimental/*` (balanced-sampling and no-weight seed variants) and all of `docs/IMPROVEMENT_BALANCED_REPLICATION.md`, `docs/IMPROVEMENT_WEIGHTED_REPLICATION.md`, `docs/ICCAD5_SEED_REPLICATION.md`, `docs/ICCAD5_IMBALANCE_ABLATION.md`.
- Every Grad-CAM/Grad-CAM++/LayerCAM/occlusion/geometry finding computed from any of the above checkpoints inherits the same bias one level removed (the *attribution maps* are conditioned on a model chosen partly for scoring well on the very images being explained).

**Not directly affected by checkpoint-selection leakage** (but still evaluated on the test set, per §3.1):

- `models/custom_cnn_iccad{1..5}.keras` (baseline CNN) — no selection callback, final-epoch weights only. Baseline numbers in `BASELINE.md` are not checkpoint-selection-biased, though they still lack a true held-out set for the epoch-count design choice (§3.1).

This project already tracks a related, narrower concern in `docs/IMPROVEMENT_RESEARCH_STATE.md`:

> "src/data.py:15–16 maps validation_dir to the official test directory... Best-checkpoint evaluation on that same partition would create test-selection bias; A/G/H are tracing all consumers before approving replication."

That note flags the mechanism but (per the task registry) is still marked as an open/pending audit item (task **A**), not yet resolved — this document supersedes it with a confirmed trace and severity breakdown.

---

## 5. Ambiguities / things this audit does NOT establish

- **Magnitude of the bias is not measured.** No experiment in the repo compares a `BestBalancedAccuracyCheckpoint`-selected model's performance on the test set vs. a genuinely unseen third split. The direction of the bias (optimistic) is certain; the size is not quantified.
- **Whether `BENCHMARK_EPOCHS` was itself chosen by watching test curves** cannot be confirmed or ruled out from the code/git history alone — flagging as ambiguous rather than assuming either way.
- **GPU is currently unavailable** (per `docs/IMPROVEMENT_RESEARCH_STATE.md`, `nvidia-smi` exit 6 as of 2026-09-17), so no new training runs (with or without a fix) could be executed as part of this audit. This audit is static code/doc analysis only.
- Whether any *baseline* epoch-count decisions were carried over unchanged from the original notebook (`LHD_CustomModel.ipynb`) — which would reduce the "tuned against this repo's test set" concern for the baseline specifically — was not verified; the notebook itself is not present in this repository to check.

---

## 6. Concrete fix

1. **Introduce a real three-way split.** From each `iccad{N}/train/` directory, carve out a stratified held-out validation subset (e.g., 15-20% of `train_hs`/`train_nhs`, fixed by a seeded shuffle) *before* any training begins. Keep `iccad{N}/test/` completely untouched and unused until the very last, one-time reporting step.
   - Minimal-diff approach: add a `val_dir` (separate from both `train_dir` and the ICCAD `test_dir`) produced by `ImageDataGenerator(...).flow_from_directory(..., subset="validation")` with `validation_split=0.15-0.2` set on the *train* generator only, or by physically splitting file lists with a fixed seed and building three separate generators.
2. **Point `BestBalancedAccuracyCheckpoint` and `model.fit(validation_data=...)` at the new held-out validation split, not `test/`.**
3. **Reserve `test/` (`val_data_gen` as currently defined) exclusively for the final, single, post-hoc evaluation call** (`evaluate_generator` / `evaluate_xai_model`), run once after the checkpoint is already fixed by the validation split — never inside the training loop or any callback.
4. **Re-run and re-report**, at minimum: all XAI-CNN production checkpoints (`train_xai.py` for all 5 benchmarks) and the ICCAD5 seed-replication / imbalance-ablation studies, since those are the artifacts identified in §4 as checkpoint-selection-biased. Baseline CNN numbers do not require a checkpoint-selection re-run but would benefit from the same 3-way split for the epoch-count decision to be defensible.
5. **Do not fix by simply removing `BestBalancedAccuracyCheckpoint`** — reverting to final-epoch-only saving (the baseline's approach) avoids checkpoint-selection leakage but reintroduces the non-monotonic-training instability problem that motivated the callback in the first place (`docs/ICCAD5_CHECKPOINT_FIX.md` §2). The correct fix is a proper validation split, not removing checkpoint selection.

No training code has been modified as part of this audit.
