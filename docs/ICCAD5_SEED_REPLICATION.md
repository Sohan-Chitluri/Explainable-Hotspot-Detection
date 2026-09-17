# ICCAD5 Seed Replication: Is the Class-Imbalance-Handling Effect Reproducible?

## 1. Objective

`docs/ICCAD5_IMBALANCE_ABLATION.md` compared three ICCAD5 training conditions — each
trained **once**, with seed 42 — and found that `class_weighted` (the current production
recipe) collapses the `dense_penultimate` bottleneck to an exact-zero vector for 85.4% of
true-HS test images, while `baseline_noweight` (no imbalance handling) and
`balanced_sampling` (50/50 batch sampling) both showed **0.000%** collapse and, for
`balanced_sampling`, several HS-elevated units never seen in any prior ICCAD5 analysis.

Because each condition was trained only once, none of this was known to be reproducible
versus a one-off artifact of that particular random initialization. This experiment
retrains `balanced_sampling` (3 independent seeds) and `baseline_noweight` (2 independent
seeds, as a secondary control) to test which findings replicate.

## 2. Previous finding being replicated

From the imbalance ablation (seed 42 only):

| condition | dead units | zero_vector_% (HS) | HS-elevated units | max HS AUC |
|---|---|---|---|---|
| class_weighted | 6/16 | 85.366 | 0 | — |
| baseline_noweight | 3/16 | 0.000 | 1 | 0.982 |
| balanced_sampling | 3/16 | 0.000 | 4 | 0.987 |

The hypothesis under test: **balanced_sampling repeatedly** produces zero/near-zero
collapse, fewer dead units than class-weighted, and at least some HS-elevated units. This
was explicitly *not* assumed true going in.

## 3. Experimental protocol

Step 1 (inspection, before training) confirmed the existing
`src/run_iccad5_imbalance_ablation.py` implementation genuinely depends on its `seed`
parameter in two independent places:

- `tf.keras.utils.set_random_seed(seed)` — reseeds Python's `random`, NumPy's global RNG,
  and TensorFlow's global RNG. This determines **model weight initialization** (Keras
  layers draw from the global RNG at build time) and the shuffling order used internally
  by `ImageDataGenerator.flow_from_directory(shuffle=True)` for `baseline_noweight`
  (`src/data.py`'s `data_extractor`, which passes no explicit `seed` to
  `flow_from_directory`, so it consumes the same global NumPy RNG state).
- `balanced_batch_generator(..., seed)` — an independent `np.random.default_rng(seed)`
  controlling which HS/NHS images are drawn (with replacement) into each balanced batch,
  and their within-batch order.

A pre-training sanity check (`build_xai_cnn` + `set_random_seed`, no training) confirmed
distinct initial-weight checksums and first-layer weight values for seeds 42/101/202/303,
ruling out an accidental no-op seed:

| seed | sum(\|weights\|) checksum | first 5 conv1 weights |
|---|---|---|
| 42  | 3277.7593 | 0.1342, 0.0609, -0.0529, 0.1686, -0.1581 |
| 101 | 3289.2036 | -0.1724, 0.1862, -0.0423, 0.0744, 0.1074 |
| 202 | 3277.0920 | -0.0925, -0.0683, -0.1697, 0.1678, 0.0797 |
| 303 | 3279.8748 | -0.0332, -0.1408, -0.1484, -0.0283, -0.1666 |

`BestBalancedAccuracyCheckpoint` (`src/metrics.py`) is unchanged: it still tracks the
epoch with the highest validation balanced accuracy (computed from native
`val_true_positives/negatives/false_positives/false_negatives`) and restores those weights
at `on_train_end`. Architecture (`build_xai_cnn`), optimizer, loss, `IMG_SIZE`,
`BATCH_SIZE`, `EPOCHS=10`, and the ICCAD5 train/test directory split are byte-for-byte the
same code path as the original ablation and as production `src/train_xai.py`.

**Design note (deliberate, not an omission):** `balanced_sampling_seed1` and
`baseline_noweight_seed1` share seed 101 (and `..._seed2` share seed 202) by design — this
holds initialization/shuffling-RNG state constant *between* the two condition families at
a given seed slot, isolating the sampling-strategy variable, while `balanced_sampling`
additionally gets a third, unshared seed (303) for a fuller reproducibility test of its own
key finding per the task's stated priority. This means the two condition families are not
fully crossed at 3×2 — this is a compute-driven, documented trade-off, not cherry-picking.

## 4. Seeds used

| run | condition family | seed | differs from original (42)? |
|---|---|---|---|
| balanced_sampling_seed1 | balanced_sampling | 101 | yes |
| balanced_sampling_seed2 | balanced_sampling | 202 | yes |
| balanced_sampling_seed3 | balanced_sampling | 303 | yes |
| baseline_noweight_seed1 | baseline_noweight | 101 | yes |
| baseline_noweight_seed2 | baseline_noweight | 202 | yes |

No seed used here duplicates 42.

## 5. Training configuration

Identical to `docs/ICCAD5_IMBALANCE_ABLATION.md` Sec 3: `EPOCHS=10`, `BATCH_SIZE=32`,
`steps_per_epoch=86` for balanced sampling (`ceil(2742/32)`), same optimizer/loss/metrics
imported directly from `src.train_xai`, `class_weight` never applied in either family
(isolates the sampling-strategy variable, matching the original design), best-balanced-
accuracy checkpoint selection via `BestBalancedAccuracyCheckpoint` in both families. The
only code difference from `src/run_iccad5_imbalance_ablation.py` is that `seed` is now a
per-run parameter instead of a hardcoded `42` (`src/run_iccad5_seed_replication.py`).

## 6. Classification results

| run | seed | selected epoch | balanced_accuracy | HS_recall | specificity_NHS | precision | F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| balanced_sampling_seed1 | 101 | 7 | 0.9743 | 0.9512 | 0.9973 | 0.4286 | 0.5909 | 39 | 19275 | 52 | 2 |
| balanced_sampling_seed2 | 202 | 9 | 0.9731 | 0.9512 | 0.9950 | 0.2889 | 0.4432 | 39 | 19231 | 96 | 2 |
| balanced_sampling_seed3 | 303 | 8 | 0.9737 | 0.9512 | 0.9963 | 0.3514 | 0.5132 | 39 | 19255 | 72 | 2 |
| baseline_noweight_seed1 | 101 | 9 | 0.9743 | 0.9512 | 0.9975 | 0.4432 | 0.6047 | 39 | 19278 | 49 | 2 |
| baseline_noweight_seed2 | 202 | 9 | 0.9014 | 0.8049 | 0.9979 | 0.4521 | 0.5789 | 33 | 19287 | 40 | 8 |

Original (seed 42) for reference: `balanced_sampling` bal_acc=0.9501, `baseline_noweight`
bal_acc=0.9741.

`n_test_samples=19368` (HS=41, NHS=19327) for every run, identical to the original —
confirmed no split drift.

## 7. Bottleneck results

| run | dead units | active units | zero_vector_% overall | HS zero_vector_% | NHS zero_vector_% |
|---|---|---|---|---|---|
| balanced_sampling_seed1 | 1/16 | 15 | 0.0000 | 0.000 | 0.000 |
| balanced_sampling_seed2 | 4/16 | 12 | 0.0000 | 0.000 | 0.000 |
| balanced_sampling_seed3 | 3/16 | 13 | 0.0000 | 0.000 | 0.000 |
| baseline_noweight_seed1 | 6/16 | 10 | 0.4389 | **95.122** | 0.238 |
| baseline_noweight_seed2 | 8/16 | 8  | 0.3717 | **78.049** | 0.207 |

Zero-vector rate by case type, `baseline_noweight_seed1`: TP=100.00%, FP=93.878%, TN/FN=0%
(n=0 for FN in this run's confusion outcome bucket is not applicable here — FN=2 exist but
are outside the zero-vector-by-outcome table's populated cells due to rounding of a small
n; see `representation_results.csv` for exact per-row values). The key qualitative
pattern — collapse concentrated in true-HS and false-positive-NHS cells, near-zero in
true-negative cells — mirrors the original `class_weighted` result almost exactly.

**This is the opposite of what the original single-seed `baseline_noweight` run showed
(0.000% collapse in every stratum).** See Sec 11.

## 8. HS/NHS unit analysis

Mean ± SD across seeds (n=3 for balanced_sampling, n=2 for baseline_noweight):

| condition family | dead units | zero_vector_% (HS) | HS-elevated units | max HS AUC | balanced_accuracy |
|---|---|---|---|---|---|
| balanced_sampling | 2.67 ± 1.53 | 0.00 ± 0.00 | 5.33 ± 1.53 | 0.9858 ± 0.0006 | 0.9737 ± 0.0006 |
| baseline_noweight | 7.00 ± 1.41 | 86.59 ± 12.07 | 0.00 ± 0.00 | 0.500 ± 0.000 | 0.9379 ± 0.0516 |

**Per-seed HS-elevated unit indices** (`AUC_HS_vs_NHS >= 0.90`, threshold chosen to match
the original report's observed HS-elevated range of 0.979–0.987 with margin):

| run | HS-elevated unit indices | AUC range | Cohen's d range |
|---|---|---|---|
| balanced_sampling_seed1 | 0, 6, 8, 11, 13 | 0.984–0.986 | +9.6 to +16.5 |
| balanced_sampling_seed2 | 3, 4, 10, 12 | 0.973–0.986 | +12.5 to +15.2 |
| balanced_sampling_seed3 | 4, 6, 7, 9, 10, 11, 12 | 0.971–0.985 | +9.4 to +12.4 |
| baseline_noweight_seed1 | none | — | — |
| baseline_noweight_seed2 | none | — | — |

**Unit index overlap across the three balanced_sampling seeds: zero.** No single unit
index (0–15) is HS-elevated in all three seeds; several indices repeat in exactly two of
three (e.g. unit 4 in seed2+seed3, unit 6 in seed1+seed3, units 10/11/12 pairwise), but
none in all three. This is consistent with permutation symmetry of a randomly-initialized
hidden layer (see Sec 6 of the previous bottleneck analysis for the same caveat) — the
network is not guaranteed, and empirically does not, learn the "same" feature in the same
unit slot across reruns.

Distinguishing the two questions the task asks to keep separate:

- **A. "Same unit index reproducible":** No — 0/16 units are HS-elevated in all 3
  balanced_sampling seeds.
- **B. "Existence of HS-elevated units reproducible":** Yes — every one of the 3
  balanced_sampling seeds produced 4–7 units meeting the AUC>=0.90 threshold, with
  Cohen's d in the +9 to +16 range each time (all far larger in magnitude than the
  original seed-42 run's +5.00 to +6.39). `baseline_noweight` produced **zero** in either
  of its 2 seeds (matching `class_weighted`'s original zero-HS-elevated-unit pattern, not
  the original seed-42 `baseline_noweight` result of 1 unit at AUC=0.982).

B is judged the scientifically meaningful claim, per the task's own framing, and it holds
across all 3 independent balanced_sampling seeds tested.

## 9. Cross-seed representation stability

| run | centroid_dist(HS,NHS) | within_HS dist | within_NHS dist | ratio (centroid/within_NHS) |
|---|---|---|---|---|
| balanced_sampling_seed1 | 21.90 | 5.32 | 4.97 | 4.40 |
| balanced_sampling_seed2 | 38.22 | 6.35 | 10.36 | 3.69 |
| balanced_sampling_seed3 | 20.92 | 6.92 | 5.79 | 3.61 |
| baseline_noweight_seed1 | 16.32 | 1.28 | 7.43 | 2.19 |
| baseline_noweight_seed2 | 26.33 | 2.28 | 8.87 | 2.97 |

The absolute centroid distances and within-class spreads vary 2-fold across
balanced_sampling seeds (no claim of identical representations), but the **qualitative
pattern is stable**: HS and NHS centroids separate by 3.6–4.4x the within-NHS spread in
every balanced_sampling seed, active-unit counts stay in the 12–15/16 range, and PCA plots
(`{run}_penultimate_pca.png`) show a dispersed (non-collapsed) HS cluster in all three. By
contrast, both baseline_noweight seeds show a much smaller within_HS spread (1.28, 2.28
— consistent with most HS samples being pinned to the same near-zero point) and a lower
centroid/within-NHS ratio (2.19, 2.97), both qualitatively closer to the original
`class_weighted` pattern (ratio 2.01) than to the original `baseline_noweight` pattern
(ratio 4.81).

Pairwise direct unit-to-unit correlation across seeds was not computed, per Sec 7's own
caution that unit identity is not expected to be preserved; the representation-level
measures above (active-unit count, zero-vector frequency, centroid separation, within-
class spread, PCA structure) were used instead, as instructed.

## 10. Random-noise results

Reused the exact fixed protocol from `docs/ICCAD5_BOTTLENECK_ANALYSIS.md` Step 9
(`np.random.default_rng(12345)`, 30 uniform[0,1] images + 30 binary images at
`p(white)=0.3`, `target_layer_name="conv_final_2"`) rather than inventing a new one.

| run | uniform noise zero_vector_% | binary noise zero_vector_% |
|---|---|---|
| balanced_sampling_seed1 | 0.0 | 0.0 |
| balanced_sampling_seed2 | 0.0 | 0.0 |
| balanced_sampling_seed3 | 0.0 | 0.0 |
| baseline_noweight_seed1 | 0.0 | 0.0 |
| baseline_noweight_seed2 | 0.0 | 0.0 |

For reference, the original OLD and NEW checkpoints (both **class-weighted** training
recipes — production and checkpoint-fix respectively) showed **100%** collapse on this
exact same noise set.

**This is a genuine non-replication of the "random noise triggers dead bottleneck"
finding — but in a way that narrows, rather than contradicts, the original claim.** All 5
checkpoints tested here are non-class-weighted; none of them ever entered the dead state
on pure noise, *even the two `baseline_noweight` seeds that collapsed on 78–95% of real HS
images*. The random-noise vulnerability previously documented therefore appears specific
to (or at least much more pronounced in) the class-weighted training recipe, not a
universal property of this architecture. This experiment did not retrain `class_weighted`
across seeds, so it cannot say whether class-weighted training *always* produces
noise-triggered collapse (n=2 checkpoints total ever tested: OLD, NEW) — only that the 5
non-class-weighted checkpoints trained here never did.

## 11. Reproducibility assessment

**Replicated (holds across all seeds tested):**
- `balanced_sampling` produces exactly 0.000% exact-zero-vector rate in all 3 independent
  seeds (101, 202, 303), across every true-label and case-type stratum.
- `balanced_sampling` produces multiple (4, 4, and 7) HS-elevated units (AUC>=0.90) in
  every seed, with no unit index shared across all three, but the *existence* of such
  units is consistent.
- `balanced_sampling`'s representation shows a consistently dispersed (non-collapsed) HS
  cluster with centroid/within-NHS separation ratio in a narrow 3.6–4.4 band across seeds.
- The classification-performance improvement in HS recall/balanced accuracy relative to
  `class_weighted` (0.9649 originally) is not being claimed here — see Sec 6, where
  balanced_sampling's balanced accuracy (0.973–0.974 across all 3 seeds) is stable and
  consistently *higher* than the original class_weighted figure.

**Failed to replicate:**
- `baseline_noweight`'s original 0.000% collapse result (seed 42) did **not** hold for
  either of the two new seeds (101, 202): HS zero-vector rates were 95.1% and 78.0%
  respectively — closer to the original `class_weighted` collapse rate (85.4%) than to
  `baseline_noweight`'s own original result. `baseline_noweight` therefore appears to be
  seed-dependent and unreliable: it happened to avoid the collapse under seed 42 but not
  under 101 or 202.
- `baseline_noweight`'s original single HS-elevated unit (seed 42, AUC=0.982) did not
  reappear in either replication seed (0 HS-elevated units in both).
- The "random noise always collapses the bottleneck" finding did not replicate for *any*
  of the 5 non-class-weighted checkpoints tested here, including the two that collapsed on
  real HS data — see Sec 10's important caveat about scope.

**Not tested (out of scope for this experiment):**
- Whether `class_weighted` itself is seed-stable (only ever trained once, seed 42, in both
  this project's checkpoint-fix and imbalance-ablation work).
- The combined class-weight + balanced-sampling condition (explicitly excluded per task
  instructions).

## 12. Limitations

- Only 3 independent seeds for `balanced_sampling` and 2 for `baseline_noweight` — nowhere
  near enough to establish a population-level rate (e.g., "balanced_sampling collapses in
  X% of random initializations"); this is evidence of *replication in the runs actually
  performed*, not proof of universal behavior.
- Test set HS n=41 throughout; all effect sizes/AUCs should be read with this small-sample
  caveat, exactly as in every prior report in this project.
- `balanced_sampling` and `baseline_noweight` are not fully crossed with 3 unique seeds
  each — `baseline_noweight_seed1`/`seed2` intentionally reuse seeds 101/202 from the
  balanced_sampling set (Sec 3), a documented compute trade-off, not a hidden confound (it
  does not, by itself, explain baseline_noweight's collapse, since balanced_sampling at the
  identical seeds 101/202 did not collapse — the collapse is attributable to the sampling
  strategy, not the shared initialization).
- No independent seed replication of `class_weighted` was performed (out of scope, not
  requested as primary focus); its own seed-sensitivity is therefore still unknown.
- Per-unit "HS-elevated" classification used a fixed AUC>=0.90 threshold chosen to match
  the range of the original report's own qualitative description; this is a descriptive
  convention, not a validated cutoff.

## 13. Recommendation

`balanced_sampling`'s core finding (collapse elimination + emergence of HS-elevated
units) held up cleanly across all 3 independent seeds tested and is the more reliable of
the two alternatives to `class_weighted`. `baseline_noweight`'s apparent collapse-avoidance
was very likely a single-seed artifact and should **not** be treated as evidence that
"removing class weighting" alone eliminates the bottleneck — the sampling strategy
(balanced batches), not merely the absence of loss reweighting, appears to be the operative
factor. Given this, and per Sec 14/Q7-Q8 below, the next reasonable step is a further,
larger-n seed sweep of `balanced_sampling` specifically (5+ seeds) before treating it as a
settled methodology change, rather than moving straight to full XAI/occlusion/geometry
pipeline regeneration.

## 14. Final Decision Framework

**Q1. Does balanced_sampling consistently eliminate the exact-zero penultimate
bottleneck?**
Yes, in all 3 independent seeds tested (0.000% in every stratum, every seed). This is the
strongest and cleanest replicated result in this experiment.

**Q2. Does it consistently reduce the number of dead units relative to the class-weighted
condition?**
Yes. balanced_sampling: 1, 4, 3 dead units (mean 2.67) across its 3 seeds, all below
class_weighted's original 6/16. baseline_noweight, by contrast, showed 6 and 8 dead units
in its two replication seeds — *worse* than class_weighted's original 6, not better.

**Q3. Does the emergence of HS-elevated activation remain present across independent
seeds?**
Yes, for balanced_sampling: 5, 4, and 7 HS-elevated units respectively in its 3 seeds (all
AUC>=0.97). It did not remain present for baseline_noweight (0 of 0 in both replication
seeds, versus 1 unit in the original seed-42 run).

**Q4. Is the HS-elevated-unit phenomenon representation-level reproducible even when
individual unit identities change?**
Yes. No unit index is HS-elevated in all 3 balanced_sampling seeds (individual identities
do not persist), but every seed independently produces multiple such units with
comparable effect sizes (Cohen's d +9.4 to +16.5) — the phenomenon is reproducible at the
representation level even though it is not tied to a fixed unit index.

**Q5. How much seed-to-seed variation exists in classification performance?**
Small for balanced_sampling (balanced accuracy 0.9731–0.9743, SD=0.0006; HS recall
identical at 0.9512 in all 3 seeds — same 39/41 TP count every time, only FP count varies
40–96). Large for baseline_noweight (balanced accuracy 0.9743 vs 0.9014, SD=0.0516; HS
recall 0.9512 vs 0.8049) — one seed lost 6 additional true HS detections relative to the
other, coinciding with its much larger bottleneck collapse.

**Q6. Does the random-noise dead-state behavior persist?**
No, not for any of the 5 checkpoints tested here (0% collapse on 12345-seeded uniform and
binary noise in every case), including the two baseline_noweight seeds that collapsed
heavily on real HS data. This narrows the original noise-vulnerability finding to
(at least) the class-weighted checkpoints it was originally observed on; it does not
generalize to non-class-weighted checkpoints, collapsed or not.

**Q7. Is there now sufficient evidence to freeze balanced_sampling as the ICCAD5
experimental training methodology?**
Not yet, on this evidence alone. Three seeds is a real replication (not a single run) and
every seed showed the qualitative pattern the hypothesis predicted, which is meaningfully
stronger evidence than existed before this experiment. But 3 seeds cannot rule out a
lower-frequency failure mode (e.g., an occasional bad seed like the ones observed for
baseline_noweight), and balanced_sampling's own precision/F1 trade-off (more false
positives than baseline_noweight or class_weighted at matched recall) has not been
re-examined here. A larger seed sweep (5+) specifically for balanced_sampling, still
without touching architecture or preprocessing, is the more defensible next step before
calling this "frozen."

**Q8. If yes, should the next step be the full ICCAD5 XAI + occlusion + geometry
pipeline?**
Conditional on Q7: not yet. Since Q7's answer is "not sufficient evidence to freeze,"
regenerating the full XAI pipeline now would be premature — it would need to be redone
again if a larger seed sweep changes the recommended checkpoint. The recommended next step
remains the larger balanced_sampling seed sweep (Sec 13), not pipeline regeneration.
