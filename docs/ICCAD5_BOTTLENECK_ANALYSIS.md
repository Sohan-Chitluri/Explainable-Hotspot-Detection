# ICCAD5 Bottleneck Analysis

Activation-level forensic comparison of the OLD (`models/xai/xai_cnn_iccad5.keras`,
final-epoch) and NEW (`models/xai/retrained/xai_cnn_iccad5_best.keras`, best-epoch)
iccad5 checkpoints, focused on the `dense_penultimate` 16-unit ReLU bottleneck identified
in `docs/ICCAD5_FORENSIC_AUDIT.md` and re-examined in `docs/ICCAD5_CHECKPOINT_FIX.md`. This
is an **analysis-only** pass: no retraining, no architecture change, no checkpoint
overwritten, no XAI implementation touched. Full outputs in
`results/xai/iccad5_bottleneck/`.

## 1. Objective

Determine, with direct evidence rather than inference from aggregate metrics alone:
whether `dense_penultimate` carries measurable HS/NHS information; whether particular
units are class-selective; how the collapse (exact-zero output) is distributed across
classes and prediction outcomes; whether the best-epoch checkpoint fix improved the
*representation* itself or only shifted the decision threshold; and whether the evidence
justifies a follow-up class-imbalance experiment.

## 2. Models Compared

| | Path | Provenance |
|---|---|---|
| OLD | `models/xai/xai_cnn_iccad5.keras` | Original, final-epoch-saved checkpoint (unmodified, byte-identical to the committed version — re-verified) |
| NEW | `models/xai/retrained/xai_cnn_iccad5_best.keras` | Best-validation-balanced-accuracy checkpoint from `docs/ICCAD5_CHECKPOINT_FIX.md`, epoch 8/10, saved to a separate directory |

Neither file was modified by this analysis.

## 3. Architecture Verification

Both checkpoints loaded independently; confirmed identical: 21 layers, 128,321 parameters,
identical layer names/classes/shapes (same check as `docs/ICCAD5_FORENSIC_AUDIT.md` §1,
re-run here for these two specific files). `dense_penultimate` is `Dense(16,
activation="relu")` in both. This is the OLD-vs-NEW-best-checkpoint comparison the task
specifies, not a comparison against a modified architecture.

## 4. Dataset / Sample Counts

Full iccad5 **test** set: 41 HS, 19,327 NHS (471:1 imbalance) — every figure below states
HS and NHS counts explicitly per the task's statistical-discipline requirement, since HS
n=41 (and often far fewer once split further by prediction outcome) makes many per-group
estimates inherently unstable.

## 5. Per-Unit Activation Analysis

Full table: `results/xai/iccad5_bottleneck/per_unit_statistics.csv`. Distributions:
`old_unit_activation_distributions.png`, `new_unit_activation_distributions.png`.

**Six of sixteen units are completely dead — output exactly zero for every one of the
19,368 test samples, in both checkpoints** (OLD: units 0, 1, 2, 9, 11, 12; NEW: units 0,
1, 6, 11, 14, 15 — an overlapping but not identical set). The bottleneck's *effective*
capacity is therefore closer to 10 live units than 16, in both checkpoints.

For the live units, activation is **strongly, consistently higher for NHS than HS** —
every live unit's `auc_HS_vs_NHS` (Mann-Whitney rank statistic, i.e. P(random HS value >
random NHS value)) is well below 0.5, several below 0.10 (e.g. OLD unit 15: AUC=0.046,
Cohen's d=−2.03; NEW unit 8: AUC=0.026, Cohen's d=−2.66 — large effect sizes by
conventional thresholds). **This large effect size is driven almost entirely by the
zero-vector phenomenon itself, not by a graded difference among "live" activations** —
see the alive-only supplementary check below. It should be read as "HS samples
overwhelmingly produce the zero vector; NHS samples overwhelmingly produce a nonzero,
often large, vector," which is a real and measurable separation, but a *binary* one
(dead-vs-alive) more than a graded discriminative code.

**Supplementary check restricted to alive samples only**
(`alive_only_unit_comparison.csv`): OLD has only **1** alive HS sample in the entire test
set; NEW has **9**. Both are far too small for any robust inference about whether alive-HS
activations differ systematically from alive-NHS activations on a per-unit basis — this
check is reported descriptively only, and no claim of "class-selective" is made from it.

**No unit is labeled a "detector."** Per the task's instruction, findings are described as
"higher activation in NHS samples" or "class-associated," not "HS/NHS detector" units,
since the underlying phenomenon (binary dead/alive routing) does not support a
feature-detector interpretation.

## 6. Dead-Unit Analysis

`results/xai/iccad5_bottleneck/dead_unit_classification.csv`, `zero_vector_analysis.csv`,
`zero_vector_rate_by_case_type.png`.

| Checkpoint | Case | n | Zero-vector count | Zero-vector % |
|---|---|---:|---:|---:|
| OLD | TP | 40 | 40 | **100.0%** |
| OLD | FN | 1 | 0 | 0.0% |
| OLD | TN | 18,084 | 0 | 0.0% |
| OLD | FP | 1,243 | 912 | 73.4% |
| NEW | TP | 38 | 32 | 84.2% |
| NEW | FN | 3 | 0 | 0.0% |
| NEW | TN | 18,914 | 0 | 0.0% |
| NEW | FP | 413 | 319 | 77.2% |

**Internal consistency check (passes)**: TN and FN are exactly 0% zero-vector in *both*
checkpoints. This is logically required — the zero vector always produces the same
negative logit (bias-only, §7 of the forensic audit), which always falls on the HS side
of the decision threshold, so a zero-vector sample can never be predicted NHS. Getting
exactly 0% for TN/FN in unbiased, independently-computed data is a strong sanity check
that the extraction pipeline and the mechanism description are both correct.

**Predominant-class pattern**: every "live" unit that is not globally dead is classified
`predominant_class = NHS` in `dead_unit_classification.csv` (none flagged HS-predominant)
— consistent with §5's finding that the live units encode "NHS-ness," and HS prediction is
substantially achieved through the *absence* of that signal (the dead branch) rather than
a dedicated positive pathway.

## 7. Zero-Vector Forensics

`zero_vector_conv_forensics_summary.csv`, `zero_vector_cosine_forensics.csv`. Central
question: **does information exist in `conv_final_2` that is being destroyed by
`dense_penultimate`?**

| Checkpoint | Group | Mean pairwise cosine similarity (conv_final_2, n=20 each) | Mean conv_final_2 norm |
|---|---|---:|---:|
| OLD | zero-vector HS samples | 0.808 | 490.4 |
| OLD | non-zero-vector NHS samples | 0.529 | 312.7 |
| NEW | zero-vector HS samples | 0.834 | 466.8 |
| NEW | non-zero-vector NHS samples | 0.706 | 306.8 |

**Answer: yes, real but limited information exists and is discarded.** Zero-vector
samples' `conv_final_2` activations are *not* literally identical to each other (pairwise
cosine 0.81–0.83, not 1.0 — genuine per-sample variation survives all the way to the final
conv layer) but they are considerably more homogeneous than the (already-somewhat-similar)
non-zero-vector NHS population. `conv_final_2` norm is also **larger**, not smaller, for
the dead-branch cluster (490–550 vs 307–341) — the collapse is not a case of "weak signal
reaching the bottleneck"; it is a strong, fairly consistent-direction signal that the
specific learned `dense_penultimate` weight matrix happens to map to negative
pre-activations across nearly all 16 units. `conv_final_2_nonzero_frac` is at ceiling
(std=0.0 across every group) for all samples regardless of zero/non-zero downstream state
— confirming (as in the original forensic audit) that the collapse is specific to the
`dense_penultimate` linear projection, not a property of the conv stack itself.

## 8. OLD vs NEW Representation Comparison

`representation_summary.csv`, all 19,368 samples matched by `sample_id` between
checkpoints.

| Transition | Count | Notes |
|---|---:|---|
| nonzero → nonzero | 18,394 | mean cosine(old,new) = 0.280 (std 0.165) — representation genuinely *changed*, not merely rescaled, even where both stayed alive |
| zero → nonzero | 623 | "revived" by the fix |
| zero → zero | 329 | unaffected by the fix |
| nonzero → zero | 22 | new regressions (0.1% of all samples) |

**By old case type** (the key breakdown for understanding *what* the fix actually did):

| OLD case | zero→zero | zero→nonzero | nonzero→zero | nonzero→nonzero |
|---|---:|---:|---:|---:|
| TP (n=40, all zero) | 32 | **8** | — | — |
| FP (n=1,243) | 297 | **615** | 4 | 327 |
| TN (n=18,084) | — | — | 18 | 18,066 |
| FN (n=1) | — | — | — | 1 |

**67.4% (615/912) of OLD's zero-vector False Positives were revived into a genuinely
input-dependent representation by the checkpoint fix** — this is the single largest
concrete effect of the fix and is the most direct mechanistic explanation for why FP count
fell 1,456→413 (`ICCAD5_CHECKPOINT_FIX.md` §7). By contrast, only 20% (8/40) of OLD's
zero-vector True Positives were revived — **the fix disproportionately helped false
positives, not true positives**. Overall prediction agreement between OLD and NEW is
95.3%.

**Class separability** (`class_separability_summary.csv`): centroid distance between HS
and NHS in the full 16-D space increased from 18.79 (OLD) to 26.49 (NEW), and the ratio of
centroid distance to within-NHS spread increased from 1.57 to 1.99. **This number should
be read cautiously**: since 97.6%/78.0% of HS points sit exactly at the origin in both
checkpoints, "within-HS distance" and "centroid distance" are substantially statistics
about how far the *NHS* cloud sits from the origin, not evidence of rich internal HS
structure — flagged explicitly rather than presented as unqualified evidence of improved
separability.

## 9. HS/NHS Representation Analysis (PCA)

`old_penultimate_pca.png`, `new_penultimate_pca.png` — all 41 HS points plus a random
1,000-point NHS subsample (documented, seed=0; full-population NHS retained for every
numeric statistic above, subsampled only for a plottable/tractable 2-D projection).

**Both plots show the same qualitative structure**: the NHS population forms a broad,
diffuse cloud; the large majority of HS points (40/41 OLD, 32/41 NEW) collapse onto a
single, tightly overlapping point at the extreme edge of PC1 (77–83% of variance — largely
an overall-activation-magnitude axis) — this is the zero vector, and it projects to the
same location precisely because it *is* the same vector for every such sample. The
handful of "alive" HS points (1 for OLD, up to 9 for NEW) scatter inside or at the margin
of the NHS cloud, not forming a distinguishable HS sub-cluster of their own.

**This is descriptive, not a proof of separability or non-separability** — per the task
instruction, no silhouette score or nearest-neighbor claim is over-interpreted here: with
effectively 1–9 non-degenerate HS points, any formal separability statistic on the "alive"
subpopulation would be near-meaningless. The honest reading is: (a) the bottleneck clearly
"separates" the classes in the trivial sense of routing HS overwhelmingly through one
specific point (the origin) and NHS through a broad cloud elsewhere, and (b) this reveals
essentially nothing about whether the network has learned any graded, feature-based HS
representation, because there are too few non-degenerate HS examples to check.

## 10. Random-Noise Experiment

`noise_analysis.csv`. Fixed, documented seed (`numpy.random.default_rng(12345)`), 30
uniform-noise images and 30 binary-noise images (`p(white)=0.3`, matching the real
dataset's rough white-pixel fraction) per checkpoint — diagnostic only, not treated as a
benchmark.

**100% of all 60 noise images (both types) produced the exact-zero `dense_penultimate`
vector, in both OLD and NEW checkpoints** — no change from the original forensic audit's
finding. Compared to real test-set zero-vector rates (HS: 97.6%→78.0%, NHS: 4.7%→1.7%),
the noise result is **more extreme than either real-data rate in both checkpoints** — pure
noise is *more* likely to trigger the dead state than even a real true-HS image. This
supports the forensic audit's conclusion that the dead region of input space is broad and
not narrowly tailored to genuine hotspot semantics, and confirms **the checkpoint fix did
not narrow this broad dead region** — it only changed which specific *real* images happen
to fall inside vs. outside it.

## 11. Relationship to Existing XAI Degeneracy

`xai_degeneracy_relationship.csv` — reuses the existing 10-sample XAI sanity check from
`ICCAD5_CHECKPOINT_FIX.md` §9 (Grad-CAM/Grad-CAM++/LayerCAM/Occlusion degenerate flags);
the full XAI pipeline was **not** rerun, per instruction. `is_zero_vector` was recomputed
correctly for these exact 10 samples (an initial merge attempt using raw test-set filename
stems mismatched 5 of 10 samples due to an unrelated filename-sanitization quirk in the
original dataset — `NHSCAD51.png1.png` vs. the project's sanitized `NHSCAD51_1` sample ID
— corrected before reporting; this was a bug in this analysis script only, not in any
existing project code, and is noted for transparency):

| Method | Concordance with `is_zero_vector` (n=20, both checkpoints) |
|---|---:|
| LayerCAM | **1.000** |
| Occlusion | 0.900 |
| Grad-CAM | 0.500 |
| Grad-CAM++ | 0.450 |

**Association, not causation, is claimed.** LayerCAM's degeneracy is perfectly
concordant with the zero-vector state on this small sample — consistent with (not proof
of) the mechanism already established in `TP_FP_GEOMETRY_ANALYSIS.md` §3/§11: LayerCAM's
pixel-wise gradient weighting has no separate failure mode beyond the shared zero-gradient
state. Occlusion (a pure forward-pass method) is nearly as concordant (0.900), as expected
mechanistically — it degenerates precisely when the model's *output* doesn't move, which
is exactly the zero-vector condition. **Grad-CAM and Grad-CAM++ are markedly less
concordant (0.50, 0.45)** — both remain degenerate on several samples where
`dense_penultimate` is *not* the zero vector (e.g. all three TN samples in this subset),
confirming they have an additional failure mode (global-average-pooling gradient
cancellation, per `TP_FP_GEOMETRY_ANALYSIS.md` §11b) that operates independently of full
bottleneck collapse and is therefore **not** expected to improve from a checkpoint-
selection fix alone — consistent with `ICCAD5_CHECKPOINT_FIX.md` §9's finding that
Grad-CAM/Grad-CAM++ degeneracy did not improve on the matched 10-sample subset.

## 12. Limitations

- HS test n=41 (and as few as 1 for "alive-HS" under OLD) makes every HS-specific
  statistic in this report high-variance; all such numbers are reported with explicit
  counts and should not be read as precise population estimates.
- PCA and centroid-distance statistics are descriptive; no formal separability test
  (silhouette, classifier-based) was performed given the above sample-size constraint —
  this was a deliberate choice, not an oversight (the task explicitly cautions against
  claiming a visualization "proves" separability).
- The `conv_final_2` cosine-similarity comparison (§7) used n=20 samples per group for
  tractability, not the full population — reported as such.
- Effect sizes in §5 are driven substantially by the binary dead/alive split rather than
  graded within-class variation; this is stated explicitly to avoid overclaiming
  "class-selective detector" units.
- This analysis does not establish *why* the dense_penultimate weight matrix routes the
  way it does (i.e. which upstream conv_final_2 directions specifically trigger the dead
  state) — that would require further weight-space analysis not performed here.

## 13. Evidence Regarding Class Imbalance

Every quantitative finding above is consistent with, and none contradicts, the class-
imbalance explanation already advanced in `docs/ICCAD5_FORENSIC_AUDIT.md`:

- The bottleneck's live units encode "NHS-ness" almost exclusively (§5, §6) — plausible
  given NHS outnumbers HS 104.5:1 in training; a narrow 16-unit layer trained under this
  imbalance may default to a "detect the majority class positively, everything else
  defaults to the other class via the bias" strategy rather than learning a genuine
  positive HS representation, since there were only 26 positive examples to shape one.
- The checkpoint fix (which does not touch class weighting) revived a majority of FP zero-
  vectors but only a fifth of TP zero-vectors (§8) — consistent with the bottleneck simply
  having very little to learn *from* for the HS class regardless of which epoch is
  selected.
- Noise images collapse into the dead state even more reliably than real HS images (§10)
  — consistent with the dead region being the "default"/majority-class-adjacent basin of a
  representation shaped overwhelmingly by 2,716 NHS training examples against only 26 HS
  ones, rather than a specifically HS-tuned decision surface.

This is **evidence consistent with** the class-imbalance hypothesis, not proof that class
weighting or balanced sampling will resolve it — a controlled experiment isolating that
variable (§14) is the appropriate next step to test it directly, which this analysis does
not attempt.

## 14. Recommended Next Experiment (not implemented)

Per the task's final decision framework (Q7): a controlled experiment that holds
architecture, preprocessing, and the train/test split fixed, and isolates **how positive-
class scarcity is handled during training** — for example, comparing (a) the current
`class_weight` scheme (already in use) against (b) an oversampling/balanced-batch scheme
for the 26 HS training images, or (c) a range of class-weight magnitudes, all evaluated
with the same best-epoch checkpoint-selection mechanism already implemented
(`BestBalancedAccuracyCheckpoint`) so that the class-imbalance variable is isolated from
the checkpoint-selection variable already fixed. This experiment is **specified, not
run**, per the task's explicit instruction not to implement it now.

---

## Files Created

`results/xai/iccad5_bottleneck/`:
`activation_summary.csv`, `sample_activation_comparison.csv`, `per_unit_statistics.csv`,
`alive_only_unit_comparison.csv`, `dead_unit_classification.csv`,
`zero_vector_analysis.csv`, `zero_vector_conv_forensics_summary.csv`,
`zero_vector_cosine_forensics.csv`, `representation_summary.csv`,
`class_separability_summary.csv`, `noise_analysis.csv`,
`xai_degeneracy_relationship.csv`, `old_penultimate_pca.png`, `new_penultimate_pca.png`,
`old_unit_activation_distributions.png`, `new_unit_activation_distributions.png`,
`zero_vector_rate_by_case_type.png`.

## Final Decision Framework

- **Q1 (does dense_penultimate preserve measurable HS/NHS information?)**: Yes, but almost
  entirely in binary (dead-vs-alive) form rather than graded form — HS routes to the exact
  zero vector 78–98% of the time; when alive, live units are strongly NHS-elevated
  (large effect sizes), but this is confounded by the zero/nonzero split itself (§5).
- **Q2 (are specific units class-selective?)**: No unit is HS-selective in either
  checkpoint; several live units are consistently NHS-elevated, and 6/16 units are
  globally dead regardless of checkpoint or class. Described as "class-associated
  activation," not "detectors," per task instruction.
- **Q3 (how much did the fix reduce the all-zero bottleneck?)**: Overall 4.92%→1.81%;
  HS-specific 97.56%→78.05%; NHS-specific 4.72%→1.65% — a real, substantial reduction, not
  elimination.
- **Q4 (is the remaining collapse concentrated in particular groups?)**: Yes — concentrated
  in TP and FP (the two HS-predicting outcomes; TN/FN are structurally always 0% by
  construction, §6). Under NEW, TP still 84.2% zero-vector vs. FP 77.2% — true hotspots
  remain more collapse-prone than false alarms even after the fix.
- **Q5 (does the new checkpoint improve the representation, or mostly the decision layer?)**:
  Both, but the representation change is real and substantial, not merely a bias shift —
  623 samples transitioned zero→nonzero (§8), mean cosine(old,new)=0.28 even among samples
  that stayed nonzero in both (i.e. the live representation itself changed direction, not
  just scale), and 67.4% of FP zero-vectors were revived specifically — this is a genuine,
  quantified representational change, concentrated on false positives rather than true
  positives.
- **Q6 (do the results support proceeding to a class-imbalance experiment?)**: Yes — §13's
  evidence (NHS-dominated live units, disproportionate FP-vs-TP revival, noise more
  collapse-prone than real HS images) is consistent with a class-imbalance-driven
  representational limitation that checkpoint selection alone cannot resolve, and none of
  it is explained by a code or evaluation bug (ruled out in the prior forensic audit).
- **Q7 (next experiment, not implemented)**: See §14 — a class-weighting/balanced-sampling
  ablation with architecture, preprocessing, and split held fixed, using the already-
  implemented best-epoch checkpoint selection as the shared evaluation procedure across
  conditions.
