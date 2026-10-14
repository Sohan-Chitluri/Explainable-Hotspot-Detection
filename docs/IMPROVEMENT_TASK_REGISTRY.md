# Research improvement task registry

Baseline: c4f7cb1, main, initially clean. Lead owns this registry, final synthesis integration and final report. Existing checkpoints/results are immutable. New outputs belong under results/xai/improvements/<workstream>/; D may add a unique degeneracy_forensics subdirectory under results/xai/final_iccad5/. No agent commits or pushes.

## Current stopping point

Update (2026-09-17, post-interrupt resumption): preprocessing fix committed as 9ea630d with RED/GREEN-verified tests. CPU pilot (20 deterministic ICCAD5 test rasters, seed-1729 selection, frozen checkpoint SHA verified): corrected tensors exactly equal official path on every image; nearest-vs-bicubic probability delta max 0.0808 / mean 0.0130 with 0 prediction flips (results/xai/improvements/preprocessing/prediction_pilot20.csv, prediction_pilot_summary.json). C historical sensitivity analysis completed (docs/IMPROVEMENT_GEOMETRY_REANALYSIS.md, results/xai/improvements/geometry_reanalysis/): published top-20% medians reproduce exactly; spread==0 proxy reproduces published degeneracy counts exactly; under a pre-declared 64-test Holm family, LayerCAM/Occlusion geometry claims survive while Grad-CAM TP/FP per-cell claims do not; all FN cells null. D artifact audit (partial, interrupted) established the two flagged final-set TNs are byte-identical duplicate sources of one raster — direct decoded-pixel hashing confirms 1/2 unique TN rasters flagged, corresponding to 2/3 TN filenames; a full-cohort unique denominator is not established. G/H training remains NO-GO (test-as-validation selection, per audit A-01). E completed with robust metric implementation but actual agreement blocked by missing lossless CAM arrays. F/G/I remain incomplete after denied inspection commands. J, K, synthesis, final-report synchronization and final release validation are NOT complete. No pushes.

## Current stopping point (superseded context)

## Current stopping point (superseded context)

Lead applied a minimal explicit-nearest resize at src/xai_engine.py:117, not the broader shared-loader migration. Regression failed before and passed after. Deterministic CPU comparison covered 1,219 real rasters: corrected tensor max/mean absolute difference 0, zero unequal images; historical bicubic differed on 1,214. Outstanding agents were asked to stop; table below records assignments, not completed experiments. Full C/D/G/H/I work, J, synthesis, final-report synchronization and final release validation are NOT complete. No commits/pushes. Interim integrity validation found all 1,971 baseline artifacts /19 checkpoint files unchanged. See improvements/validation/artifact_validation.json. New audit/preparation files are partial deliverables and must be reviewed before resuming.

## Execution gates

Note: Prior temporary GPU environment issues (nvidia-smi exit 6 / CUDA device visibility) are resolved. GPU acceleration is available. Jobs will be serialized by the lead, never autonomously launched by agents.

| ID | Agent | Objective / allowed writes | Status | Dependencies | Output | Validation |
|---|---|---|---|---|---|---|
| A | methodology auditor | Read all relevant source/results; only docs/IMPROVEMENT_METHODOLOGY_AUDIT.md | **resolved** (2026-09-20) | none | audit with issue/evidence/severity/fix/conclusion impact | **RESOLVED: checkpoint-selection leakage confirmed HIGH (train_xai.py) + LOW-MODERATE (train_baseline.py). Fix: src/data.py::make_three_way_split (val_frac=0.15, seed=42, stratified per class). All 5 XAI models retrained; average bias quantified at −0.45 pp (range −5.32 pp iccad5 to +5.27 pp iccad2). FINAL_RESEARCH_REPORT.md §7/§11/§20 updated with clean-split numbers. Residual leakage in run_iccad5_imbalance_ablation.py + run_iccad5_seed_replication.py flagged TODO(leakage-fix) — next migration targets. See docs/TEST_VALIDATION_LEAKAGE_AUDIT.md, docs/THREE_WAY_SPLIT.md, docs/RETRAIN_CLEAN_SPLIT_RESULTS.md.** |
| B | gradcam++ auditor | docs/GRADCAMPP_VALIDATION.md; scripts/improvements/gradcampp_validation.py (stub) | completed | none | implementation deviates from published formula (2nd/3rd derivative, alpha) | manual verification + CPU sanity stub |
| C | geometry auditor | docs/GEOMETRY_STATISTICAL_AUDIT.md; results/xai/improvements/geometry_statistical_audit/ | completed | none | 16/16 medians reproduced; 24/64 tests survive Holm; duplicates found | independent reproduction from stored CSVs |
| D | degeneracy auditor | docs/FINAL_GRADCAM_DEGENERACY_ANALYSIS.md; scripts/improvements/degeneracy_forensics.py; final_iccad5/degeneracy_forensics/ only | audit/preparation | B for corrected inference | direct final-checkpoint gradient evidence or explicit blocked status | pending |
| E | agreement analyst | docs/XAI_AGREEMENT_ANALYSIS.md; scripts/improvements/agreement_analysis.py; improvements/agreement/ only | historical artifact audit | corrected rerun depends B | robust overlap/support correlation metrics | pending |
| F | concentration analyst | docs/XAI_CONCENTRATION_ANALYSIS.md; scripts/improvements/concentration_analysis.py; improvements/concentration/ only | historical artifact audit | corrected rerun depends B | concentration CSV/plots | completed |
| G | balanced replication auditor | docs/IMPROVEMENT_BALANCED_REPLICATION.md; scripts/improvements/balanced_replication.py; improvements/balanced_replication/ only | audit/preparation, no training | A approval | seeds 404/505/606 only if exact protocol verified | completed |
| H | weighted replication auditor | docs/IMPROVEMENT_WEIGHTED_REPLICATION.md; scripts/improvements/weighted_replication.py; improvements/weighted_replication/ only | audit/preparation, no training | A approval | class-weighted seeds 101/202/303 | completed |
| I | expansion analyst | docs/IMPROVEMENT_SAMPLE_EXPANSION.md; scripts/improvements/sample_expansion.py; improvements/sample_expansion/ only | existing-prediction audit | B if new predictions/maps needed | deterministic expanded selection, limitations | completed |
| J | pending | enrichment from C, isolated outputs | gated | C | ratios/uncertainty | pending |
| K | pending | optional synthetic sanity test | deferred | higher priorities complete | nonphysical sanity tests only | not started |
| S | pending synthesis agent | docs/IMPROVEMENT_SYNTHESIS.md only | gated | completed/blocked workstream evidence | integrated recommendations | pending |
| V | lead | docs/FINAL_VALIDATION.md; improvements/validation/ | baseline inventory | all work | immutable artifact checksum comparison, CSV/figure/code checks | pending |

Agents must not touch other agents' source files, baseline report, existing result files or checkpoints. Undefined scientific quantities must carry explicit validity/reason fields, not invented zeros or unreported NaNs. Hypothesis testing is exploratory with specified multiple-testing families. No production-freeze claim, no geometry-to-physics causality claim, no universal XAI ranking.
