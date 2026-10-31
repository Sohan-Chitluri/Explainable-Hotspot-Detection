# Pre-experiment research state

Baseline commit: c4f7cb1 (main initially clean). Graphify query used existing docs graph: 44 nodes, 91 edges, 7 communities. It maps document relationships, not executable-source relationships; graph claims are navigation aids, not independent evidence.

## Preserved baseline

- Existing baseline/XAI models, 57-sample XAI/occlusion/geometry lineage, ICCAD5 checkpoint-fix/imbalance/seed investigations remain intact.
- Final working checkpoint is balanced_sampling seed 101. SHA256 verified directly: 9479f07781b0128e19d891b295c99b544f9e6fd2b9a4628533138a47c8085b70. The copied final_iccad5/_model_input checkpoint matches it.
- Baseline artifact inventory covers 1,971 existing files, including 19 .keras checkpoint files (one final-input copy), totaling 268,250,047 bytes. See results/xai/improvements/validation/baseline_artifact_manifest.json. This inventory precedes new numerical work and excludes assigned improvement/forensics outputs.
- Report baseline: balanced seeds 101/202/303, historical balanced accuracy about 0.9731–0.9743, 0% exact-zero vectors. CPU independent recomputation delegated; no new seed results exist yet.
- Three-seed capstone adoption is not production-level freezing. Noise-collapse finding must remain scoped to tested checkpoint lineages.

## Verified methodological issues / open claims

1. src/data.py:15–16 maps validation_dir to the official test directory. src/data.py:54–73 builds training/validation generators without a separate holdout. Best-checkpoint evaluation on that same partition would create test-selection bias; A/G/H are tracing all consumers before approving replication.
2. Official generators use Keras defaults (nearest-neighbor); XAI uses the historical PIL resize path. Corrected preprocessing must start from original 1200px dataset files, not already-bicubic 224px original.png artifacts. B requires pixel and GPU prediction equivalence before downstream corrected results.
3. src/run_geometry_analysis.py:147–154 flags ptp < 1e-6 but still computes arbitrary top-K masks for flat maps. Existing main geometry predates the guard. Separate historical all/nondegenerate/degenerate analyses are necessary; flat-map overlap is not attribution evidence.
4. Final report makes claims stronger than its cited documents: class-weight causality despite unweighted-seed collapse; balanced accuracy described as inflated by prevalence; four-method TP convergence/significance despite mixed original-set Grad-CAM++ findings; final TN GAP mechanism inferred rather than directly measured. These require evidence-led wording correction at synthesis, not ad hoc replacement of historical numbers.
5. Final Grad-CAM degeneracy and penultimate-vector collapse are distinct. Direct final-checkpoint gradient probe has not been run in this improvement pass.

## Environment and experiment gate

GPU environment note: Prior device visibility / driver issue (nvidia-smi exit 6) has been resolved. NVIDIA GPU acceleration is available and operational. GPU-heavy training and inference jobs must be serialized after methodology approval. Optional synthetic experiment is deferred.

## Scientific framing

Keep classifier prediction, model spatial attribution, perturbation sensitivity, raster-proxy correspondence and physical lithography causality separate. Existing raster evidence cannot establish physical mechanisms. Analyses are descriptive/exploratory, confidence-selected small samples are not population estimates, zero-map undefined metrics are not evidence of localization, and test-selected checkpoint metrics are not untouched-test generalization estimates.
