# Graph Report - docs  (2026-09-17)

## Corpus Check
- Corpus is ~42,433 words - fits in a single context window. You may not need a graph.

## Summary
- 44 nodes · 91 edges · 7 communities (6 shown, 1 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- ICCAD5 Imbalance & Bottleneck Investigation
- ICCAD5 Ablation & Seed Replication Docs
- CNN Architecture & ICCAD-12 Dataset
- XAI Attribution Methods (Grad-CAM family)
- Occlusion, Geometry & TP/TN/FP/FN Pipeline
- Geometry/Occlusion Analysis Reports
- Grad-CAM Citation

## God Nodes (most connected - your core abstractions)
1. `Explainable Hotspot Detection on ICCAD-12: A Multi-Method XAI Study of a Custom CNN Lithography Hotspot Classifier` - 18 edges
2. `XAI Research Audit` - 10 edges
3. `ICCAD5 Forensic Audit` - 9 edges
4. `TP vs FP Geometry Analysis` - 9 edges
5. `ICCAD5 Class-Imbalance Ablation` - 8 edges
6. `Geometry Analysis Report` - 7 edges
7. `ICCAD5 Checkpoint Fix` - 7 edges
8. `Final ICCAD5 XAI Synthesis (balanced_sampling seed1, frozen checkpoint)` - 6 edges
9. `Grad-CAM Visual Explanations for Lithography Hotspot Detection` - 6 edges
10. `ICCAD5 Bottleneck Analysis` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Occlusion Sensitivity: gradient-free perturbation method, 32x32 sliding window / 16px stride / baseline 0.0. Rationale: baseline 0.0 justified as the literal exact background pixel value in this strictly binary raster domain` --semantically_similar_to--> `Raster-Geometry Correlation: Methodology`  [INFERRED] [semantically similar]
  OCCLUSION.md → RASTER_GEOMETRY_METHOD.md
- `TP/TN/FP/FN Interpretation stage: case-type-stratified reading of attribution/occlusion/geometry evidence` --conceptually_related_to--> `TP vs FP Geometry Analysis`  [EXTRACTED]
  FINAL_RESEARCH_REPORT.md → TP_FP_GEOMETRY_ANALYSIS.md
- `Final ICCAD5 XAI/occlusion/geometry evidence pass on the frozen checkpoint (11 samples: TP=3,TN=3,FP=3,FN=2). Rationale: reproduces the TP-strongest/Grad-CAM-least-reliable-outside-TP pattern; 6/132 geometry rows (18.2% of Grad-CAM rows) flagged is_degenerate_map, all Grad-CAM, all TN` --conceptually_related_to--> `XAI Attribution stage (Grad-CAM, Grad-CAM++, LayerCAM)`  [EXTRACTED]
  FINAL_ICCAD5_SYNTHESIS.md → FINAL_RESEARCH_REPORT.md
- `Final ICCAD5 XAI Synthesis (balanced_sampling seed1, frozen checkpoint)` --cites--> `TP vs FP Geometry Analysis`  [EXTRACTED]
  FINAL_ICCAD5_SYNTHESIS.md → TP_FP_GEOMETRY_ANALYSIS.md
- `Explainable Hotspot Detection on ICCAD-12: A Multi-Method XAI Study of a Custom CNN Lithography Hotspot Classifier` --cites--> `Geometry Analysis Report`  [EXTRACTED]
  FINAL_RESEARCH_REPORT.md → GEOMETRY_ANALYSIS_REPORT.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Grad-CAM Degeneracy Investigation Across Documents** — docs_gradcam_grad_cam, docs_final_iccad5_synthesis_is_degenerate_map_finding, docs_tp_fp_geometry_analysis_gap_cancellation_mechanism, docs_geometry_analysis_report_document, docs_xai_research_audit_document [INFERRED 0.85]
- **ICCAD5 Methodological Escalation Chain** — docs_iccad5_forensic_audit_document, docs_iccad5_checkpoint_fix_document, docs_iccad5_bottleneck_analysis_document, docs_iccad5_imbalance_ablation_document, docs_iccad5_seed_replication_document, docs_final_iccad5_synthesis_document [EXTRACTED 1.00]
- **Multi-Method XAI Convergence on True Positives** — docs_gradcam_grad_cam, docs_final_research_report_grad_cam_plus_plus, docs_final_research_report_layercam, docs_occlusion_occlusion_sensitivity, concept_raster_geometry_correlation [INFERRED 0.85]

## Communities (7 total, 1 thin omitted)

### Community 0 - "ICCAD5 Imbalance & Bottleneck Investigation"
Cohesion: 0.20
Nodes (10): Balanced Sampling condition: 50/50 HS/NHS batches drawn with replacement (each HS image seen ~53x/epoch). Rationale: eliminates dense_penultimate collapse (0.000% zero-vector) and produces 4 HS-elevated units (AUC 0.97-0.99) never seen under class weighting, Class-Imbalance Ablation: baseline_noweight vs class_weighted vs balanced_sampling. Rationale: isolates loss reweighting (not imbalance per se) as the operative collapse cause; class_weighted shows 85.4% HS zero-vector rate vs 0.000% for the other two conditions in every case-type stratum, Final frozen ICCAD5 checkpoint: xai_cnn_iccad5_balanced_sampling_seed1.keras (seed 101, sha256 9479f07781b0128e19d891b295c99b544f9e6fd2b9a4628533138a47c8085b70, balanced accuracy 0.9743). Rationale: adopted as an explicit project-scope decision, despite the seed-replication document itself recommending a larger 5+ seed sweep before calling it settled, Final ICCAD5 XAI/occlusion/geometry evidence pass on the frozen checkpoint (11 samples: TP=3,TN=3,FP=3,FN=2). Rationale: reproduces the TP-strongest/Grad-CAM-least-reliable-outside-TP pattern; 6/132 geometry rows (18.2% of Grad-CAM rows) flagged is_degenerate_map, all Grad-CAM, all TN, Penultimate-Layer Bottleneck: dense_penultimate Dense(16,ReLU) exact-zero-vector collapse. Rationale: OLD final-epoch checkpoint = 97.6% HS / 4.7% NHS zero-vector rate; best-epoch fix reduced to 78.05% HS / 1.65% NHS; synthetic noise probes collapse 100% in both OLD and NEW checkpoints, Seed Replication: balanced_sampling retrained with 3 independent seeds (101,202,303), baseline_noweight with 2 (101,202). Rationale: balanced_sampling reproduces 0.000% zero-vector rate and 4-7 HS-elevated units in all 3 seeds; baseline_noweight's original collapse-avoidance did NOT replicate (95.1%, 78.0% HS zero-vector rate in the two new seeds), is_degenerate_map finding (final pass): 6/132 geometry rows flagged, all Grad-CAM, all TN, concentrated in 2 of 3 TN samples (NNHSCAD5999_9, NNHSCAD5999_6) — structurally distinct from, and must not be conflated with, the dense_penultimate collapse, BestBalancedAccuracyCheckpoint callback (src/metrics.py): best-epoch checkpoint selection fix, selected epoch 8, balanced accuracy 0.9649 (+2 more)

### Community 1 - "ICCAD5 Ablation & Seed Replication Docs"
Cohesion: 0.56
Nodes (9): Final ICCAD5 XAI Synthesis (balanced_sampling seed1, frozen checkpoint), Explainable Hotspot Detection on ICCAD-12: A Multi-Method XAI Study of a Custom CNN Lithography Hotspot Classifier, ICCAD5 Bottleneck Analysis, ICCAD5 Checkpoint Fix, ICCAD5 Forensic Audit, baseline_noweight condition (best-checkpoint selection, no class weighting, natural batches; balanced accuracy 0.9741, 0.000% zero-vector, seed 42), class_weighted condition (production recipe, class_weight={0:52.73,1:0.505}; 85.366% HS zero-vector rate), ICCAD5 Class-Imbalance Ablation (+1 more)

### Community 2 - "CNN Architecture & ICCAD-12 Dataset"
Cohesion: 0.29
Nodes (7): Binary hotspot/non-hotspot classification stage, CNN (custom lithography hotspot classifier: baseline and XAI-ready architectures), ICCAD5 Class Imbalance (train 26 HS / 2,716 NHS = 104.5:1; test 41 HS / 19,327 NHS = 471:1). Rationale: most extreme imbalance of all 5 ICCAD-12 benchmarks; identified as root contributing factor to the dense_penultimate collapse, ICCAD-12 Contest hotspot-detection benchmark suite (iccad1-iccad5, binary raster PNGs), XAI-ready CNN (src/model_xai.py: conv_early_2, conv_mid_2, conv_final_2, dense_penultimate bottleneck), Baseline CNN (src/model.py, ~12,873 parameters, conv2d_5 target layer, 15x15 final grid), Baseline Model & Pipeline Audit Report (Phase 1)

### Community 3 - "XAI Attribution Methods (Grad-CAM family)"
Cohesion: 0.47
Nodes (6): XAI Attribution stage (Grad-CAM, Grad-CAM++, LayerCAM), Grad-CAM++: higher-order-derivative channel weighting. Rationale: degenerates on 42.1% of the 57-sample set, generally positive geometry correspondence but higher-variance than Grad-CAM, LayerCAM: pixel-wise ReLU-clipped gradient weighting, no GAP step. Rationale: only 8.8% degenerate rate; most consistent occlusion agreement and geometry correspondence across case types, Grad-CAM (Selvaraju et al., ICCV 2017): GAP-weighted class activation mapping on conv_final_2. Rationale: degenerates (flat map) on 31.6% of the 57-sample set, concentrated in TN/FN, due to saturated pre-sigmoid logits driving conv_final_2's gradient to zero and near-cancellation under GAP weighting, XAI Pipeline Audit, XAIEngine (src/xai_engine.py): unified implementation of Grad-CAM, Grad-CAM++, and LayerCAM using pre-sigmoid logits

### Community 4 - "Occlusion, Geometry & TP/TN/FP/FN Pipeline"
Cohesion: 0.40
Nodes (5): Occlusion Sensitivity validation stage, Raster Geometry Correlation stage. Rationale: tests whether attribution overlaps narrow-gap/thin-line/corner raster proxies more than area-matched random control; TP paired-diff median ~0.23-0.24 across all four methods, Grad-CAM disproportionately weak outside TP (0.006-0.044 vs 0.07-0.27 for others), TP/TN/FP/FN Interpretation stage: case-type-stratified reading of attribution/occlusion/geometry evidence, Occlusion-Based Sensitivity Analysis for Lithography Hotspot Detection, Occlusion Sensitivity: gradient-free perturbation method, 32x32 sliding window / 16px stride / baseline 0.0. Rationale: baseline 0.0 justified as the literal exact background pixel value in this strictly binary raster domain

### Community 5 - "Geometry/Occlusion Analysis Reports"
Cohesion: 0.80
Nodes (5): Geometry Analysis Report, Raster-Geometry Correlation: Methodology, TP vs FP Geometry Analysis, XAI Research Audit, Occlusion TN/FN instability finding: Pearson as low as 0.21-0.45 (IoU@top-10% as low as 0.03) under baseline/resolution perturbation for TN samples vs 0.72-0.99 for TP/FP — a real structural finding about model TN behavior, not an implementation defect

## Knowledge Gaps
- **3 isolated node(s):** `baseline_noweight condition (best-checkpoint selection, no class weighting, natural batches; balanced accuracy 0.9741, 0.000% zero-vector, seed 42)`, `class_weighted condition (production recipe, class_weight={0:52.73,1:0.505}; 85.366% HS zero-vector rate)`, `Selvaraju, R.R. et al. (2017), Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization, ICCV 2017, pp.618-626`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 3 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Explainable Hotspot Detection on ICCAD-12: A Multi-Method XAI Study of a Custom CNN Lithography Hotspot Classifier` connect `ICCAD5 Ablation & Seed Replication Docs` to `CNN Architecture & ICCAD-12 Dataset`, `XAI Attribution Methods (Grad-CAM family)`, `Occlusion, Geometry & TP/TN/FP/FN Pipeline`, `Geometry/Occlusion Analysis Reports`, `Grad-CAM Citation`?**
  _High betweenness centrality (0.460) - this node is a cross-community bridge._
- **Why does `ICCAD5 Class-Imbalance Ablation` connect `ICCAD5 Ablation & Seed Replication Docs` to `ICCAD5 Imbalance & Bottleneck Investigation`?**
  _High betweenness centrality (0.190) - this node is a cross-community bridge._
- **Why does `XAI Attribution stage (Grad-CAM, Grad-CAM++, LayerCAM)` connect `XAI Attribution Methods (Grad-CAM family)` to `ICCAD5 Imbalance & Bottleneck Investigation`, `CNN Architecture & ICCAD-12 Dataset`, `Occlusion, Geometry & TP/TN/FP/FN Pipeline`?**
  _High betweenness centrality (0.121) - this node is a cross-community bridge._
- **What connects `baseline_noweight condition (best-checkpoint selection, no class weighting, natural batches; balanced accuracy 0.9741, 0.000% zero-vector, seed 42)`, `class_weighted condition (production recipe, class_weight={0:52.73,1:0.505}; 85.366% HS zero-vector rate)`, `Selvaraju, R.R. et al. (2017), Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization, ICCV 2017, pp.618-626` to the rest of the system?**
  _3 weakly-connected nodes found - possible documentation gaps or missing edges._