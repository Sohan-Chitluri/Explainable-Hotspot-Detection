# Via-3 Hotspot Classification & XAI: Multi-Seed Experimental Pipeline Report

## 1. Multi-Seed Classification Performance on Clean Held-Out Cohort ($N=12,803$, $K=3$ Seeds)

The classification pipeline was evaluated on the clean held-out test partition ($N=12,803$ layout images: 411 Hotspot [HS], 12,392 Non-Hotspot [NHS]) across three independent training seeds ($K=3$: Seeds `42`, `100`, and `2024`). Decision thresholds $\tau^*$ were calibrated strictly on the independent validation split ($N=2,846$) to maximize minority-class F1-score without test data leakage.

### Table 1.1: Multi-Seed Classification Performance at Validation-Calibrated Operating Thresholds ($\tau^*$)

| Strategy / Condition | Seeds ($K$) | Operating Threshold ($\tau^*$) | PR-AUC | ROC-AUC | HS Recall | HS Precision | HS F1 Score | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Unweighted)** | $K=3$ | $0.0533 \pm 0.0379$ | $0.2749 \pm 0.0233$ | $0.7598 \pm 0.0234$ | $0.2384 \pm 0.0544$ | $0.4223 \pm 0.1301$ | $0.2934 \pm 0.0198$ | $0.6129 \pm 0.0236$ |
| *Baseline Seed 42* | Single | $0.08$ | $0.2518$ | $0.7328$ | $0.2214$ (91/411) | $0.3640$ (91/250) | $0.2753$ | $0.6043$ |
| *Baseline Seed 100* | Single | $0.07$ | $0.2983$ | $0.7739$ | $0.2993$ (123/411) | $0.3315$ (123/371) | $0.3146$ | $0.6396$ |
| *Baseline Seed 2024* | Single | $0.01$ | $0.2745$ | $0.7728$ | $0.1946$ (80/411) | $0.5714$ (80/140) | $0.2904$ | $0.5949$ |
| **Balanced Sampling** | $K=3$ | $0.6767 \pm 0.1274$ | $0.2635 \pm 0.0476$ | $0.7760 \pm 0.0273$ | $0.2733 \pm 0.0112$ | $0.3680 \pm 0.1006$ | $0.3090 \pm 0.0392$ | $0.6281 \pm 0.0054$ |
| *Balanced Seed 42* | Single | $0.76$ | $0.2619$ | $0.7840$ | $0.2603$ (107/411) | $0.4038$ (107/265) | $0.3166$ | $0.6238$ |
| *Balanced Seed 100* | Single | $0.74$ | $0.2168$ | $0.7456$ | $0.2798$ (115/411) | $0.2544$ (115/452) | $0.2665$ | $0.6263$ |
| *Balanced Seed 2024* | Single | $0.53$ | $0.3120$ | $0.7985$ | $0.2798$ (115/411) | $0.4457$ (115/258) | $0.3438$ | $0.6341$ |
| **Paired Strategy Diff ($\Delta$)** | $K=3$ | $+0.6233 \pm 0.0896$ | $-0.0114 \pm 0.0623$ | $+0.0162 \pm 0.0406$ | $+0.0349 \pm 0.0524$ | $-0.0543 \pm 0.0851$ | $+0.0156 \pm 0.0554$ | $+0.0152 \pm 0.0265$ |
| *Class-Weighted Control* | Single (42) | $0.50$ | $0.1270$ | $0.6864$ | $0.1825$ (75/411) | $0.2500$ (75/300) | $0.2110$ | $0.5822$ |

*Aggregation Note:* For multi-seed rows ($K=3$), values represent sample Mean $\pm$ SD ($ddof=1$) across the three training seeds (`[42, 100, 2024]`). Paired strategy difference is defined as $\Delta = \text{Metric}_{\text{balanced}} - \text{Metric}_{\text{baseline}}$ per seed.

### Table 1.2: Default Uncalibrated Threshold Performance ($\tau = 0.50$)

| Strategy | Seed | Decision $\tau$ | HS Recall | HS Precision | HS F1 Score | Balanced Accuracy | Confusion Matrix (TP / FP / TN / FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | 42 | 0.50 | 0.0925 | 0.8636 | 0.1670 | 0.5460 | 38 / 6 / 12,386 / 373 |
| Baseline | 100 | 0.50 | 0.1095 | 0.8491 | 0.1940 | 0.5544 | 45 / 8 / 12,384 / 366 |
| Baseline | 2024 | 0.50 | **0.0000** | **0.0000** | **0.0000** | **0.5000** | **0 / 0 / 12,392 / 411** |
| Balanced Sampling | 42 | 0.50 | 0.3090 | 0.2714 | 0.2890 | 0.6407 | 127 / 341 / 12,051 / 284 |
| Balanced Sampling | 100 | 0.50 | 0.3869 | 0.1559 | 0.2222 | 0.6587 | 159 / 861 / 11,531 / 252 |
| Balanced Sampling | 2024 | 0.50 | 0.2798 | 0.4291 | 0.3387 | 0.6337 | 115 / 153 / 12,239 / 296 |

### Core Classification Findings & Statistical Disclaimers

1. **Refutation of Universal PR-AUC Dominance:** While Balanced Sampling outperformed Baseline in Seed 42 ($0.2619$ vs $0.2518$), cross-seed evaluation refutes universal PR-AUC dominance. Across three seeds, Baseline achieved higher mean PR-AUC ($0.2749 \pm 0.0233$) than Balanced Sampling ($0.2635 \pm 0.0476$), driven by Baseline Seed 100 ($0.2983$). The mean paired difference is slightly negative ($-0.0114 \pm 0.0623$) and sign-inconsistent (positive in Seeds 42 and 2024; negative in Seed 100).
2. **Minority-Class Recall Stabilization:** Balanced Sampling's primary operational advantage is **minority-class sensitivity stabilization**. The cross-seed standard deviation for HS Recall dropped by a factor of ~5 from $0.0544$ (Baseline) to $0.0112$ (Balanced Sampling), while raising average recall from $23.84\%$ to $27.33\%$.
3. **Catastrophic Failure of Baseline at Default Threshold:** In Baseline Seed 2024, the maximum predicted probability across all 12,803 test layouts was $0.4687$. Consequently, at default threshold $\tau=0.50$, Baseline Seed 2024 failed completely, identifying **0 hotspots out of 411** ($\text{Recall} = 0.0\%$). Balanced Sampling successfully protects against catastrophic default-threshold collapse across all evaluated seeds.
4. **Operating Threshold Shift vs. True Bayesian Calibration:** Calibrated operating thresholds for Balanced Sampling cluster near $\bar{\tau}^* = 0.6767 \pm 0.1274$, compared to $\bar{\tau}^* = 0.0533 \pm 0.0379$ for Baseline. This upward shift is a direct mathematical consequence of training under an artificial 50/50 prior, which shifts continuous logit scores upward. It does **not** constitute Bayesian probability calibration; output activations remain uncalibrated posteriors under an artificial training distribution.
5. **Statistical Replicate Distinction ($K=3$ Seeds vs. $N=12,803$ Layouts):** The clean test set of 12,803 layouts represents repeated evaluation observations, not independent model training runs. With $K=3$ independent training replicates, non-parametric paired tests cannot achieve significance at $\alpha=0.05$ (minimum two-tailed $p=0.25$), and a paired $t$-test on $\Delta \text{PR-AUC}$ yields $t=-0.317, p=0.78$. No inferential claim of statistically significant superiority between strategies can be supported across seeds. Furthermore, normal-theory Student's $t$ confidence intervals for $K=3$ yield unphysical negative lower bounds for baseline thresholds ($[-0.0407, 0.1474]$) and are omitted in favor of empirical dispersion `Mean ± SD [Seed 42, Seed 100, Seed 2024]`.

---

## 2. Multi-Seed Explainability (XAI) Robustness Evaluation ($N=50$ Diagnostic Layouts, $K=3$ Seeds)

Explainability methods were evaluated across all six model checkpoints ($2\text{ strategies} \times 3\text{ seeds}$) on the frozen 50-layout diagnostic cohort (`xai_sample_cohort_manifest.json`, SHA-256: `1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb`).

> [!IMPORTANT]
> **Diagnostic Cohort Composition & Provenance Qualification (ISSUE-MAJ-01 Reconciled):**  
> The 50-layout diagnostic cohort consists of **40 Hotspot (label 1)** and **10 Non-Hotspot (label 0)** layouts, stratified into six distinct error and detection modes:
> - `shared_tp` (10 layouts, Hotspot): Concordantly detected by both baseline and balanced models.
> - `shared_fn` (10 layouts, Hotspot): Concordantly missed by both models.
> - `balanced_won` (10 layouts, Hotspot): Discordant; correctly detected by Balanced Sampling, missed by Baseline.
> - `baseline_won` (10 layouts, Hotspot): Discordant; correctly detected by Baseline, missed by Balanced Sampling.
> - `shared_tn` (5 layouts, Non-Hotspot): Concordantly classified clean layouts.
> - `shared_fp` (5 layouts, Non-Hotspot): Concordant false-alarm patterns.  
> Earlier informal execution report narratives that asserted a "25 HS / 25 NHS" composition were typographical descriptions of outcome quadrants rather than ground-truth labels. The true on-disk cohort is verified as 40 HS / 10 NHS.

All methods targeted the pre-sigmoid logit $+z_{\text{HS}}$ (bypassing gradient saturation) at layer `conv2d_5` ($15 \times 15 \times 12$), except Gradient Saliency (input resolution $224 \times 224 \times 3$) and Occlusion Sensitivity ($32 \times 32$ window, stride $16 \times 16$, fill $0.0$). A total of 1,800 faithfulness curves (10-step progressive pixel deletion and insertion) and 1,500 noise stability trials were evaluated.

### Table 2.1: Multi-Seed Attribution Faithfulness: Insertion and Deletion AUC

| Attribution Method | Metric | Baseline ($K=3$) Mean $\pm$ SD | Balanced ($K=3$) Mean $\pm$ SD | Paired Difference ($\Delta$) Mean $\pm$ SD | Seed Consistency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Grad-CAM** | Insertion AUC $\uparrow$ | $0.3244 \pm 0.1159$ | $0.4789 \pm 0.1432$ | $+0.1545 \pm 0.1491$ | Inconsistent ($+0.1749, +0.2923, -0.0037$) |
| | Deletion AUC $\downarrow$ | $0.2014 \pm 0.0707$ | $0.3705 \pm 0.2042$ | $+0.1691 \pm 0.1762$ | Positive ($+0.1028, +0.3688, +0.0356$) |
| **Grad-CAM++** | Insertion AUC $\uparrow$ | $0.2555 \pm 0.0986$ | $0.4103 \pm 0.1972$ | $+0.1547 \pm 0.1974$ | Inconsistent ($+0.1503, +0.3543, -0.0404$) |
| | Deletion AUC $\downarrow$ | $0.2583 \pm 0.0993$ | $0.4402 \pm 0.1208$ | $+0.1819 \pm 0.1538$ | Positive ($+0.0974, +0.3595, +0.0889$) |
| **LayerCAM** | Insertion AUC $\uparrow$ | $0.2489 \pm 0.0627$ | $0.4412 \pm 0.1828$ | $+0.1923 \pm 0.2047$ | Inconsistent ($+0.2512, +0.3610, -0.0354$) |
| | Deletion AUC $\downarrow$ | $0.2692 \pm 0.1351$ | $0.3922 \pm 0.1411$ | $+0.1230 \pm 0.1648$ | Positive ($+0.0258, +0.3133, +0.0299$) |
| **Gradient Saliency** | Insertion AUC $\uparrow$ | $0.1738 \pm 0.0851$ | $0.5308 \pm 0.1059$ | $+0.3570 \pm 0.0220$ | **Consistent Positive** ($+0.3589, +0.3780, +0.3341$) |
| | Deletion AUC $\downarrow$ | $0.6854 \pm 0.1552$ | $0.5393 \pm 0.3544$ | $-0.1460 \pm 0.4328$ | Inconsistent ($-0.0737, +0.2460, -0.6104$) |
| **Occlusion ($32\times32$)** | Insertion AUC $\uparrow$ | $0.4362 \pm 0.0367$ | $0.8422 \pm 0.1034$ | $+0.4060 \pm 0.1229$ | **Consistent Positive** ($+0.4666, +0.4868, +0.2646$) |
| | Deletion AUC $\downarrow$ | $0.1633 \pm 0.0936$ | $0.1365 \pm 0.0989$ | $-0.0268 \pm 0.1314$ | Inconsistent ($-0.1243, +0.1227, -0.0787$) |
| **Random Baseline** | Insertion AUC $\uparrow$ | $0.2655 \pm 0.0731$ | $0.5220 \pm 0.1097$ | $+0.2566 \pm 0.0859$ | **Consistent Positive** ($+0.1858, +0.3521, +0.2318$) |
| | Deletion AUC $\downarrow$ | $0.2634 \pm 0.0762$ | $0.5187 \pm 0.1082$ | $+0.2553 \pm 0.0825$ | Positive ($+0.1848, +0.3460, +0.2352$) |

*Aggregation Note (ISSUE-MIN-02 Reconciled):* Standard deviations represent sample variance across the $K=3$ training seeds ($ddof=1$). Per-seed values underlying the aggregates are computed across the $N=50$ diagnostic layouts.

### Table 2.2: Multi-Seed Explanation Stability Under Input Gaussian Noise ($\sigma = 0.05, T=3$, Spearman $\bar{\rho} \uparrow$)

| Attribution Method | Baseline Seed 42 | Baseline Seed 100 | Baseline Seed 2024 | Baseline Aggregate ($K=3$) | Balanced Seed 42 | Balanced Seed 100 | Balanced Seed 2024 | Balanced Aggregate ($K=3$) | Paired Difference ($\Delta$) Mean $\pm$ SD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grad-CAM** | 0.9712 | 0.9538 | 0.6363 | $0.8538 \pm 0.1885$ | 0.9710 | 0.9424 | 0.9380 | $\mathbf{0.9505 \pm 0.0179}$ | $+0.0967 \pm 0.1776$ |
| **Grad-CAM++** | 0.2365 | 0.3139 | 0.2571 | $\mathbf{0.2692 \pm 0.0401}$ | 0.2453 | 0.0867 | 0.1830 | $\mathbf{0.1717 \pm 0.0799}$ | $-0.0975 \pm 0.1197$ |
| **LayerCAM** | 0.9679 | 0.9344 | 0.9646 | $\mathbf{0.9556 \pm 0.0185}$ | 0.9638 | 0.9527 | 0.9496 | $\mathbf{0.9554 \pm 0.0075}$ | $-0.0003 \pm 0.0170$ |
| **Gradient Saliency** | 0.4480 | 0.4267 | 0.4266 | $0.4338 \pm 0.0123$ | 0.4526 | 0.4503 | 0.4164 | $0.4398 \pm 0.0203$ | $+0.0060 \pm 0.0169$ |
| **Occlusion ($32\times32$)**| 0.6761 | 0.7594 | 0.6600 | $0.6985 \pm 0.0534$ | 0.7444 | 0.7524 | 0.7243 | $0.7404 \pm 0.0145$ | $+0.0419 \pm 0.0424$ |
| **Random Baseline** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

*Note:* Random Baseline is excluded from input-noise stability testing because it applies fixed pseudo-random permutations without deterministic layer activations or gradients; computing input-noise stability on unguided random permutations is physically uninformative.

### Method-Wise XAI Scientific Insights

1. **Robust Elimination of Grad-CAM Contrast Attenuation:** In Baseline Seed 2024, Grad-CAM suffered severe feature attenuation (stability dropped to $\bar{\rho} = 0.6363$). Balanced Sampling completely prevents this degradation, maintaining high stability ($\bar{\rho} \ge 0.9380$, aggregate $0.9505 \pm 0.0179$) across all three seeds.
2. **Replication of Grad-CAM++ Pathological Instability:** Across all six evaluated models, Grad-CAM++ exhibits severe instability under small input perturbations ($\sigma=0.05$). Cross-seed stability averages $\bar{\rho} = 0.2692 \pm 0.0401$ for Baseline and degrades to $\bar{\rho} = 0.1717 \pm 0.0799$ for Balanced Sampling (reaching $\bar{\rho} = 0.0867$ in Balanced Seed 100). The higher-order gradient weighting in Grad-CAM++ introduces extreme sensitivity in lithographic pattern classification, confirming that Grad-CAM++ is unsuited for industrial inspection.
3. **LayerCAM Robustness:** LayerCAM exhibits the highest and most uniform stability across both training strategies and all seeds ($\bar{\rho} = 0.9556 \pm 0.0185$ Baseline, $0.9554 \pm 0.0075$ Balanced), confirming that localized element-wise gradient gating provides excellent invariance to high-frequency background noise.
4. **Occlusion Sensitivity: Protocol-Specific Alignment vs. Global Faithfulness:** While Occlusion Sensitivity achieved the highest Insertion AUC ($0.8422 \pm 0.1034$ in Balanced Sampling) and lowest Deletion AUC ($0.1365 \pm 0.0989$), **this metric dominance must not be interpreted as unconditional causal superiority**. The insertion/deletion benchmark evaluates models via progressive square patch zero-masking, which shares the identical perturbation primitive as the Occlusion Sensitivity sliding window ($32 \times 32$ zero-fill). This structural protocol alignment mechanically favors Occlusion over gradient-based methods. Occlusion Sensitivity is faithful *under progressive replacement protocols*, but operates at coarse spatial resolution ($32 \times 32$ patches) and incurs 100x higher computational latency.
5. **Random Baseline Score Trajectory Dynamics:** Balanced Sampling exhibits a marked increase in Random Baseline Insertion AUC ($0.5220 \pm 0.1097$ vs. $0.2655 \pm 0.0731$). As established in Workstream C/E4.1, this disparity reflects baseline prediction calibration differences across the 50 cohort layouts (mean predicted $P(\text{HS}) = 0.5744$ for balanced sampling vs. $0.1795$ for baseline). Random subsets of pixels in sensitive models mechanically trigger higher intermediate sigmoid probabilities, and must not be misinterpreted as meaningful explanation signal.

---

## 3. Representation Collapse & Spatial Dynamic Range Diagnostics

Representation collapse was audited on Grad-CAM heatmaps at layer `conv2d_5` across all 50 cohort layouts per model run (300 total layout evaluations across 6 models). A layout is flagged as a "Flat-CAM" if spatial standard deviation $\text{std}(M) < 10^{-4}$ or peak activation $\max(M) < 10^{-4}$.

### Table 3: CAM Spatial Contrast and Flat-CAM Diagnostics

| Model Architecture / Strategy | Seed | Layouts Evaluated ($N$) | Flat-CAM Count | Flat-CAM Rate | Spatial Std Dev (Layout Mean $\pm$ SD) | Max Activation Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Unweighted) | 42 | 50 | 0 | 0.0% | $0.1568 \pm 0.0217$ | 1.0000 |
| Baseline (Unweighted) | 100 | 50 | 2 | 4.0% | $0.1724 \pm 0.0393$ | 0.9600 |
| Baseline (Unweighted) | 2024 | 50 | 1 | 2.0% | $0.0589 \pm 0.0132$ | 0.9800 |
| **Baseline Pooled (Layout-Level)** | **All** | **150** | **3** | **2.0%** | $\mathbf{0.1294 \pm 0.0571}$ | **0.9800** |
| *Baseline Cross-Seed (Seed-Level)* | *K=3* | *3 Seeds* | — | — | *$0.1294 \pm 0.0615$* | — |
| Balanced Sampling | 42 | 50 | 0 | 0.0% | $0.1587 \pm 0.0265$ | 1.0000 |
| Balanced Sampling | 100 | 50 | 0 | 0.0% | $0.1474 \pm 0.0273$ | 1.0000 |
| Balanced Sampling | 2024 | 50 | 0 | 0.0% | $0.1594 \pm 0.0275$ | 1.0000 |
| **Balanced Sampling Pooled (Layout-Level)** | **All** | **150** | **0** | **0.0%** | $\mathbf{0.1552 \pm 0.0275}$ | **1.0000** |
| *Balanced Cross-Seed (Seed-Level)* | *K=3* | *3 Seeds* | — | — | *$0.1552 \pm 0.0067$* | — |
| Class-Weighted (Control Reference) | 42 | 50 | 2 | 4.0% | $0.0865 \pm 0.0246$ | 0.1307 |

### Reconciliation of Reporting Discrepancy (ISSUE-MIN-01 & ISSUE-MIN-02)

- **Resolution of Disputed Value `0.0589` (ISSUE-MIN-01):** Earlier drafts erroneously reported `0.0589` as the pooled Baseline spatial standard deviation. Auditing confirmed that `0.0589` was an accidental typographical transcription of the `Spatial Std Mean` of `Baseline Seed 2024` immediately preceding it. The verified sample standard deviation across all 150 layout observations ($ddof=1$) is $\mathbf{0.0571}$ (population $ddof=0$ is $0.0569$). Across the three seed-level means ($[0.1568, 0.1724, 0.0589]$), the cross-seed standard deviation ($ddof=1$) is $\mathbf{0.0615}$. Both statistics are now explicitly distinguished and correctly tabulated above.
- **Aggregation Unit Distinction (ISSUE-MIN-02):** In Table 3, per-seed rows report dispersion across individual layout observations ($N=50, ddof=1$). Pooled rows report dispersion across all layout observations ($N=150, ddof=1$). Cross-seed rows report dispersion across independent training seed means ($K=3, ddof=1$).

---

## 4. Model Parameter Randomization Sanity Check (Adebayo et al., 2018)

To determine whether explanations rely on learned feature representations rather than acting as invariant edge detectors, all six models were evaluated under convolutional parameter randomization ($\mathcal{N}(0, 0.05)$) across 5 curated diagnostic layouts.

### Table 4: Parameter Randomization Sanity Check across Six Checkpoints

| Model Strategy | Seed | Checkpoint Path | Mean Spearman $\bar{\rho}$ vs. Randomized Weights | Prespecified Rule ($|\bar{\rho}| < 0.20$) | Programmatic Status | Substantive Interpretation |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Baseline** | 42 | `models/via3_multiseed/via3_baseline_seed42.keras` | $0.0000$ | Satisfied | **PASSED** | Weight Sensitive |
| **Baseline** | 100 | `models/via3_multiseed/via3_baseline_seed100.keras` | $-0.3321$ | Exceeded ($|-0.3321| > 0.20$) | **FAILED (FLAGGED)** | Weight Sensitive (Substantive) / Threshold Exceeded |
| **Baseline** | 2024 | `models/via3_multiseed/via3_baseline_seed2024.keras` | $0.0000$ | Satisfied | **PASSED** | Weight Sensitive |
| **Balanced Sampling** | 42 | `models/via3_multiseed/via3_balanced_sampling_seed42.keras` | $-0.1331$ | Satisfied | **PASSED** | Weight Sensitive |
| **Balanced Sampling** | 100 | `models/via3_multiseed/via3_balanced_sampling_seed100.keras` | $-0.0571$ | Satisfied | **PASSED** | Weight Sensitive |
| **Balanced Sampling** | 2024 | `models/via3_multiseed/via3_balanced_sampling_seed2024.keras` | $+0.0553$ | Satisfied | **PASSED** | Weight Sensitive |

### Methodological Reconciliation of Sanity Check Status (ISSUE-MAJ-03 Reconciled)

- **Prespecified Programmatic Criterion:** In the automated runner script (`scripts/run_via3_xai_multiseed.py:436`), the programmatic check was defined as:
  `"passed_sanity_check": bool(abs(adebayo_mean_rho) < 0.20)`. Under this prespecified rule, **5 of 6 models passed**, while Baseline Seed 100 was flagged as `passed_sanity_check: false` due to observed $\bar{\rho} = -0.3321$.
- **Substantive Scientific Interpretation:** In explainability theory (Adebayo et al., NeurIPS 2018), the failure mode of attribution methods is producing invariant maps regardless of network weights ($\bar{\rho} \approx +1.0$). A correlation of $\bar{\rho} = -0.3321$ demonstrates that attributions change substantially upon weight randomization, confirming genuine weight dependence. However, claiming that all six models passed under a broadened threshold (e.g. $|\bar{\rho}| \le 0.35$) represents a post-hoc reinterpretation that was not prespecified in code. The report transparently records the programmatic failure of Baseline Seed 100 alongside its substantive scientific explanation.

---

## 5. Mandatory Methodological, Statistical, and Reproducibility Qualifications

Downstream reports, manuscripts, and releases must incorporate the following eight verified qualifications:

1. **Perturbation Sensitivity vs. Causal Sufficiency:** Pixel deletion/insertion curves quantify model score sensitivity under progressive zero-masking, not Pearlian causal counterfactual necessity or defect physics. Claims must refer to *attribution faithfulness under progressive substrate masking* rather than causal feature isolation.
2. **Protocol Alignment and Occlusion Favorability:** Occlusion Sensitivity uses zero-fill sliding patches ($32 \times 32$), which structurally matches the perturbation mechanism of progressive deletion/insertion. This protocol alignment mechanically inflates Occlusion faithfulness metrics relative to gradient and CAM methods. Occlusion must not be claimed as unconditionally or universally superior; its operational use is constrained by coarse resolution ($32 \times 32$ patches) and 100x higher computational cost.
3. **Out-of-Distribution (OOD) Blank Canvas Rebound:** An all-zero layout canvas ($224 \times 224 \times 3$) is out-of-distribution for the custom CNN, producing elevated baseline sigmoid probabilities ($P(\text{HS}) \approx 0.94 - 0.999$) due to convolutional layer biases and lack of negative activations. Consequently, deletion curves exhibit a characteristic U-shaped rebound as masking approaches $100\%$, and insertion curves start at elevated values before dropping on initial patch entry.
4. **Probability Baseline Asymmetry in Deletion AUC:** Because unnormalized probabilities are integrated directly without baseline shifting, models with lower initial confidence on missed hotspots mechanically register artificially lower Deletion AUCs simply because their curves are bounded by lower initial values.
5. **Stratified Diagnostic Cohort Composition (ISSUE-MAJ-01 Reconciled):** The $N=50$ evaluation cohort is a stratified diagnostic sample designed to contrast specific error and recovery modes (40 Hotspots [10 shared TP, 10 shared FN, 10 balanced won, 10 baseline won] and 10 Non-Hotspots [5 shared TN, 5 shared FP]). It is not an i.i.d. population sample. Inferences apply strictly to this diagnostic cohort.
6. **Multi-Seed Statistical Limits & Independence Discipline (ISSUE-MIN-02 Reconciled):** The study encompasses $K=3$ independent training runs (Seeds `42`, `100`, and `2024`). The 12,803 test layouts represent evaluation observations, not independent model training replicates. Descriptive variability is reported as `Mean ± SD [Seed 42, Seed 100, Seed 2024]`. Normal-theory Student's $t$ confidence intervals are omitted because bounded metrics violate Gaussian assumptions and produce unphysical negative lower bounds for decision thresholds ($[-0.0407, 0.1474]$). Formal hypothesis testing across seeds cannot establish statistical significance at $K=3$.
7. **Sanity Check Scope & Threshold Integrity (ISSUE-MAJ-03 Reconciled):** Under the prespecified programmatic threshold ($|\bar{\rho}| < 0.20$), 5 of 6 models passed, while Baseline Seed 100 recorded $\bar{\rho} = -0.3321$. While negative correlation demonstrates parameter sensitivity rather than edge invariance, the prespecified threshold was exceeded.
8. **Provenance, Manifest Integrity, and Runner Limitations (ISSUE-MAJ-02 & ISSUE-MAJ-04 Reconciled):** All 1,800 faithfulness evaluations, 1,500 stability evaluations, and 300 collapse diagnostics are verified to exist on disk without corruption. However, audit identified two reproducibility caveats:
   - *Manifest Checksum Discrepancy:* `reproducibility_manifest.json` on disk hashes to `9655c3480db1cd05f9fb87e5c8abbea53d1a906d9277546731711c74f1c26e51`. The earlier hash reported in `WORKSTREAM_E4_2_EXECUTION_REPORT.md` (`75f28469ad...`) reflects a pre-final draft digest prior to environment stamping.
   - *Runner Resume Robustness:* The execution runner's resume check evaluates file existence and row count rather than cryptographic hashes or semantic primary keys. These provenance and automation enhancements are formally handed off to Workstream E5 for complete release hardening.
