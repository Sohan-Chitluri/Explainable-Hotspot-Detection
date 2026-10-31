# Balanced-Sampling Replication Audit — Agent G

**Audit Date:** 2026-09-20  
**Scope:** Audit of existing balanced-sampling replication (seeds 101, 202, 303) and evaluation of proposed extension (seeds 404, 505, 606). Based on `docs/ICCAD5_SEED_REPLICATION.md`, `results/xai/iccad5_seed_replication/`, and `docs/IMPROVEMENT_METHODOLOGY_AUDIT.md`.

---

## 1. Existing Balanced-Sampling Replication (3 Seeds)

### 1.1 Implementation (Verified from source)

**Script:** `src/run_iccad5_seed_replication.py`

| Parameter | Value |
|-----------|-------|
| Seeds tested | 101, 202, 303 (balanced_sampling); 101, 202 (baseline_noweight) |
| Sampler | Balanced 50/50 with replacement: 16 HS + 16 NHS per step × 86 steps/epoch |
| HS draws/epoch | 1,376 (≈53× per HS image; 26 HS images) |
| NHS draws/epoch | 1,376 (≈0.5× per NHS image; 2,716 NHS images) |
| RNG | `np.random.default_rng(seed)` per condition |
| Paths | Sorted `glob('*')` from train_hs/train_nhs directories |
| Class weights | None (`class_weight=None`) |
| Checkpoint selection | `BestBalancedAccuracyCheckpoint` on **test set** (validation_data = test) |
| Architecture | XAI-ready CNN (`build_xai_cnn`) |
| Optimizer | Nadam (default lr=0.001) |
| Epochs | 10 |
| Batch size | 32 |

### 1.2 Replicated Results (from stored metrics)

| Metric | Seed 101 | Seed 202 | Seed 303 | Mean ± SD |
|--------|----------|----------|----------|-----------|
| Balanced accuracy | 0.9743 | 0.9731 | 0.9741 | 0.9738 ± 0.0006 |
| HS recall | 0.9512 | 0.9512 | 0.9512 | 0.9512 |
| NHS specificity | 0.9974 | 0.9951 | 0.9970 | 0.9965 ± 0.0012 |
| Precision | 0.6170 | 0.6056 | 0.6127 | 0.6118 ± 0.0057 |
| F1 | 0.7559 | 0.7431 | 0.7502 | 0.7497 ± 0.0064 |
| **HS zero-vector rate** | **0.000%** | **0.000%** | **0.000%** | **0.000%** |
| **NHS zero-vector rate** | **0.000%** | **0.000%** | **0.000%** | **0.000%** |
| Dead units | 3 | 4 | 3 | 3.3 ± 0.6 |
| HS-elevated units (AUC≥0.90) | 4 | 4 | 7 | 5.0 ± 1.7 |

**Checkpoint SHA256s (from `results/xai/iccad5_seed_replication/CHECKPOINT_CHECKSUMS.txt`):**
- seed101: `9479f07781b0128e19d891b295c99b544f9e6fd2b9a4628533138a47c8085b70` (adopted final)
- seed202: `8d7abdc8799eec9a0bccf376e867ac90911131c2cbbab02c32f4f1627d9f218f`
- seed303: `c7da62143fba626f432f63a6b7b440344db7a674e63b9e93f0386b2f54c090e3`

### 1.3 Initialization Checksums (from `metrics.json`)

| Seed | Init Weight Checksum |
|------|---------------------|
| 101 | 130,412.3 |
| 202 | 130,412.3 |
| 303 | 130,412.3 |

**Issue:** All three seeds report **identical** initialization checksum. This suggests either:
- The checksum (`np.sum(np.abs(w))`) is insufficiently discriminative, or
- The seeds did not actually produce different initializations despite `tf.keras.utils.set_random_seed(seed)` + `np.random.default_rng(seed)`

**Verification needed:** Full weight tensor SHA256 comparison before claiming true seed diversity.

---

## 2. Baseline No-Weight Replication (2 Seeds)

| Metric | Seed 101 | Seed 202 |
|--------|----------|----------|
| Balanced accuracy | 0.9741 | 0.9707 |
| HS zero-vector rate | 95.1% | 78.0% |
| Dead units | 6 | 8 |
| HS-elevated units | 0 | 0 |

**Conclusion:** Baseline no-weight collapse-avoidance **failed to replicate** — both new seeds show high HS zero-vector rates similar to class-weighted condition. Only seed 42 (original ablation) avoided collapse.

---

## 3. Proposed Extension: Seeds 404, 505, 606

### 3.1 Registry Status
- **Agent G** (this audit): `IMPROVEMENT_BALANCED_REPLICATION.md` — completed
- **Training execution**: PENDING_LEAD_APPROVAL (test-selection leakage mitigation per Agent A in progress)
- **Registry entry**: `docs/IMPROVEMENT_TASK_REGISTRY.md` — status "audit/preparation, no training"

### 3.2 Extension Protocol (Predeclared)

If GPU restored and training protocol corrected (Agent A/H):

| Condition | Seeds | Protocol |
|-----------|-------|----------|
| `balanced_nowt` | 404, 505, 606 | Same as seeds 101/202/303: balanced sampler, no class weights, corrected validation split, proper RNG |
| **Key requirement** | Full weight tensor SHA256 recorded pre-training; must differ from seeds 101/202/303 |

### 3.3 Caveats from Existing Evidence

1. **Initialization checksum collision** — current checksum may not detect seed differences; need cryptographic weight SHA256.
2. **Test-selected checkpoint** — all existing seeds selected best epoch on test set; corrected protocol must use disjoint validation.
3. **Sample size** — ICCAD5 has only 26 HS training images; balanced sampler draws each ~53×/epoch with replacement. With only 3–6 seeds, low-frequency failure modes cannot be ruled out (Agent A: 3 seeds → binomial 95% upper bound ~39% on failure rate).
4. **Unit index non-replication** — HS-elevated unit indices permute across seeds (0/16 shared), but *existence* of 4–7 such units replicates.
5. **No class-weighted replication** — Agent H audit shows class-weighted never replicated (only seed 42).

---

## 4. Recommendation

**Do NOT execute seeds 404/505/606 under current protocol.**

Required preconditions:
1. ✅ Fix training protocol (Agent A/H): disjoint validation split, test set held out entirely
2. ✅ Verify initialization SHA256 differentiates seeds (current checksum collides)
3. ✅ GPU availability restored (`nvidia-smi` exit 0)
4. ✅ Lead approval of corrected protocol and compute budget

If all preconditions met, the 3-condition × 3-seed (9 runs) design from Agent H is the minimum corrected experiment. Seeds 404/505/606 would be a *separate robustness extension*, not a paired comparison.

---

## 5. Files

- Source: `docs/ICCAD5_SEED_REPLICATION.md`
- Source: `results/xai/iccad5_seed_replication/`
- This document: `docs/IMPROVEMENT_BALANCED_REPLICATION.md`

*Audit performed by Agent G (CPU-only). No training executed. All data from stored metrics JSON/CSV.*