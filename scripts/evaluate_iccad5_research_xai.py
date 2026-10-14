#!/usr/bin/env python3
"""
Evaluate XAI Heatmaps and Faithfulness Curves on ICCAD-5.
Generates metrics for collapsed vs non-collapsed samples and balanced model.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from src.xai_engine import XAIEngine
from src.faithfulness import compute_deletion_insertion_curves

ROOT = Path(__file__).resolve().parents[1]
CLEAN_MODEL_PATH = ROOT / "models" / "xai" / "retrained_clean" / "xai_cnn_iccad5.keras"
BALANCED_MODEL_PATH = ROOT / "models" / "xai" / "experimental" / "xai_cnn_iccad5_balanced_sampling_seed1.keras"
DIAG_CSV = ROOT / "results" / "iccad5_research" / "sample_diagnostics.csv"
OUT_XAI_CSV = ROOT / "results" / "iccad5_research" / "xai_collapse_evaluation.csv"
OUT_FAITH_CSV = ROOT / "results" / "iccad5_research" / "faithfulness_comparison.csv"

def main():
    m_clean = tf.keras.models.load_model(str(CLEAN_MODEL_PATH), compile=False)
    eng_clean = XAIEngine(m_clean, target_layer_name="conv_final_2")
    
    diag = pd.read_csv(DIAG_CSV)
    hs_samples = diag[diag["class"] == "HS"]
    
    # 1. Full CAM evaluation on 41 HS
    cam_records = []
    for idx, row in hs_samples.iterrows():
        fn = row["filename"]
        img_t, _ = eng_clean.preprocess_image(ROOT / "iccad-official" / "iccad5" / "test" / fn)
        is_col = bool(row["is_exact_zero"])
        
        _, gcam_res, _ = eng_clean.explain_gradcam(img_t, target_class="HS")
        _, gcampp_res, _ = eng_clean.explain_gradcam_plus_plus(img_t, target_class="HS")
        _, lcam_res, _ = eng_clean.explain_layercam(img_t, target_class="HS")
        
        cam_records.append({
            "filename": fn,
            "is_collapsed": is_col,
            "prob": row["prob"],
            "gcam_is_zero": bool(np.all(gcam_res == 0.0)),
            "gcam_var": float(np.var(gcam_res)),
            "gcampp_is_zero": bool(np.all(gcampp_res == 0.0)),
            "gcampp_var": float(np.var(gcampp_res)),
            "lcam_is_zero": bool(np.all(lcam_res == 0.0)),
            "lcam_var": float(np.var(lcam_res)),
        })
    pd.DataFrame(cam_records).to_csv(OUT_XAI_CSV, index=False)
    print(f"XAI evaluations written to {OUT_XAI_CSV}")

    # 2. Faithfulness curves on subset
    col_samples = hs_samples[hs_samples["is_exact_zero"]].head(5)
    non_col_samples = hs_samples[~hs_samples["is_exact_zero"]].head(5)
    
    faith_records = []
    for subset, label in [(col_samples, "collapsed"), (non_col_samples, "non_collapsed")]:
        for idx, row in subset.iterrows():
            fn = row["filename"]
            img_t, _ = eng_clean.preprocess_image(ROOT / "iccad-official" / "iccad5" / "test" / fn)
            curves = compute_deletion_insertion_curves(m_clean, eng_clean, img_t, target_idx=0, steps=10)
            for m in ["gradcam", "gradcam_plus_plus", "layercam"]:
                faith_records.append({
                    "filename": fn,
                    "status": label,
                    "method": m,
                    "deletion_auc": curves[m]["deletion_auc"],
                    "insertion_auc": curves[m]["insertion_auc"]
                })
    pd.DataFrame(faith_records).to_csv(OUT_FAITH_CSV, index=False)
    print(f"Faithfulness metrics written to {OUT_FAITH_CSV}")

if __name__ == "__main__":
    main()
