#!/usr/bin/env python3
"""
Comprehensive Penultimate Representation Diagnostic for ICCAD-5.
Extracts pre- and post-activations, classification metrics, and layer statistics
for all test samples.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "xai" / "retrained_clean" / "xai_cnn_iccad5.keras"
OUT_CSV = ROOT / "results" / "iccad5_research" / "sample_diagnostics.csv"

def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    
    conv_layer = model.get_layer("conv_final_2")
    act_final_layer = model.get_layer("act_final")
    dropout_layer = model.get_layer("dropout")
    pen_layer = model.get_layer("dense_penultimate")
    out_layer = model.get_layer("dense_output")

    pen_W, pen_b = pen_layer.get_weights()
    out_W, out_b = out_layer.get_weights()

    submodel = Model(
        inputs=model.input,
        outputs=[
            conv_layer.output,
            act_final_layer.output,
            dropout_layer.output,
            pen_layer.output,
            model.output
        ]
    )

    # 41 HS
    hs_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
        str(ROOT / "iccad-official" / "iccad5" / "test"),
        classes=["test_hs"],
        batch_size=41,
        shuffle=False,
        target_size=(224, 224),
        class_mode=None
    )
    # 500 NHS
    nhs_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
        str(ROOT / "iccad-official" / "iccad5" / "test"),
        classes=["test_nhs"],
        batch_size=500,
        shuffle=False,
        target_size=(224, 224),
        class_mode=None
    )

    hs_imgs = next(hs_gen)
    nhs_imgs = next(nhs_gen)

    hs_outs = submodel.predict(hs_imgs, verbose=0)
    nhs_outs = submodel.predict(nhs_imgs, verbose=0)

    def process(outs, filenames, label_name):
        conv_f, act_f, drop_f, pen_f, final_p = outs
        pre_act = np.dot(drop_f, pen_W) + pen_b
        records = []
        for i in range(len(filenames)):
            is_zero = bool(np.all(pen_f[i] == 0.0))
            prob = float(final_p[i, 0])
            logit = float(np.dot(pen_f[i], out_W) + out_b)
            pred = 0 if prob < 0.5 else 1
            true_l = 0 if label_name == "HS" else 1
            records.append({
                "filename": filenames[i],
                "class": label_name,
                "true_label": true_l,
                "pred_label": pred,
                "is_correct": (pred == true_l),
                "prob": prob,
                "logit": logit,
                "is_exact_zero": is_zero,
                "pen_norm": float(np.linalg.norm(pen_f[i])),
                "pre_act_min": float(np.min(pre_act[i])),
                "pre_act_max": float(np.max(pre_act[i])),
                "pre_act_mean": float(np.mean(pre_act[i])),
                "conv_norm": float(np.linalg.norm(conv_f[i])),
                "act_f_norm": float(np.linalg.norm(act_f[i])),
            })
        return records

    records = process(hs_outs, hs_gen.filenames[:41], "HS") + process(nhs_outs, nhs_gen.filenames[:500], "NHS")
    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"Diagnostics written to {OUT_CSV}")

if __name__ == "__main__":
    main()
