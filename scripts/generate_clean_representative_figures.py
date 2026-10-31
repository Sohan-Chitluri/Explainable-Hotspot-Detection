"""
Generate Clean Representative Attribution Visualizations (TP, TN, FP, FN)
across all five ICCAD-12 benchmarks from retrained clean checkpoints
with verified provenance.
"""

import glob
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from src.xai_engine import XAIEngine


def load_and_pred(model, path_str: str):
    img = Image.open(path_str).convert("RGB").resize((224, 224), Image.Resampling.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(arr, axis=0)
    p_nhs = float(model(x, training=False).numpy().flatten()[0])
    p_hs = float(1.0 - p_nhs)
    return p_hs, p_nhs, arr, x


def find_benchmark_cases(model, benchmark: str):
    """Deterministically find the first alphabetical TP, FN, TN, FP cases."""
    hs_files = sorted(glob.glob(f"iccad-official/{benchmark}/test/test_hs/*"))
    nhs_files = sorted(glob.glob(f"iccad-official/{benchmark}/test/test_nhs/*"))

    tp_case, fn_case, tn_case, fp_case = None, None, None, None

    # Search HS files for TP (>=0.5) and FN (<0.5)
    for p in hs_files:
        p_hs, _, _, _ = load_and_pred(model, p)
        if p_hs >= 0.5 and tp_case is None:
            tp_case = ("TP", p, 0)
        elif p_hs < 0.5 and fn_case is None:
            fn_case = ("FN", p, 0)
        if tp_case and fn_case:
            break

    # Search NHS files for TN (<0.5) and FP (>=0.5)
    for p in nhs_files:
        p_hs, _, _, _ = load_and_pred(model, p)
        if p_hs < 0.5 and tn_case is None:
            tn_case = ("TN", p, 1)
        elif p_hs >= 0.5 and fp_case is None:
            fp_case = ("FP", p, 0)
        if tn_case and fp_case:
            break

    return [tp_case, fn_case, tn_case, fp_case]


def generate_all_clean_figures():
    out_dir = Path("results/xai_clean_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    benchmarks = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]

    for bench in benchmarks:
        ckpt_path = Path(f"models/xai/retrained_clean/xai_cnn_{bench}.keras")
        if not ckpt_path.exists():
            print(f"Skipping {bench}: checkpoint not found at {ckpt_path}")
            continue

        print(f"Generating clean figures for {bench}...")
        model = tf.keras.models.load_model(ckpt_path, compile=False)
        engine = XAIEngine(model)

        cases = find_benchmark_cases(model, bench)

        for case_info in cases:
            if case_info is None:
                continue
            case_type, path_str, target_class = case_info
            p_hs, p_nhs, arr, x = load_and_pred(model, path_str)

            # Generate attribution maps
            _, gcam, _ = engine.explain_gradcam(x, target_class=target_class)
            _, gcam_pp, _ = engine.explain_gradcam_plus_plus(x, target_class=target_class)
            _, layercam, _ = engine.explain_layercam(x, target_class=target_class)

            # Plot 1x4 comparison figure
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(arr)
            axes[0].set_title(f"Input ({bench.upper()} {case_type})\np(HS)={p_hs:.3f}")
            axes[0].axis("off")

            axes[1].imshow(arr)
            axes[1].imshow(gcam, cmap="jet", alpha=0.5)
            axes[1].set_title("Grad-CAM")
            axes[1].axis("off")

            axes[2].imshow(arr)
            axes[2].imshow(gcam_pp, cmap="jet", alpha=0.5)
            axes[2].set_title("Grad-CAM++ (Eq 21)")
            axes[2].axis("off")

            axes[3].imshow(arr)
            axes[3].imshow(layercam, cmap="jet", alpha=0.5)
            axes[3].set_title("LayerCAM")
            axes[3].axis("off")

            out_path = out_dir / f"{bench}_{case_type}_clean_comparison.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()

            # Save lossless float32 numpy arrays
            np.save(out_dir / f"{bench}_{case_type}_gradcam.npy", gcam)
            np.save(out_dir / f"{bench}_{case_type}_gradcam_pp.npy", gcam_pp)
            np.save(out_dir / f"{bench}_{case_type}_layercam.npy", layercam)

            manifest.append({
                "benchmark": bench,
                "case_type": case_type,
                "source_image": path_str,
                "checkpoint": str(ckpt_path),
                "p_hs": float(p_hs),
                "figure_path": str(out_path),
                "arrays": {
                    "gradcam": str(out_dir / f"{bench}_{case_type}_gradcam.npy"),
                    "gradcam_pp": str(out_dir / f"{bench}_{case_type}_gradcam_pp.npy"),
                    "layercam": str(out_dir / f"{bench}_{case_type}_layercam.npy"),
                }
            })

    with open(out_dir / "provenance_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully generated clean figures for {len(manifest)} cases across {len(benchmarks)} benchmarks in results/xai_clean_figures/")


if __name__ == "__main__":
    generate_all_clean_figures()
