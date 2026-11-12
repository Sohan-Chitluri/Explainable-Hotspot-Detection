"""
Run comprehensive Faithfulness, Sanity Checks, and Seed Stability Studies.

Evaluates clean models on representative test cases across ICCAD-12 benchmarks.
Saves results to results/xai_faithfulness/.
"""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import glob
from scipy.stats import spearmanr

from src.xai_engine import XAIEngine
from src.faithfulness import compute_deletion_insertion_curves, run_model_parameter_randomization_check


def load_test_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    out_dir = Path("results/xai_faithfulness")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("XAI Faithfulness, Sanity Checks, and Seed Stability Sprint")
    print("================================================================================")

    # 1. Faithfulness (Deletion / Insertion AUC) across benchmarks
    benchmarks = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]
    faithfulness_results = {}
    
    for bench in benchmarks:
        ckpt_path = Path(f"models/xai/retrained_clean/xai_cnn_{bench}.keras")
        if not ckpt_path.exists():
            continue
        print(f"\nEvaluating Faithfulness on {bench}...")
        model = tf.keras.models.load_model(ckpt_path, compile=False)
        engine = XAIEngine(model)

        # Select up to 4 TP (HS test) samples and 2 TN (NHS test) samples
        hs_files = sorted(glob.glob(f"iccad-official/{bench}/test/test_hs/*"))[:4]
        bench_samples = []

        for p in hs_files:
            img_tensor = load_test_image(p)
            curves = compute_deletion_insertion_curves(
                model=model,
                engine=engine,
                img_tensor=img_tensor,
                target_idx=0,
                steps=10,
                methods=["gradcam", "gradcam_plus_plus", "layercam"],
            )
            bench_samples.append({
                "sample_path": p,
                "target_class": "HS",
                "curves": curves,
            })
        
        # Aggregate AUCs
        method_summary = {}
        for m in ["gradcam", "gradcam_plus_plus", "layercam"]:
            del_aucs = [s["curves"][m]["deletion_auc"] for s in bench_samples]
            ins_aucs = [s["curves"][m]["insertion_auc"] for s in bench_samples]
            method_summary[m] = {
                "mean_deletion_auc": float(np.mean(del_aucs)),
                "mean_insertion_auc": float(np.mean(ins_aucs)),
            }
        
        faithfulness_results[bench] = {
            "num_samples": len(bench_samples),
            "summary": method_summary,
            "samples": bench_samples,
        }
        print(f"{bench} Faithfulness Summary: {method_summary}")

    # 2. Sanity Checks (Parameter Randomization Test)
    print("\nEvaluating Parameter Randomization Sanity Checks...")
    sanity_results = {}
    for bench in ["iccad1", "iccad2", "iccad5"]:
        ckpt_path = Path(f"models/xai/retrained_clean/xai_cnn_{bench}.keras")
        hs_files = sorted(glob.glob(f"iccad-official/{bench}/test/test_hs/*"))
        if hs_files and ckpt_path.exists():
            img_tensor = load_test_image(hs_files[0])
            check = run_model_parameter_randomization_check(ckpt_path, img_tensor, target_idx=0)
            sanity_results[bench] = check
            print(f"{bench} Randomization Check: {check}")

    # 3. Seed Stability Study across Seeds 101, 202, 303
    print("\nEvaluating Cross-Seed Attribution Stability...")
    seed_stability_results = {}
    for bench in ["iccad1", "iccad2", "iccad5"]:
        seed_ckpts = [
            Path(f"models/xai/retrained_clean/seed_variance/{bench}_seed101.keras"),
            Path(f"models/xai/retrained_clean/seed_variance/{bench}_seed202.keras"),
            Path(f"models/xai/retrained_clean/seed_variance/{bench}_seed303.keras"),
        ]
        if all(p.exists() for p in seed_ckpts):
            print(f"Testing seed stability on {bench}...")
            models = [tf.keras.models.load_model(p, compile=False) for p in seed_ckpts]
            engines = [XAIEngine(m) for m in models]
            hs_files = sorted(glob.glob(f"iccad-official/{bench}/test/test_hs/*"))[:5]

            bench_seed_corrs = {"gradcam": [], "layercam": []}
            for p in hs_files:
                img_tensor = load_test_image(p)
                # Compute maps across the 3 seeds
                gcams = [eng.explain_gradcam(img_tensor, target_class=0)[1] for eng in engines]
                lcams = [eng.explain_layercam(img_tensor, target_class=0)[1] for eng in engines]

                # Pairwise rank correlations: (0,1), (0,2), (1,2)
                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        rg, _ = spearmanr(gcams[i].flatten(), gcams[j].flatten())
                        rl, _ = spearmanr(lcams[i].flatten(), lcams[j].flatten())
                        if not np.isnan(rg):
                            bench_seed_corrs["gradcam"].append(float(rg))
                        if not np.isnan(rl):
                            bench_seed_corrs["layercam"].append(float(rl))

            seed_stability_results[bench] = {
                "gradcam_mean_cross_seed_rank_corr": float(np.mean(bench_seed_corrs["gradcam"])),
                "gradcam_std_cross_seed_rank_corr": float(np.std(bench_seed_corrs["gradcam"])),
                "layercam_mean_cross_seed_rank_corr": float(np.mean(bench_seed_corrs["layercam"])),
                "layercam_std_cross_seed_rank_corr": float(np.std(bench_seed_corrs["layercam"])),
                "num_eval_pairs": len(bench_seed_corrs["gradcam"]),
            }
            print(f"{bench} Seed Stability Summary: {seed_stability_results[bench]}")

    # Compile Final Execution Artifact
    full_output = {
        "faithfulness": faithfulness_results,
        "parameter_randomization_sanity_check": sanity_results,
        "cross_seed_attribution_stability": seed_stability_results,
    }

    with open(out_dir / "faithfulness_and_stability_results.json", "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print("\n✓ Evaluation complete. Saved to results/xai_faithfulness/faithfulness_and_stability_results.json")


if __name__ == "__main__":
    main()
