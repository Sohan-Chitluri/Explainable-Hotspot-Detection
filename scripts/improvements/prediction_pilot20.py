"""Lead-run reduced B prediction validation on 20 deterministic ICCAD5 test rasters.

Self-check directive: this CPU-only forward pass is a pilot, not a GPU substitute.
No checkpoint/production result is modified. GPU unavailability reconfirmed before run.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from src.xai_engine import XAIEngine

CHECKPOINT = ROOT / "models/xai/experimental/xai_cnn_iccad5_balanced_sampling_seed1.keras"
EXPECTED_SHA256 = "9479f07781b0128e19d891b295c99b544f9e6fd2b9a4628533138a47c8085b70"
SEED = 1729
PER_CLASS = 10


def legacy_tensor(path: Path) -> np.ndarray:
    """Historical XAI pipeline tensor: PIL RGB bicubic 1200->224, uint8, /255."""
    with Image.open(path) as source:
        legacy = np.asarray(
            source.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )
    return np.expand_dims(legacy.astype(np.float32) / 255.0, axis=0)


def main():
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"Checkpoint identity mismatch: {digest}")
    engine = XAIEngine.from_checkpoint(CHECKPOINT, target_layer_name="conv_final_2")
    directory = ROOT / "iccad-official/iccad5/test"
    paths = []
    for label in ("test_hs", "test_nhs"):
        class_paths = list((directory / label).glob("*.png"))
        class_paths.sort(key=lambda p: hashlib.sha256(f"{SEED}|{p.relative_to(ROOT)}".encode()).digest())
        paths.extend((p, label) for p in class_paths[:PER_CLASS])

    records = []
    for path, label in paths:
        tensor, _ = engine.preprocess_image(path)
        official_image = load_img(path, color_mode="rgb", target_size=(224, 224), interpolation="nearest")
        official = img_to_array(official_image, dtype="float32") / 255.0
        tensor_equal = bool(np.array_equal(tensor[0], official))
        if not tensor_equal:
            raise SystemExit(f"Tensor mismatch vs official path: {path}")
        p_hs_corrected, _, pred_corrected, _ = engine.predict(tensor)
        p_hs_legacy, _, pred_legacy, _ = engine.predict(legacy_tensor(path))
        records.append({
            "source": str(path.relative_to(ROOT)),
            "label": label,
            "p_hs_nearest": round(p_hs_corrected, 6),
            "pred_nearest": pred_corrected,
            "p_hs_legacy_bicubic": round(p_hs_legacy, 6),
            "pred_legacy_bicubic": pred_legacy,
            "abs_p_delta": round(abs(p_hs_corrected - p_hs_legacy), 6),
            "prediction_flip": pred_corrected != pred_legacy,
            "official_tensor_equal": tensor_equal,
        })

    deltas = np.array([r["abs_p_delta"] for r in records])
    output = ROOT / "results/xai/improvements/preprocessing"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "prediction_pilot20.csv"
    summary_path = output / "prediction_pilot_summary.json"
    if target.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite existing pilot outputs")
    with target.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "n": len(records),
        "selection": f"First {PER_CLASS} HS + {PER_CLASS} NHS by SHA256(seed|relative path); no confidence selection",
        "device_note": "CPU forward pass pilot; GPU unavailable (nvidia-smi exit 6 reconfirmed 2026-09-17)",
        "checkpoint_sha256": digest,
        "tensor_equivalence": "official-path tensor equality asserted per image (all true or run aborts)",
        "probability_max_difference_nearest_vs_legacy_bicubic": float(deltas.max()),
        "probability_mean_difference_nearest_vs_legacy_bicubic": float(deltas.mean()),
        "classification_disagreements_nearest_vs_legacy_bicubic": int(sum(r["prediction_flip"] for r in records)),
        "status": "PILOT_n20_descriptive",
        "limitations": "Single checkpoint, 20 deterministic images, CPU forward passes. Full-set GPU inference and per-benchmark sweeps remain blocked. Descriptive only; no production claim.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
