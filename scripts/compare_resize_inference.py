#!/usr/bin/env python3
"""
compare_resize_inference.py
===========================
Re-run inference on all 5 ICCAD test sets under two preprocessing paths:

  1. Canonical (nearest-neighbor) - current pipeline introduced in commit 9ea630d.
     Uses PIL.Image.resize(..., resample=Image.Resampling.NEAREST), identical to
     what Keras ImageDataGenerator produces internally.

  2. Legacy (PIL bicubic) - behaviour prior to commit 9ea630d.
     Uses PIL.Image.resize(...) with default resampling (BICUBIC).

For each image both predictions are collected and compared.

Output
------
- results/resize_comparison/iccad{N}_resize_comparison.csv  (one per benchmark)
  Columns: image_id, pred_nearest, prob_nearest, pred_bicubic, prob_bicubic, changed

- results/resize_comparison/summary.txt  (human-readable summary table)

The XAI model checkpoints in models/xai/ are used (xai_cnn_iccad{N}.keras).
Falls back to the baseline custom_cnn_iccad{N}.keras if the XAI model is absent.

Usage (from repo root, with .venv activated)
--------------------------------------------
  python -m scripts.compare_resize_inference
  python -m scripts.compare_resize_inference --benchmarks 1 2 --batch-size 64
  python -m scripts.compare_resize_inference --data-root ./iccad-official --results-dir ./results
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import textwrap
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare nearest-neighbor vs bicubic preprocessing across ICCAD test sets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "iccad-official",
        help="Root folder containing iccad1..iccad5 (default: %(default)s)",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
        help="Directory holding Keras model checkpoints (default: %(default)s)",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "resize_comparison",
        help="Output directory for CSVs and summary (default: %(default)s)",
    )
    p.add_argument(
        "--benchmarks",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
        help="Which ICCAD benchmarks to process (default: all five)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of images processed per TF inference call (default: %(default)s)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for the sigmoid output (default: %(default)s)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Heavy imports after arg parsing so --help is fast
# ---------------------------------------------------------------------------

# Ensure repo root is on sys.path so `src` package is importable whether this
# script is run as   python scripts/compare_resize_inference.py   or as
#                    python -m scripts.compare_resize_inference
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import tensorflow as tf  # noqa: E402

# Reuse existing architecture builders — do not reimplement from scratch.
# build_xai_cnn / build_custom_cnn are imported so Keras can reconstruct the
# saved model graph; tf.keras.models.load_model handles actual weight loading.
from src.model_xai import build_xai_cnn  # noqa: E402,F401
from src.model import build_custom_cnn   # noqa: E402,F401


# ---------------------------------------------------------------------------
# Preprocessing helpers — mirror exactly what changed in commit 9ea630d
# ---------------------------------------------------------------------------

_TARGET: Tuple[int, int] = (224, 224)  # PIL uses (width, height)


def _open_rgb(path: Path) -> Image.Image:
    """Open any raster file and return a 3-channel RGB PIL image."""
    return Image.open(str(path)).convert("RGB")


def preprocess_nearest(path: Path) -> np.ndarray:
    """
    Canonical preprocessing (current pipeline, post-commit 9ea630d).

    Replicates XAIEngine.preprocess_image with resample=Image.Resampling.NEAREST,
    which is tensor-identical to Keras ImageDataGenerator(rescale=1/255).
    """
    arr = np.array(_open_rgb(path).resize(_TARGET, resample=Image.Resampling.NEAREST),
                   dtype=np.float32) / 255.0
    return arr  # (224, 224, 3)


def preprocess_bicubic(path: Path) -> np.ndarray:
    """
    Legacy preprocessing (prior to commit 9ea630d).

    PIL.Image.resize() without an explicit resample kwarg defaults to BICUBIC
    (since PIL >= 9.1 the default changed from NEAREST to BICUBIC, so modern
    PIL without explicit resample == BICUBIC).
    """
    arr = np.array(_open_rgb(path).resize(_TARGET, resample=Image.Resampling.BICUBIC),
                   dtype=np.float32) / 255.0
    return arr  # (224, 224, 3)


# ---------------------------------------------------------------------------
# Model loading — reuses saved .keras checkpoints, no retraining
# ---------------------------------------------------------------------------

def load_model(benchmark: int, models_dir: Path) -> tf.keras.Model:
    """
    Load the saved Keras model for a benchmark.

    Preference order:
      1. models/xai/xai_cnn_iccad{N}.keras   (XAI-ready architecture; build_xai_cnn)
      2. models/custom_cnn_iccad{N}.keras     (baseline CNN; build_custom_cnn)

    Raises FileNotFoundError if neither checkpoint is found.
    """
    candidates = [
        models_dir / "xai" / f"xai_cnn_iccad{benchmark}.keras",
        models_dir / f"custom_cnn_iccad{benchmark}.keras",
    ]
    for path in candidates:
        if path.is_file():
            print(f"  Loading model: {path}")
            model = tf.keras.models.load_model(str(path))
            return model
    raise FileNotFoundError(
        f"No model checkpoint found for iccad{benchmark}.\n"
        + "\n".join(f"  Searched: {c}" for c in candidates)
        + "\nRun src/train_baseline.py or src/train_xai.py first."
    )


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_test_images(data_root: Path, benchmark: int) -> List[Tuple[str, Path]]:
    """
    Walk iccad{N}/test/{test_hs,test_nhs} and return a sorted list of
    (image_id, absolute_path) pairs.

    image_id format: <subfolder>/<stem>  e.g. "test_hs/HS0"
    """
    test_dir = data_root / f"iccad{benchmark}" / "test"
    entries: List[Tuple[str, Path]] = []
    for subdir in sorted(test_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for img_path in sorted(subdir.iterdir()):
            if img_path.suffix.lower() in _IMG_EXTS:
                image_id = f"{subdir.name}/{img_path.stem}"
                entries.append((image_id, img_path))
    return entries


# ---------------------------------------------------------------------------
# Batched inference with tqdm progress bar
# ---------------------------------------------------------------------------

def run_inference_batched(
    model: tf.keras.Model,
    paths: List[Path],
    preprocess_fn,
    batch_size: int,
    threshold: float,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model.predict in batches.

    Returns
    -------
    preds : np.ndarray[int]   shape (N,)   0=HS, 1=NHS
    probs : np.ndarray[float] shape (N,)   raw sigmoid P(NHS)
    """
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("  [info] tqdm not installed; progress printed every 5 000 images.")

    n = len(paths)
    all_probs: List[np.ndarray] = []
    start = time.time()

    batch_indices = list(range(0, n, batch_size))
    if use_tqdm:
        batch_indices = tqdm(
            batch_indices,
            desc=f"    {label:8s}",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
        )

    for step, batch_start in enumerate(batch_indices):
        batch_paths = paths[batch_start : batch_start + batch_size]
        # Stack preprocessed arrays into a single batch tensor
        batch_arr = np.stack([preprocess_fn(p) for p in batch_paths], axis=0)
        # model() call; TF will dispatch to GPU if available
        raw = model(batch_arr, training=False).numpy()
        all_probs.append(raw.ravel())

        # Fallback progress logging when tqdm is unavailable
        if not use_tqdm and step % 20 == 0:
            done = min(batch_start + batch_size, n)
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta_s = (n - done) / rate if rate > 0 else 0.0
            print(
                f"    {label}: {done:>7,}/{n:,}  "
                f"({100.0 * done / n:5.1f}%)  "
                f"ETA {eta_s / 60:.1f} min",
                flush=True,
            )

    probs = np.concatenate(all_probs)          # P(NHS)
    preds = (probs >= threshold).astype(int)   # 0=HS, 1=NHS
    return preds, probs


# ---------------------------------------------------------------------------
# Per-benchmark driver
# ---------------------------------------------------------------------------

_LABEL_MAP = {0: "HS", 1: "NHS"}


def process_benchmark(
    benchmark: int,
    data_root: Path,
    models_dir: Path,
    results_dir: Path,
    batch_size: int,
    threshold: float,
) -> dict:
    """Run both preprocessing paths for one benchmark and write the CSV."""
    print(f"\n{'='*70}")
    print(f"  ICCAD{benchmark}")
    print(f"{'='*70}")

    # ── discover test images ───────────────────────────────────────────────
    entries = collect_test_images(data_root, benchmark)
    n = len(entries)
    if n == 0:
        raise RuntimeError(
            f"No images found in {data_root}/iccad{benchmark}/test/. "
            "Check --data-root."
        )
    image_ids = [e[0] for e in entries]
    paths     = [e[1] for e in entries]
    print(f"  Found {n:,} test images")

    # ── load checkpoint ────────────────────────────────────────────────────
    model = load_model(benchmark, models_dir)

    # ── nearest-neighbor pass ──────────────────────────────────────────────
    print("  Running nearest-neighbor (canonical) inference …")
    preds_nn, probs_nn = run_inference_batched(
        model, paths, preprocess_nearest, batch_size, threshold, label="nearest"
    )

    # ── bicubic pass ───────────────────────────────────────────────────────
    print("  Running bicubic (legacy) inference …")
    preds_bc, probs_bc = run_inference_batched(
        model, paths, preprocess_bicubic, batch_size, threshold, label="bicubic"
    )

    # ── release GPU memory before next benchmark ───────────────────────────
    del model
    tf.keras.backend.clear_session()

    # ── compute "changed" flag ─────────────────────────────────────────────
    changed = (preds_nn != preds_bc).astype(int)
    n_changed = int(changed.sum())

    # ── write per-benchmark CSV ────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"iccad{benchmark}_resize_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "image_id",
            "pred_nearest", "prob_nearest",
            "pred_bicubic", "prob_bicubic",
            "changed",
        ])
        for img_id, pn, prn, pb, prb, ch in zip(
            image_ids,
            preds_nn, probs_nn,
            preds_bc, probs_bc,
            changed,
        ):
            writer.writerow([
                img_id,
                _LABEL_MAP[int(pn)],
                f"{float(prn):.8f}",
                _LABEL_MAP[int(pb)],
                f"{float(prb):.8f}",
                bool(ch),
            ])

    print(f"  CSV written → {csv_path}")
    print(f"  Changed: {n_changed:,} / {n:,}  ({100.0 * n_changed / n:.4f}%)")

    return {
        "benchmark":   f"iccad{benchmark}",
        "total":       n,
        "changed":     n_changed,
        "pct_changed": 100.0 * n_changed / n,
        "csv":         str(csv_path),
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_and_save_summary(all_stats: list, results_dir: Path) -> None:
    col_w = [12, 12, 12, 14]
    header = (
        f"{'Benchmark':<{col_w[0]}} "
        f"{'Total':>{col_w[1]}} "
        f"{'Changed':>{col_w[2]}} "
        f"{'Changed %':>{col_w[3]}}"
    )
    sep = "-" * (sum(col_w) + len(col_w))
    lines = [sep, header, sep]

    total_all = 0
    changed_all = 0
    for s in all_stats:
        lines.append(
            f"{s['benchmark']:<{col_w[0]}} "
            f"{s['total']:>{col_w[1]},} "
            f"{s['changed']:>{col_w[2]},} "
            f"{s['pct_changed']:>{col_w[3]-1}.4f}%"
        )
        total_all   += s["total"]
        changed_all += s["changed"]

    pct_all = 100.0 * changed_all / total_all if total_all > 0 else 0.0
    lines += [
        sep,
        (
            f"{'TOTAL':<{col_w[0]}} "
            f"{total_all:>{col_w[1]},} "
            f"{changed_all:>{col_w[2]},} "
            f"{pct_all:>{col_w[3]-1}.4f}%"
        ),
        sep,
    ]

    body = "\n".join(lines)
    print("\n\nSUMMARY — Nearest-Neighbor vs Bicubic Preprocessing\n")
    print(body)

    out_path = results_dir / "summary.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(
            "Nearest-Neighbor (canonical, post-9ea630d) "
            "vs Bicubic (legacy, pre-9ea630d)\n\n"
        )
        fh.write(body + "\n\n")
        fh.write(
            "Column legend\n"
            "  image_id      : <subfolder>/<stem>  e.g. test_hs/HS0\n"
            "  pred_nearest  : class under nearest-neighbor resize (HS / NHS)\n"
            "  prob_nearest  : raw sigmoid P(NHS) under nearest-neighbor resize\n"
            "  pred_bicubic  : class under PIL bicubic resize (HS / NHS)\n"
            "  prob_bicubic  : raw sigmoid P(NHS) under PIL bicubic resize\n"
            "  changed       : True when pred_nearest != pred_bicubic\n"
        )
    print(f"\nSummary written → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    print(f"Python     : {sys.version.split()[0]}")
    print(f"TensorFlow : {tf.__version__}")

    gpus = tf.config.list_physical_devices("GPU")
    device_str = f"GPU x{len(gpus)}" if gpus else "CPU"
    print(f"Device     : {device_str}")

    # Enable GPU memory growth to avoid pre-allocating the whole VRAM
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # must be called before GPUs are initialised

    print(f"\nData root  : {args.data_root}")
    print(f"Models dir : {args.models_dir}")
    print(f"Results dir: {args.results_dir}")
    print(f"Benchmarks : {args.benchmarks}")
    print(f"Batch size : {args.batch_size}")
    print(f"Threshold  : {args.threshold}")

    if not args.data_root.is_dir():
        print(f"\nERROR: --data-root not found: {args.data_root}", file=sys.stderr)
        return 1

    wall_start = time.time()
    all_stats: list = []
    failed: list = []

    for b in args.benchmarks:
        try:
            stats = process_benchmark(
                benchmark=b,
                data_root=args.data_root,
                models_dir=args.models_dir,
                results_dir=args.results_dir,
                batch_size=args.batch_size,
                threshold=args.threshold,
            )
            all_stats.append(stats)
        except Exception as exc:
            print(f"\n  ERROR on iccad{b}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            failed.append(b)

    if all_stats:
        print_and_save_summary(all_stats, args.results_dir)

    elapsed = time.time() - wall_start
    print(f"\nTotal wall time: {elapsed / 60:.1f} min")

    if failed:
        print(f"\nFailed benchmarks: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
