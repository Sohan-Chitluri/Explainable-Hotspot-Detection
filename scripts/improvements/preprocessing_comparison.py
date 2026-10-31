"""Deterministic CPU-only loader comparison; never constructs a CNN."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from src.xai_engine import XAIEngine


def main():
    output = ROOT / 'results/xai/improvements/preprocessing'
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'tensor_comparison.csv'
    if target.exists():
        raise FileExistsError(f'Refusing to replace {target}')
    # Hash ordering gives deterministic spread, not filename/confidence extremes.
    seed = 1729
    selected = []
    for benchmark in range(1, 6):
        for split in ('train', 'test'):
            for label in ('hs', 'nhs'):
                paths = list((ROOT / f'iccad-official/iccad{benchmark}/{split}/{split}_{label}').glob('*.png'))
                paths.sort(key=lambda p: hashlib.sha256(f'{seed}|{p.relative_to(ROOT)}'.encode()).digest())
                selected.extend(paths[:64])
    scaler = ImageDataGenerator(rescale=1.0 / 255)
    records = []
    for path in selected:
        official_image = load_img(path, color_mode='rgb', target_size=(224, 224), interpolation='nearest')
        official = scaler.standardize(img_to_array(official_image, dtype='float32'))
        corrected, raw = XAIEngine.preprocess_image(path)
        with Image.open(path) as source:
            source_size = source.size
            old = np.asarray(source.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC), dtype=np.uint8).astype(np.float32) / 255.0
        if corrected.shape != (1, 224, 224, 3) or corrected.dtype != np.float32:
            raise AssertionError(f'Invalid tensor contract: {path}')
        delta = np.abs(corrected[0] - official)
        legacy_delta = np.abs(old - official)
        records.append({'source': str(path.relative_to(ROOT)), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'source_width': source_size[0], 'source_height': source_size[1],
                        'corrected_max_abs': float(delta.max()), 'corrected_mean_abs': float(delta.mean(dtype=np.float64)),
                        'corrected_unequal_elements': int(np.count_nonzero(delta)),
                        'legacy_max_abs': float(legacy_delta.max()), 'legacy_mean_abs': float(legacy_delta.mean(dtype=np.float64)),
                        'legacy_unequal_elements': int(np.count_nonzero(legacy_delta)),
                        'raw_equal': bool(np.array_equal(raw, np.asarray(official_image))),
                        'channels_equal': bool(np.array_equal(corrected[0, :, :, 0], corrected[0, :, :, 1]) and np.array_equal(corrected[0, :, :, 1], corrected[0, :, :, 2]))})
    assert len(records) >= 1000, len(records)
    with target.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {'n': len(records), 'seed': seed,
               'selection': 'First min(64,N) SHA256(seed|relative path) per benchmark/split/class; no inference selection',
               'official': 'Keras RGB, 224x224, explicit nearest, img_to_array float32, ImageDataGenerator standardize rescale1/255',
               'corrected': 'XAIEngine explicit Pillow nearest; existing float32 division normalization',
               'legacy': 'Pillow RGB bicubic; float32 division normalization',
               'corrected_max_abs': max(r['corrected_max_abs'] for r in records),
               'corrected_mean_abs': float(np.mean([r['corrected_mean_abs'] for r in records])),
               'corrected_unequal_images': sum(r['corrected_unequal_elements'] > 0 for r in records),
               'legacy_max_abs': max(r['legacy_max_abs'] for r in records),
               'legacy_mean_abs': float(np.mean([r['legacy_mean_abs'] for r in records])),
               'legacy_unequal_images': sum(r['legacy_unequal_elements'] > 0 for r in records),
               'prediction_validation_status': 'NOT_MEASURED_GPU_UNAVAILABLE',
               'probability_max_difference': None, 'probability_mean_difference': None,
               'classification_disagreements': None,
               'limitations': 'Tensor equality tested on binary dataset rasters only; does not validate arbitrary nonbinary array normalization or cached bicubic inputs. No CNN created, no GPU inference claimed.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))
    if summary['corrected_unequal_images']:
        raise SystemExit('Corrected tensor comparison failed')


if __name__ == '__main__':
    main()
