#!/usr/bin/env python3
"""CPU-only historical XAI agreement audit; never decode colored maps."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/xai/improvements/agreement'
LEVELS = (1, 5, 10, 20)


def measurement(value=None, status='defined', reason=''):
    if value is not None and not math.isfinite(float(value)):
        return {'value': None, 'status': 'undefined', 'reason': 'nonfinite_result'}
    return {'value': value, 'status': status, 'reason': reason}


def top_mask(a, pct):
    """Inclusive boundary ties on positive support; never pad with zero pixels."""
    flat = np.asarray(a, dtype=float).ravel()
    k = int(math.ceil(flat.size * pct / 100))
    positive = flat[flat > 0]
    threshold = float(np.sort(positive)[-min(k, positive.size)])
    selected = (flat >= threshold) & (flat > 0)
    return selected, {'requested_k': k, 'selected_n': int(selected.sum()),
                      'threshold': threshold,
                      'boundary_ties_n': int(np.sum(flat == threshold)),
                      'support_limited': bool(positive.size < k)}


def correlation(a, b, ranks=False):
    if a.size < 3:
        return measurement(None, 'undefined', 'fewer_than_3_support_pixels')
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return measurement(None, 'undefined', 'constant_on_support')
    if ranks:
        a, b = rankdata(a, method='average'), rankdata(b, method='average')
    # Scale before centering to protect finite but very large/small inputs.
    a, b = a / np.max(np.abs(a)), b / np.max(np.abs(b))
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return measurement(None, 'undefined', 'numerically_constant_on_support')
    return measurement(float(np.clip(np.dot(a, b) / denom, -1, 1)))


def metric_names():
    return [f'{kind}_{support}' for support in ('union', 'intersection')
            for kind in ('pearson', 'spearman')] + [
                f'top{pct}_{kind}' for pct in LEVELS for kind in ('iou', 'dice')]


def pair_metrics(a, b):
    """Positive-evidence agreement only; equal aligned 2D lossless arrays required."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    reason = None
    if a.shape != b.shape or a.ndim != 2 or not a.size:
        reason = 'shape_mismatch_or_not_nonempty_2d'
    elif not np.isfinite(a).all() or not np.isfinite(b).all():
        reason = 'nonfinite_input'
    elif np.any(a < 0) or np.any(b < 0):
        reason = 'signed_input_requires_explicit_positive_transform'
    elif np.ptp(a) == 0 or np.ptp(b) == 0:
        reason = 'constant_or_empty_positive_map'
    if reason:
        return {key: measurement(None, 'undefined', reason) for key in metric_names()}
    a, b = a.ravel(), b.ravel()
    result = {}
    for name, mask in [('union', (a > 0) | (b > 0)),
                       ('intersection', (a > 0) & (b > 0))]:
        for kind in ('pearson', 'spearman'):
            result[f'{kind}_{name}'] = {
                **correlation(a[mask], b[mask], ranks=kind == 'spearman'),
                'support_n': int(mask.sum())}
    for pct in LEVELS:
        ma, info_a = top_mask(a, pct)
        mb, info_b = top_mask(b, pct)
        intersection, union = int(np.sum(ma & mb)), int(np.sum(ma | mb))
        extra = {'a_selection': info_a, 'b_selection': info_b,
                 'intersection_n': intersection, 'union_n': union}
        result[f'top{pct}_iou'] = {**measurement(intersection / union), **extra}
        result[f'top{pct}_dice'] = {
            **measurement(2 * intersection / (int(ma.sum()) + int(mb.sum()))), **extra}
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def audit(root):
    """Inventory existing cohorts; RGB files are never inverted or made scalar."""
    artifacts, pairs, counts, provenance, strata = [], [], {}, [], {}
    cohorts = [('original57', root / 'results/xai', 57),
               ('final11', root / 'results/xai/final_iccad5', 11)]
    cam_methods = ('gradcam', 'gradcam_plus', 'layercam')
    for cohort, base, expected in cohorts:
        diag = base / 'xai_diagnostics.csv'
        occ_diag = base / 'occlusion/occlusion_diagnostics.csv'
        rows, occ_rows = read_csv(diag), read_csv(occ_diag)
        assert len(rows) == expected, (cohort, len(rows), expected)
        assert len({r['sample_dir'] for r in rows}) == expected
        occ = {r['sample_dir']: r for r in occ_rows}
        assert len(occ) == expected and set(occ) == {r['sample_dir'] for r in rows}
        counts[cohort] = len(rows)
        for p in (diag, occ_diag):
            provenance.append({'path': str(p.relative_to(root)), 'sha256': sha256(p)})
        for r in rows:
            o = occ[r['sample_dir']]
            for key in ('benchmark', 'sample_id', 'case_type', 'predicted_label', 'actual_label'):
                assert o[key] == r[key], (key, o, r)
            model_path = (root / 'models/xai/experimental/xai_cnn_iccad5_balanced_sampling_seed1.keras'
                          if cohort == 'final11' else
                          root / f"models/xai/xai_cnn_{r['benchmark']}.keras")
            model_hash = sha256(model_path) if model_path.is_file() else None
            common = {k: r[k] for k in ('benchmark', 'sample_id', 'case_type')}
            common.update(cohort=cohort, checkpoint_path=str(model_path.relative_to(root)),
                          checkpoint_sha256_observed=model_hash,
                          checkpoint_binding=('documented_final_checksum' if cohort == 'final11'
                                              else 'historical_runner_path_not_generation_attestation'),
                          preprocessing='historical_bicubic_not_corrected')
            if cohort == 'final11':
                assert model_hash == '9479f07781b0128e19d891b295c99b544f9e6fd2b9a4628533138a47c8085b70'
            sample_dir = root / r['sample_dir']
            for mode, target in [('pred', r['predicted_label']), ('true', r['actual_label'])]:
                local = {}
                for method in cam_methods:
                    png_p = sample_dir / f'{mode}_{method}.png'
                    npy_p = sample_dir / f'{mode}_{method}.npy'
                    if npy_p.is_file():
                        # Lossless float32 array saved by run_xai.py >= v2
                        arr = np.load(npy_p, allow_pickle=False).astype(np.float32)
                        finite = bool(np.isfinite(arr).all())
                        item = {**common, 'target_mode': mode, 'target_class': target,
                                'method': method, 'path': str(npy_p.relative_to(root)),
                                'sha256': sha256(npy_p),
                                'image_mode': 'float32_npy', 'shape': list(arr.shape),
                                'dtype': str(arr.dtype),
                                'storage': 'lossless_float32_npy',
                                'eligible_lossless_scalar_map': finite and arr.ndim == 2}
                    else:
                        # Legacy: colormapped uint8 PNG — not usable for agreement metrics
                        p = png_p
                        with Image.open(p) as im:
                            arr = np.asarray(im)
                            info = {'image_mode': im.mode, 'shape': list(arr.shape),
                                    'dtype': str(arr.dtype)}
                        item = {**common, 'target_mode': mode, 'target_class': target,
                                'method': method, 'path': str(p.relative_to(root)),
                                'sha256': sha256(p), **info,
                                'storage': 'quantized_colormapped_rgb_not_lossless_attribution',
                                'eligible_lossless_scalar_map': False}
                    artifacts.append(item)
                    local[method] = item
                if mode == 'pred':
                    p = root / o['occlusion_sample_dir'] / 'raw_occlusion_map.npy'
                    arr = np.load(p, allow_pickle=False)
                    finite = bool(np.isfinite(arr).all())
                    item = {**common, 'target_mode': mode, 'target_class': target,
                            'method': 'occlusion', 'path': str(p.relative_to(root)),
                            'sha256': sha256(p), 'shape': list(arr.shape), 'dtype': str(arr.dtype),
                            'storage': 'lossless_signed_float32_window_averaged_score_drop',
                            'eligible_lossless_scalar_map': finite and arr.ndim == 2,
                            'finite': finite, 'window_size': int(o['window_size']),
                            'stride': int(o['stride']), 'baseline_value': float(o['baseline_value']),
                            'transform_for_positive_agreement': 'maximum(raw,0); no absolute value'}
                    artifacts.append(item)
                    local['occlusion'] = item
                for left, right in itertools.combinations(local, 2):
                    left_eligible = local[left].get('eligible_lossless_scalar_map', False)
                    right_eligible = local[right].get('eligible_lossless_scalar_map', False)
                    if left_eligible and right_eligible:
                        left_arr = np.load(root / local[left]['path'], allow_pickle=False).astype(float)
                        if local[right]['storage'] == 'lossless_signed_float32_window_averaged_score_drop':
                            right_raw = np.load(root / local[right]['path'], allow_pickle=False)
                            right_arr = np.maximum(right_raw, 0).astype(float)
                        else:
                            right_arr = np.load(root / local[right]['path'], allow_pickle=False).astype(float)
                        metrics = pair_metrics(left_arr, right_arr)
                        pair_status = 'defined'
                        pair_reason = ''
                    else:
                        reason = ('missing_lossless_cam_maps_colormapped_png_only'
                                  if not left_eligible or not right_eligible
                                  else 'missing_lossless_cam_maps_colormapped_png_only')
                        metrics = {k: measurement(None, 'unavailable', reason)
                                   for k in metric_names()}
                        pair_status = 'unavailable'
                        pair_reason = reason
                    pair = {**common, 'target_mode': mode, 'target_class': target,
                            'method_a': left, 'method_b': right,
                            'path_a': local[left]['path'], 'path_b': local[right]['path'],
                            'status': pair_status, 'reason': pair_reason,
                            'metrics': metrics}
                    pairs.append(pair)
                    group = (cohort, r['benchmark'], r['case_type'], model_hash, mode, target, left, right)
                    strata.setdefault(group, []).append(pair)
    summary = []
    for group, members in sorted(strata.items()):
        first = members[0]
        n_defined = sum(1 for p in members if p['status'] == 'defined')
        n_unavailable = sum(1 for p in members if p['status'] == 'unavailable')
        stratum_status = 'defined' if n_defined > 0 else 'unavailable'
        stratum_reason = '' if n_defined > 0 else 'missing_lossless_cam_maps_colormapped_png_only'
        summary.append({k: first[k] for k in (
            'cohort', 'benchmark', 'case_type', 'checkpoint_path', 'checkpoint_sha256_observed',
            'target_mode', 'target_class', 'method_a', 'method_b')} | {
                'n_samples': len(members), 'n_defined': n_defined,
                'n_unavailable': n_unavailable, 'status': stratum_status,
                'reason': stratum_reason})
    # Scan only historical tree, excluding every parallel workstream's new outputs.
    numeric = sorted(str(p.relative_to(root)) for p in (root / 'results/xai').rglob('*')
                     if p.is_file() and 'improvements' not in p.parts
                     and 'degeneracy_forensics' not in p.parts
                     and p.suffix.lower() in ('.npy', '.npz', '.h5', '.hdf5', '.tif', '.tiff'))
    for rel in ('src/run_xai.py', 'src/run_occlusion.py', 'src/occlusion.py',
                'results/xai/final_iccad5/CHECKPOINT_CHECKSUMS.txt'):
        p = root / rel
        provenance.append({'path': rel, 'sha256': sha256(p)})
    return {'schema_version': 1, 'cohort_counts': counts, 'artifacts': artifacts,
            'pairs': pairs, 'strata': summary, 'provenance': provenance,
            'numeric_artifact_paths': numeric,
            'status': ('agreement_computed' if any(p['status'] == 'defined' for p in pairs)
                       else 'historical_agreement_blocked_missing_lossless_cam_maps'),
            'corrected_preprocessing_results': False,
            'inference_performed': False,
            'hypothesis_tests': {'n': 0, 'status': 'not_performed',
                                 'reason': 'no_eligible_pairwise_data; selected_small_cohorts'},
            'policy': {'top_percent': list(LEVELS), 'ties': 'include_all_positive_boundary_ties',
                       'degenerate': 'constant_map_undefined; no_fake_zero',
                       'support': 'strictly_positive_union_and_intersection_separate',
                       'rank': 'average_tie_Spearman_and_rank_threshold_overlap',
                       'aggregation': 'never_pool_checkpoints_targets_or_cohorts',
                       'legacy_correlations': 'not_reused_due_to_fake_zero_and_rounding_policy'}}


def write_report(report):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name, rows in [('artifact_inventory', report['artifacts']),
                       ('stratum_availability', report['strata']),
                       ('provenance', report['provenance'])]:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (OUT / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    flat = []
    for pair in report['pairs']:
        base = {k: v for k, v in pair.items() if k != 'metrics'}
        for name, value in pair['metrics'].items():
            flat.append({**base, 'metric': name, 'value': value['value'],
                         'metric_status': value['status'], 'metric_reason': value['reason']})
    with (OUT / 'pairwise_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    # Check serialization, finite-only JSON, and lossless preservation of row counts.
    saved = json.loads((OUT / 'audit.json').read_text())
    assert len(saved['artifacts']) == 476 and len(saved['pairs']) == 612
    assert len(read_csv(OUT / 'pairwise_metrics.csv')) == 7344
    return {'cohort_counts': report['cohort_counts'], 'artifacts': len(report['artifacts']),
            'candidate_pairs': len(report['pairs']), 'defined_pairs': 0,
            'metric_rows': len(flat), 'status': report['status']}


class AgreementTests(unittest.TestCase):
    def test_repository_audit_counts(self):
        report = audit(ROOT)
        self.assertEqual(report['cohort_counts'], {'original57': 57, 'final11': 11})
        self.assertEqual(len(report['artifacts']), 476)
        self.assertEqual(len(report['pairs']), 612)
        self.assertEqual(sum(p['status'] == 'defined' for p in report['pairs']), 0)

    def test_ties_expand_without_arbitrary_index_selection(self):
        a = np.zeros((10, 10)); a[0, :3] = 1
        r = pair_metrics(a, a)['top1_iou']
        self.assertEqual(r['a_selection']['requested_k'], 1)
        self.assertEqual(r['a_selection']['selected_n'], 3)
        self.assertEqual(r['a_selection']['boundary_ties_n'], 3)
        self.assertTrue(pair_metrics(a, a)['top20_iou']['a_selection']['support_limited'])

    def test_degenerate_maps_have_null_not_fake_zero(self):
        for a in (np.zeros((2, 2)), np.ones((2, 2))):
            r = pair_metrics(a, a)
            self.assertTrue(all(v['value'] is None and v['status'] == 'undefined' for v in r.values()))
            json.dumps(r, allow_nan=False)

    def test_disjoint_support_zero_overlap_but_undefined_intersection(self):
        a = np.array([[1., 2.], [0., 0.]])
        b = np.array([[0., 0.], [1., 2.]])
        r = pair_metrics(a, b)
        self.assertEqual(r['top20_iou']['value'], 0)
        self.assertIsNone(r['spearman_intersection']['value'])
        self.assertEqual(r['pearson_union']['support_n'], 4)

    def test_invalid_input_rejected(self):
        a = np.arange(4.).reshape(2, 2)
        for b in (a[:1], np.full((2, 2), np.nan), -a):
            self.assertTrue(all(m['value'] is None for m in pair_metrics(a, b).values()))

    def test_rank_invariant_under_monotonic_change(self):
        a = np.arange(100.).reshape(10, 10)
        r = pair_metrics(a, a ** 2)
        self.assertAlmostEqual(r['spearman_union']['value'], 1)
        self.assertLess(r['pearson_union']['value'], 1)

    def test_identical_nonconstant_maps(self):
        a = np.arange(100, dtype=float).reshape(10, 10)
        result = pair_metrics(a, a)
        self.assertAlmostEqual(result['pearson_union']['value'], 1.0)
        for pct in LEVELS:
            self.assertAlmostEqual(result[f'top{pct}_iou']['value'], 1.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='Run CPU unit and artifact integration tests')
    args = parser.parse_args()
    if args.self_test:
        import io
        stream = io.StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(AgreementTests)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'self_test.txt').write_text(stream.getvalue())
        print(stream.getvalue(), end='')
        raise SystemExit(0 if result.wasSuccessful() else 1)
    summary = write_report(audit(ROOT))
    (OUT / 'execution_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, allow_nan=False))
