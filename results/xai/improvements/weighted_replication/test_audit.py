"""CPU-only regression test against immutable historical stored outputs."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[4]
script = ROOT / 'scripts/improvements/weighted_replication.py'
assert script.is_file(), 'CPU artifact auditor has not been implemented'
audit = runpy.run_path(str(script))['audit']
result = audit(ROOT)
w = next(x for x in result['runs'] if x['run'] == 'class_weighted')
assert w['n'] == 19368 and w['n_hs'] == 41
assert w['confusion_matrix'] == {'tp': 39, 'tn': 18914, 'fp': 413, 'fn': 2}
assert abs(w['balanced_accuracy'] - 0.9649252215086439) < 1e-12
assert w['zero_hs'] == 35 and w['zero_nhs'] == 312
assert w['dead_units'] == 6 and w['hs_elevated_units'] == 0
assert w['stored_zero_flags_match'] and w['stored_pred_labels_match']
assert result['checkpoint_selection']['selected_epoch_zero_based'] == 8
assert result['checkpoint_selection']['selected_matches_first_argmax']
assert result['checkpoint_selection']['last_epoch_balanced_accuracy'] == 0.8473978649860487
assert result['required_replications'] == [
    {'run': f'class_weighted_replication{s}', 'seed': s, 'status': 'BLOCKED_NOT_RUN'}
    for s in (101, 202, 303)
]
assert len(result['runs']) == 8
assert all(x['n'] == 19368 and x['n_hs'] == 41 and x['stored_pred_labels_match'] and x['stored_zero_flags_match'] for x in result['runs'])
print('PASS: 8 stored-output runs; weighted confusion, collapse, selection and blocked seeds verified; no ML execution')
