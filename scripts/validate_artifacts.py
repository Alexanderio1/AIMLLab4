"""Read-only verification of the submitted Ubuntu results; no fitting or services."""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def validate():
    checks = {}
    audit = load('reports/data_audit.json')
    manifest = load('reports/submission_manifest.json')
    evidence_prefixes = ('data/', 'screenshots/', 'exported_models/', 'reports/')
    original = {name: value for name, value in manifest['files'].items() if name.startswith(evidence_prefixes)}
    checks['submitted_evidence_hashes_unchanged'] = all((ROOT / n).is_file() and digest(n) == value for n, value in original.items())
    checks['raw_archive_sha256'] = digest('data/raw/ccpp.zip') == audit['archive_sha256']
    checks['workbook_sha256'] = digest('data/raw/Folds5x2_pp.xlsx') == audit['workbook_sha256']
    checks['processed_sha256'] = digest('data/processed/data.csv') == audit['data_sha256']
    checks['split_sha256'] = digest('data/split.json') == audit['split_sha256']
    raw = pd.read_excel(ROOT / 'data/raw/Folds5x2_pp.xlsx', sheet_name=audit['selected_sheet'])
    clean = pd.read_csv(ROOT / 'data/processed/data.csv')
    expected = raw.drop_duplicates().copy()
    expected.insert(0, 'row_id', expected.index.astype(int))
    pd.testing.assert_frame_equal(clean, expected.reset_index(drop=True), check_exact=False, rtol=0, atol=1e-12)
    checks['first_sheet_and_duplicate_removal'] = len(raw) == audit['original_rows'] == 9568 and len(clean) == audit['rows_after_deduplication'] == 9527 and raw.duplicated().sum() == audit['exact_duplicates_removed'] == 41
    checks['humidity_measurements_preserved'] = int((raw.RH > 100).sum()) == audit['humidity_above_100_count'] == 55 and float(clean.RH.max()) == 100.16
    features = ['AT', 'V', 'AP', 'RH']
    split = load('data/split.json')
    parts = {name: pd.read_csv(ROOT / f'data/processed/{name}.csv') for name in ['train', 'validation', 'test']}
    row_sets = {name: set(frame.row_id) for name, frame in parts.items()}
    checks['split_coverage_disjoint'] = set.union(*row_sets.values()) == set(clean.row_id) and sum(map(len, row_sets.values())) == len(clean)
    groups = {name: set(map(tuple, frame[features].to_numpy())) for name, frame in parts.items()}
    checks['no_feature_vector_leakage'] = groups['train'].isdisjoint(groups['validation']) and groups['train'].isdisjoint(groups['test']) and groups['validation'].isdisjoint(groups['test'])
    for name, frame in parts.items():
        pd.testing.assert_frame_equal(frame, clean.set_index('row_id', drop=False).loc[split[name]].reset_index(drop=True), check_exact=False, rtol=0, atol=1e-12)
        checks[f'{name}_indices_and_count'] = frame.row_id.tolist() == split[name] and len(frame) == audit['split_counts'][name]
    # Reproduce the original split from the workbook, before a CSV float round trip.
    expected = expected.reset_index(drop=True)
    group_ids = pd.util.hash_pandas_object(expected[features], index=False).to_numpy()
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(expected, groups=group_ids))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42).split(expected.iloc[dev], groups=group_ids[dev]))
    reproduced = {'train': dev[tr], 'validation': dev[va], 'test': test}
    checks['seed_42_split_reproduced'] = all(expected.iloc[pos].row_id.tolist() == split[name] for name, pos in reproduced.items())
    runs = load('reports/experiments.json')
    table = pd.read_csv(ROOT / 'reports/experiments.csv')
    registry = load('reports/registry.json')
    versions = registry['versions']
    winner = min(runs, key=lambda r: (r['val_mae'], r['name']))
    checks['six_distinct_runs'] = len(runs) == len(table) == len({r['run_id'] for r in runs}) == 6
    checks['validation_only_winner_consistency'] = winner['run_id'] == versions['selected']['run_id'] and winner['name'] == 'rf-unlimited'
    checks['two_registry_versions'] = versions['baseline']['model_version'] != versions['selected']['model_version']
    for r in runs:
        saved = table.set_index('run_id').loc[r['run_id']]
        checks[f'metrics_consistent_{r["name"]}'] = all(np.isclose(saved[m], r[m], rtol=0, atol=1e-12) for m in ['val_mae', 'val_rmse', 'val_r2'])
    truth = load('reports/test_metrics.json')
    predictions = pd.read_csv(ROOT / 'reports/test_predictions.csv')
    np.testing.assert_allclose(predictions.actual_PE_MW, parts['test'].PE, rtol=0, atol=1e-12)
    y, yhat = predictions.actual_PE_MW, predictions.predicted_PE_MW
    recomputed = {'test_mae': float(mean_absolute_error(y, yhat)), 'test_rmse': float(np.sqrt(mean_squared_error(y, yhat))), 'test_r2': float(r2_score(y, yhat))}
    checks['test_metrics_recomputed'] = len(y) == truth['n_test'] == 1906 and all(np.isclose(recomputed[k], truth[k], rtol=0, atol=1e-12) for k in recomputed)
    checks['test_is_selected_run'] = truth['run_id'] == winner['run_id'] and truth['model_version'] == versions['selected']['model_version']
    api = {phase: load(f'reports/api_{phase}.json') for phase in ['v1', 'v2', 'rollback', 'compose']}
    containers = {phase: load(f'reports/container_{phase}.json') for phase in api}
    expected_cases = {'empty_batch', 'missing_feature', 'numeric_string', 'extra_feature', 'invalid_humidity', 'wrong_body', 'too_many_rows', 'nonfinite_input'}
    for phase, values in api.items():
        label = 'v1' if phase in ['v1', 'rollback'] else 'v2'
        metadata = load(f'exported_models/{label}/metadata.json')
        expectation = load(f'exported_models/{label}/expected_predictions.json')
        checks[f'api_export_metadata_{phase}'] = all(values[k] == metadata[k] for k in ['run_id', 'model_version', 'model_sha256'])
        np.testing.assert_allclose(values['valid_response']['predictions_MW'], expectation['predictions_MW'], rtol=1e-10, atol=1e-9)
        checks[f'api_numeric_match_{phase}'] = values['valid_response']['unit'] == 'MW' and values['valid_response']['model_version'] == metadata['model_version']
        checks[f'eight_negative_requests_{phase}'] = {c['case'] for c in values['invalid_requests']} == expected_cases and all(c['actual_status'] == c['expected_status'] == 422 for c in values['invalid_requests'])
        checks[f'healthcheck_{phase}'] = load(f'reports/health_{phase}.json')['Status'] == containers[phase]['health'] == 'healthy'
        image = load(f'reports/image_{label}.json')
        checks[f'image_identity_{phase}'] = containers[phase]['image_id'] == image['Id'] and containers[phase]['image_tag'] in image['RepoTags'] and image['Architecture'] == 'arm64'
    same_input = all(value['valid_request'] == api['v1']['valid_request'] for value in api.values())
    checks['same_request_all_deployments'] = same_input
    checks['update_changes_version_and_image'] = api['v1']['model_version'] != api['v2']['model_version'] and containers['v1']['image_id'] != containers['v2']['image_id']
    checks['rollback_same_image_and_model'] = containers['v1']['image_id'] == containers['rollback']['image_id'] and api['v1']['model_sha256'] == api['rollback']['model_sha256']
    np.testing.assert_allclose(api['v1']['valid_response']['predictions_MW'], api['rollback']['valid_response']['predictions_MW'], rtol=1e-10, atol=1e-9)
    checks['compose_uses_v2_image_model'] = containers['v2']['image_id'] == containers['compose']['image_id'] and api['v2']['model_sha256'] == api['compose']['model_sha256'] and api['v2']['model_version'] == api['compose']['model_version']
    np.testing.assert_allclose(api['v2']['valid_response']['predictions_MW'], api['compose']['valid_response']['predictions_MW'], rtol=1e-10, atol=1e-9)
    cross_delta = float(np.max(np.abs(np.asarray(api['v2']['valid_response']['predictions_MW']) - api['compose']['valid_response']['predictions_MW'])))
    checks['fourteen_original_screenshots'] = len(list((ROOT / 'screenshots').glob('[0-9][0-9]-*.png'))) == 14
    # Alias was recorded in the user's console, rather than inferred from metrics.
    compose_log = (ROOT / 'reports/logs/compose.log').read_text()
    checks['champion_selected_after_compose_recorded'] = f"champion -> {versions['selected']['model_version']}" in compose_log
    result = {
        'scope': 'Offline verification of submitted artifacts, not a new Ubuntu/Docker/MLflow execution',
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'all_checks_passed': bool(all(checks.values())), 'checks': {k: bool(v) for k, v in checks.items()},
        'test_metrics_recomputed': recomputed,
        'v2_vs_compose_max_abs_difference_MW': cross_delta,
        'numeric_tolerance': {'rtol': 1e-10, 'atol_MW': 1e-9},
        'original_collector': load('reports/evidence_status.json'),
        'collector_diagnosis': 'Original update_verified=false used exact dict/float equality. v2 and Compose predictions differ by one float64 ULP; model hashes and image IDs match. Original evidence is retained unchanged.',
        'limitations': 'Original joblib binaries and stopped MLflow database were not included in the submitted ZIP. Live services are not contacted. Model hashes are compared between metadata records, not recomputed from absent binaries.',
    }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true', help='Write independent reports/verification.json; original records are never modified')
    args = parser.parse_args()
    result = validate()
    if args.write:
        (ROOT / 'reports/verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['all_checks_passed'] else 1)
