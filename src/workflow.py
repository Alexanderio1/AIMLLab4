"""Reproducible CCPP experiment workflow. Run from repository root."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import zipfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"
SEED = 42
SOURCE = "https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant"
DOWNLOAD = "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip"
UNITS = {"AT": "degC", "V": "cm Hg", "AP": "mbar", "RH": "%", "PE": "MW", "MAE": "MW", "RMSE": "MW", "R2": "dimensionless"}
EXPERIMENT = "lab4-ccpp-variant17"
MODEL_NAME = "ccpp-power"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def connect():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    return MlflowClient()


def split_frame(frame):
    """Group identical feature vectors, even if their target values differ."""
    groups = pd.util.hash_pandas_object(frame[FEATURES], index=False).to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
    dev, test = next(first.split(frame, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED)
    train_part, val_part = next(second.split(frame.iloc[dev], groups=groups[dev]))
    indices = {"train": dev[train_part], "validation": dev[val_part], "test": test}
    sets = {key: set(groups[pos]) for key, pos in indices.items()}
    assert sets["train"].isdisjoint(sets["validation"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["validation"].isdisjoint(sets["test"])
    assert sum(map(len, indices.values())) == len(frame)
    return indices


def prepare():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "ccpp.zip"
    if not archive.exists():
        print("Downloading", DOWNLOAD, flush=True)
        response = requests.get(DOWNLOAD, timeout=(20, 180))
        response.raise_for_status()
        temp = archive.with_suffix(".tmp")
        temp.write_bytes(response.content)
        if not zipfile.is_zipfile(temp):
            raise RuntimeError("UCI response is not a ZIP archive")
        temp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        matches = [n for n in z.namelist() if Path(n).name == "Folds5x2_pp.xlsx" and not n.startswith("__MACOSX/")]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one Folds5x2_pp.xlsx, got {matches}")
        workbook = raw / "Folds5x2_pp.xlsx"
        workbook.write_bytes(z.read(matches[0]))
    excel = pd.ExcelFile(workbook, engine="openpyxl")
    selected_sheet = excel.sheet_names[0]
    original = pd.read_excel(excel, sheet_name=selected_sheet)
    if list(original.columns) != FEATURES + [TARGET]:
        raise RuntimeError(f"Unexpected columns: {list(original.columns)}")
    if len(original) != 9568:
        raise RuntimeError(f"UCI first sheet should have 9568 rows, got {len(original)}. Check source.")
    if original.isna().any().any() or not np.isfinite(original.to_numpy(dtype=float)).all():
        raise RuntimeError("Missing or nonfinite data: investigate instead of silently deleting")
    # UCI CCPP contains measured RH values up to 100.16%. Preserve these
    # observations; a theoretical 100% cutoff would reject the source dataset.
    valid = (original.RH >= 0) & (original.RH <= 100.16) & (original.AP > 0) & (original.V > 0)
    if not valid.all():
        examples = original.loc[~valid, FEATURES].head(5).to_dict(orient="records")
        raise RuntimeError(f"Source values violate input constraints (RH 0..100.16%, AP > 0, V > 0): {examples}")
    clean = original.drop_duplicates().copy()
    clean.insert(0, "row_id", clean.index.astype(int))
    clean = clean.reset_index(drop=True)
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    clean.to_csv("data/processed/data.csv", index=False)
    indices = split_frame(clean)
    split = {key: clean.iloc[pos].row_id.astype(int).tolist() for key, pos in indices.items()}
    for key, pos in indices.items():
        clean.iloc[pos].to_csv(f"data/processed/{key}.csv", index=False)
    split.update({"seed": SEED, "method": "GroupShuffleSplit on identical feature vectors; approximately 60/20/20", "features": FEATURES, "target": TARGET})
    save("data/split.json", split)
    audit = {
        "source": SOURCE, "download_url": DOWNLOAD, "archive_sha256": sha(archive),
        "workbook_sha256": sha(workbook), "available_sheets": excel.sheet_names,
        "selected_sheet": selected_sheet, "sheet_policy": "Only the first sheet is used; other sheets are not concatenated",
        "original_rows": len(original), "rows_after_deduplication": len(clean),
        "exact_duplicates_removed": len(original) - len(clean),
        "feature_ranges": {key: {"min": float(original[key].min()), "max": float(original[key].max())} for key in FEATURES},
        "humidity_above_100_count": int((original.RH > 100).sum()),
        "humidity_policy": "Preserve measured UCI RH values up to 100.16%; no clipping and no removal for RH > 100%",
        "duplicate_feature_rows_after_deduplication": int(clean.duplicated(FEATURES).sum()),
        "split_counts": {key: len(pos) for key, pos in indices.items()},
        "split_sha256": sha("data/split.json"), "data_sha256": sha("data/processed/data.csv"),
        "units": UNITS, "seed": SEED,
        "limitations": "No timestamps or plant IDs: random group split estimates interpolation in this dataset, not forecasting future operating regimes. No causal interpretation.",
    }
    save("reports/data_audit.json", audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def dataset(part):
    data = pd.read_csv(f"data/processed/{part}.csv")
    return data[FEATURES], data[TARGET]


def metrics(y, pred, prefix):
    return {f"{prefix}_mae": float(mean_absolute_error(y, pred)),
            f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y, pred))),
            f"{prefix}_r2": float(r2_score(y, pred))}


def configurations():
    return [
        ("ridge-a0.1", "Ridge", {"alpha": .1}),
        ("ridge-a1", "Ridge", {"alpha": 1.0}),
        ("ridge-a10", "Ridge", {"alpha": 10.0}),
        ("rf-depth8", "RandomForestRegressor", {"n_estimators": 120, "max_depth": 8, "min_samples_leaf": 2}),
        ("rf-depth16", "RandomForestRegressor", {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 1}),
        ("rf-unlimited", "RandomForestRegressor", {"n_estimators": 250, "max_depth": None, "min_samples_leaf": 1}),
    ]


def make_model(algorithm, params):
    if algorithm == "Ridge":
        return Pipeline([("preprocess", StandardScaler()), ("regressor", Ridge(**params))])
    return Pipeline([("preprocess", "passthrough"), ("regressor", RandomForestRegressor(**params, random_state=SEED, n_jobs=2))])


def train():
    connect()
    mlflow.set_experiment(EXPERIMENT)
    X, y = dataset("train")
    Xval, yval = dataset("validation")
    audit = read("reports/data_audit.json")
    records = []
    for name, algorithm, params in configurations():
        model = make_model(algorithm, params)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({"algorithm": algorithm, **params, "seed": SEED, "dataset": "UCI CCPP first sheet",
                               "n_train": len(X), "n_validation": len(Xval), "n_test": audit["split_counts"]["test"],
                               "preprocessing": "StandardScaler" if algorithm == "Ridge" else "passthrough", "n_jobs": 2 if algorithm != "Ridge" else 1})
            mlflow.set_tags({"variant": "17", "selection_metric": "val_mae", "selection_direction": "minimize",
                             "dataset_sha256": audit["data_sha256"], "split_sha256": audit["split_sha256"],
                             "target": "PE", "target_unit": "MW", "mae_unit": "MW", "rmse_unit": "MW", "feature_units": json.dumps({k: UNITS[k] for k in FEATURES})})
            start = time.perf_counter()
            model.fit(X, y)
            elapsed = time.perf_counter() - start
            pred = model.predict(Xval)
            values = metrics(yval, pred, "val")
            mlflow.log_metrics({**values, "fit_seconds": elapsed})
            mlflow.log_dict(UNITS, "units.json")
            mlflow.log_dict(audit, "data_audit.json")
            mlflow.log_artifact("data/split.json")
            mlflow.log_artifact("requirements.lock.txt", artifact_path="source")
            mlflow.log_artifact(__file__, artifact_path="source")
            mlflow.log_input(mlflow.data.from_pandas(X.assign(PE=y), source=SOURCE, targets="PE"), context="training")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(yval, pred, s=5, alpha=.4)
            lo, hi = min(yval.min(), pred.min()), max(yval.max(), pred.max())
            ax.plot([lo, hi], [lo, hi], "r--")
            ax.set(xlabel="Actual PE (MW)", ylabel="Predicted PE (MW)", title=name + " — validation")
            fig.tight_layout()
            fig.savefig(f"reports/validation_{name}.png", dpi=160)
            mlflow.log_figure(fig, "validation_predictions.png")
            plt.close(fig)
            info = mlflow.sklearn.log_model(model, name="model", input_example=X.head(3), signature=infer_signature(X, model.predict(X)),
                                           pip_requirements=Path("requirements.lock.txt").read_text().splitlines())
            record = {"name": name, "algorithm": algorithm, "parameters": params, **values, "fit_seconds": elapsed,
                      "run_id": run.info.run_id, "model_uri": info.model_uri}
            records.append(record)
            print(json.dumps(record), flush=True)
    # Deterministic tie break: lower MAE, then name. Test data are not accessed here.
    records.sort(key=lambda r: (r["val_mae"], r["name"]))
    save("reports/experiments.json", records)
    table = pd.DataFrame([{**r, "parameters": json.dumps(r["parameters"])} for r in records])
    table.to_csv("reports/experiments.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(table["name"], table["val_mae"])
    ax.invert_yaxis()
    ax.set(xlabel="Validation MAE (MW), lower is better", title="Six experiments — validation only")
    fig.tight_layout()
    fig.savefig("reports/experiment_comparison.png", dpi=160)
    plt.close(fig)
    print("\nSIX EXPERIMENTS — selection by validation MAE, MW")
    print(table[["name", "algorithm", "val_mae", "val_rmse", "val_r2", "run_id"]].to_string(index=False))


def register():
    client = connect()
    records = read("reports/experiments.json")
    winner = records[0]
    baseline = next(r for r in records if r["algorithm"] != winner["algorithm"])
    versions = {}
    for role, record in [("baseline", baseline), ("selected", winner)]:
        mv = mlflow.register_model(record["model_uri"], MODEL_NAME, await_registration_for=120)
        client.set_model_version_tag(MODEL_NAME, mv.version, "role", role)
        client.set_model_version_tag(MODEL_NAME, mv.version, "target_unit", "MW")
        client.update_model_version(MODEL_NAME, mv.version, description=f"{role}: {record['name']}; validation MAE={record['val_mae']:.6f} MW")
        versions[role] = {**record, "model_version": str(mv.version)}
    client.set_registered_model_alias(MODEL_NAME, "champion", versions["selected"]["model_version"])
    save("reports/registry.json", {"model_name": MODEL_NAME, "alias": "champion", "versions": versions,
                                   "selection_rule": "Minimum validation MAE; baseline is best configuration of the other algorithm. Test excluded from selection."})
    print(json.dumps(read("reports/registry.json"), ensure_ascii=False, indent=2))


def evaluate():
    connect()
    chosen = read("reports/registry.json")["versions"]["selected"]
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{chosen['model_version']}")
    Xtest, ytest = dataset("test")
    pred = model.predict(Xtest)
    values = metrics(ytest, pred, "test")
    with mlflow.start_run(run_id=chosen["run_id"]):
        mlflow.log_metrics(values)
        mlflow.set_tag("test_usage", "Final evaluation after validation-only selection; no retraining")
    result = {"model_version": chosen["model_version"], "run_id": chosen["run_id"], "n_test": len(Xtest), **values, "units": UNITS}
    save("reports/test_metrics.json", result)
    pd.DataFrame({"actual_PE_MW": ytest, "predicted_PE_MW": pred}).to_csv("reports/test_predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ytest - pred, bins=40)
    ax.set(xlabel="Actual minus predicted PE (MW)", ylabel="Count", title="Selected model — final test residuals")
    fig.tight_layout()
    fig.savefig("reports/test_residuals.png", dpi=160)
    plt.close(fig)
    print("FINAL TEST (not used for selection)")
    print(json.dumps(result, indent=2))


def export(role, label):
    client = connect()
    selected = read("reports/registry.json")["versions"][role]
    version = selected["model_version"]
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    alias = client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert str(alias.version) == version
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    root = ROOT / "exported_models" / label
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, root / "model.joblib")
    save(root / "metadata.json", {"model_name": MODEL_NAME, "model_version": version, "run_id": selected["run_id"],
                                 "algorithm": selected["algorithm"], "parameters": selected["parameters"],
                                 "model_sha256": sha(root / "model.joblib"), "features": FEATURES, "target": TARGET,
                                 "units": UNITS, "validation_mae_MW": selected["val_mae"]})
    X, _ = dataset("validation")
    example = X.head(3)
    save(root / "sample_request.json", {"instances": example.to_dict(orient="records")})
    save(root / "expected_predictions.json", {"predictions_MW": model.predict(example).tolist()})
    print("EXPORTED", label, "registry version", version, "alias champion")
    print((root / "metadata.json").read_text())


def check(label, phase):
    root = ROOT / "exported_models" / label
    metadata = read(root / "metadata.json")
    model = joblib.load(root / "model.joblib")
    body = read(root / "sample_request.json")
    expected = model.predict(pd.DataFrame(body["instances"], columns=FEATURES))
    base = "http://127.0.0.1:8000"
    for _ in range(90):
        try:
            response = requests.get(base + "/health", timeout=2)
            if response.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    else:
        raise RuntimeError("API did not become healthy in 90 seconds")
    response = requests.get(base + "/metadata", timeout=10)
    response.raise_for_status()
    actual_meta = response.json()
    assert actual_meta == metadata, "Wrong model/image deployed"
    response = requests.post(base + "/predict", json=body, timeout=20)
    response.raise_for_status()
    actual = response.json()
    assert actual["model_version"] == metadata["model_version"]
    np.testing.assert_allclose(actual["predictions_MW"], expected, rtol=1e-10, atol=1e-9)
    requests.get(base + "/docs", timeout=10).raise_for_status()
    cases = []
    invalid = [
        ("empty_batch", {"instances": []}),
        ("missing_feature", {"instances": [{k: v for k, v in body["instances"][0].items() if k != "AT"}]}),
        ("numeric_string", {"instances": [{**body["instances"][0], "AT": "20"}]}),
        ("extra_feature", {"instances": [{**body["instances"][0], "PE": 450.0}]}),
        ("invalid_humidity", {"instances": [{**body["instances"][0], "RH": 101.0}]}),
        ("wrong_body", {"data": body["instances"]}),
        ("too_many_rows", {"instances": body["instances"][:1] * 1001}),
    ]
    for name, bad in invalid:
        r = requests.post(base + "/predict", json=bad, timeout=20)
        assert r.status_code == 422, f"{name}: {r.status_code} {r.text}"
        cases.append({"case": name, "expected_status": 422, "actual_status": r.status_code, "request": bad, "response": r.json()})
    nonfinite = json.dumps({"instances": [{**body["instances"][0], "AT": float("nan")}]})
    r = requests.post(base + "/predict", data=nonfinite, headers={"Content-Type": "application/json"}, timeout=20)
    assert r.status_code == 422, f"nonfinite input: {r.status_code} {r.text}"
    cases.append({"case": "nonfinite_input", "expected_status": 422, "actual_status": r.status_code, "request_raw": nonfinite, "response": r.json()})
    result = {"phase": phase, "model_version": metadata["model_version"], "run_id": metadata["run_id"],
              "model_sha256": metadata["model_sha256"], "valid_request": body, "valid_response": actual,
              "expected_predictions_MW": expected.tolist(), "max_abs_difference_MW": float(np.max(np.abs(np.array(actual["predictions_MW"]) - expected))),
              "invalid_requests": cases, "all_checks_passed": True}
    save(f"reports/api_{phase}.json", result)
    print("API CHECK:", phase, "version", metadata["model_version"])
    print("POST /predict request:", json.dumps(body))
    print("POST /predict 200:", json.dumps(actual))
    print("API == exported model; max difference (MW):", result["max_abs_difference_MW"])
    print("Invalid requests:", ", ".join(c["case"] + "=422" for c in cases))
    print("ALL API CHECKS PASSED")


def environment():
    import importlib.metadata
    packages = {name: importlib.metadata.version(name) for name in ["mlflow", "scikit-learn", "pandas", "numpy", "fastapi", "uvicorn", "openpyxl"]}
    result = {"python": sys.version, "architecture": platform.machine(), "platform": platform.platform(), "packages": packages,
              "os_release": Path("/etc/os-release").read_text() if Path("/etc/os-release").exists() else "unavailable"}
    save("reports/environment.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["prepare", "train", "register", "evaluate", "environment"]:
        sub.add_parser(name)
    ex = sub.add_parser("export")
    ex.add_argument("--role", choices=["baseline", "selected"], required=True)
    ex.add_argument("--label", choices=["v1", "v2"], required=True)
    ch = sub.add_parser("check")
    ch.add_argument("--label", choices=["v1", "v2"], required=True)
    ch.add_argument("--phase", choices=["v1", "v2", "rollback", "compose"], required=True)
    args = p.parse_args()
    if args.command == "export":
        export(args.role, args.label)
    elif args.command == "check":
        check(args.label, args.phase)
    else:
        globals()[args.command]()
