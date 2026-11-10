"""Synthetic fixtures test infrastructure; they are not laboratory experiment results."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app import app
from workflow import FEATURES, split_frame


@pytest.fixture
def exported(tmp_path, monkeypatch):
    frame = pd.DataFrame({"AT": [10., 20., 30., 35.], "V": [30., 40., 50., 60.], "AP": [1000., 1005., 1010., 1015.], "RH": [30., 40., 50., 60.]})
    model = Pipeline([("scale", StandardScaler()), ("regressor", Ridge())]).fit(frame, [480., 460., 440., 430.])
    joblib.dump(model, tmp_path / "model.joblib")
    digest = hashlib.sha256((tmp_path / "model.joblib").read_bytes()).hexdigest()
    (tmp_path / "metadata.json").write_text(json.dumps({"model_version": "test-fixture", "model_sha256": digest}))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    return frame, model, tmp_path


def test_api_predicts_exported_pipeline(exported):
    frame, model, _ = exported
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/predict", json={"instances": frame.to_dict("records")})
        assert r.status_code == 200
        np.testing.assert_allclose(r.json()["predictions_MW"], model.predict(frame), rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize("override", [
    {"AT": "20"}, {"RH": 101.}, {"AP": -1.}, {"V": 0.}, {"PE": 450.}, {"AT": None}, {"RH": True},
])
def test_api_rejects_invalid_features(exported, override):
    frame, _, _ = exported
    row = {**frame.iloc[0].to_dict(), **override}
    with TestClient(app) as client:
        assert client.post("/predict", json={"instances": [row]}).status_code == 422


def test_empty_and_missing_features(exported):
    with TestClient(app) as client:
        assert client.post("/predict", json={"instances": []}).status_code == 422
        assert client.post("/predict", json={"instances": [{"AT": 20.}]}).status_code == 422


def test_integrity_error_prevents_startup(exported):
    _, _, root = exported
    (root / "model.joblib").write_bytes(b"damaged file")
    with pytest.raises(RuntimeError, match="checksum"):
        with TestClient(app):
            pass


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_input_returns_validation_error(exported, value):
    body = '{"instances":[{"AT":' + value + ',"V":50.0,"AP":1010.0,"RH":70.0}]}'
    with TestClient(app) as client:
        response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        assert response.json()["detail"]


def test_duplicate_feature_vectors_do_not_leak():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(100, 4)), columns=FEATURES)
    frame = pd.concat([frame, frame.iloc[:20], frame.iloc[:20]], ignore_index=True)
    frame["PE"] = rng.normal(size=len(frame))
    splits = split_frame(frame)
    groups = {key: set(map(tuple, frame.iloc[pos][FEATURES].to_numpy())) for key, pos in splits.items()}
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    assert sorted(np.concatenate(list(splits.values()))) == list(range(len(frame)))
    for name in splits:
        np.testing.assert_array_equal(splits[name], split_frame(frame)[name])
