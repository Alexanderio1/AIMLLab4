"""Offline inference: container does not require MLflow or network access."""
import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

FEATURES = ["AT", "V", "AP", "RH"]


class Instance(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    AT: float = Field(description="Ambient temperature, degrees Celsius")
    V: float = Field(gt=0, description="Exhaust vacuum, cm Hg")
    AP: float = Field(gt=0, description="Ambient pressure, millibar")
    RH: float = Field(ge=0, le=100, description="Relative humidity, percent")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    instances: list[Instance] = Field(min_length=1, max_length=1000)


@asynccontextmanager
async def lifespan(app):
    root = Path(os.getenv("MODEL_DIR", "exported_models/v1"))
    metadata = json.loads((root / "metadata.json").read_text())
    digest = hashlib.sha256((root / "model.joblib").read_bytes()).hexdigest()
    if digest != metadata["model_sha256"]:
        raise RuntimeError("Exported model checksum mismatch")
    app.state.model = joblib.load(root / "model.joblib")
    app.state.metadata = metadata
    yield


app = FastAPI(title="CCPP power prediction — variant 17", version="1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # Some JSON parsers accept NaN/Infinity. Keep error responses valid JSON,
    # including when the rejected input itself contains a nonfinite number.
    errors = json.loads(json.dumps(exc.errors(), default=str), parse_constant=str)
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health():
    return {"status": "ok", "model_version": app.state.metadata["model_version"]}


@app.get("/metadata")
def metadata():
    return app.state.metadata


@app.post("/predict")
def predict(body: PredictionRequest):
    frame = pd.DataFrame([row.model_dump() for row in body.instances], columns=FEATURES)
    values = app.state.model.predict(frame)
    return {
        "model_version": app.state.metadata["model_version"],
        "predictions_MW": values.tolist(),
        "unit": "MW",
    }
