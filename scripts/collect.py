"""Bundle allowlisted report evidence. Never include venv, credentials or Docker storage."""
import hashlib
import json
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
checks = {}
for phase in ["v1", "v2", "rollback", "compose"]:
    path = ROOT / f"reports/api_{phase}.json"
    checks[phase] = path.exists() and json.loads(path.read_text()).get("all_checks_passed") is True
for name in ["experiments.csv", "registry.json", "test_metrics.json", "data_audit.json", "environment.json"]:
    checks[name] = (ROOT / "reports" / name).exists()
if checks["experiments.csv"]:
    import csv
    checks["six_experiments"] = len(list(csv.DictReader((ROOT / "reports/experiments.csv").open()))) >= 6
if checks["registry.json"]:
    d = json.loads((ROOT / "reports/registry.json").read_text())
    checks["two_versions"] = len({v["model_version"] for v in d["versions"].values()}) >= 2
if all(checks.get(k) for k in ["v1", "v2", "rollback", "compose"]):
    d = {k: json.loads((ROOT / f"reports/api_{k}.json").read_text()) for k in ["v1", "v2", "rollback", "compose"]}
    checks["rollback_verified"] = d["v1"]["model_version"] == d["rollback"]["model_version"] and d["v1"]["valid_response"] == d["rollback"]["valid_response"]
    checks["update_verified"] = d["v1"]["model_version"] != d["v2"]["model_version"] and d["v2"]["valid_response"] == d["compose"]["valid_response"]
    checks["rollback_hash_unchanged"] = d["v1"]["model_sha256"] == d["rollback"]["model_sha256"]
if (ROOT / "reports/container_v1.json").exists() and (ROOT / "reports/container_rollback.json").exists():
    old = json.loads((ROOT / "reports/container_v1.json").read_text())
    rollback = json.loads((ROOT / "reports/container_rollback.json").read_text())
    checks["rollback_image_unchanged"] = old["image_id"] == rollback["image_id"]
for phase in ["v1", "v2", "rollback", "compose"]:
    path = ROOT / f"reports/health_{phase}.json"
    checks[f"docker_health_{phase}"] = path.exists() and json.loads(path.read_text()).get("Status") == "healthy"
screens = list((ROOT / "screenshots").glob("*"))
checks["screenshots_present"] = any(p.suffix.lower() in {".png", ".jpg", ".jpeg"} for p in screens)
status = {"collected_at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks, "all_checks_passed": all(checks.values()),
          "note": "Screenshot count does not establish their content. Inspect screenshots manually against README checklist."}
(ROOT / "reports/evidence_status.json").write_text(json.dumps(status, indent=2) + "\n")
archive = ROOT.parent / "AIMLLab4-results.zip"
roots = ["src", "scripts", "tests", "data", "reports", "screenshots", "exported_models"]
files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in {".md", ".txt", ".yaml", ".ipynb"}]
files += [ROOT / name for name in ["Dockerfile", ".gitignore", ".dockerignore"]]
for name in roots:
    files.extend((ROOT / name).rglob("*"))
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(files)):
        if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts and p.suffix not in {".joblib", ".pyc"}:
            z.write(p, arcname="AIMLLab4/" + str(p.relative_to(ROOT)))
state = ROOT / "mlflow-data"
if (state / "server.pid").exists():
    raise RuntimeError("MLflow still has a PID file; refusing an inconsistent backup")
backup = ROOT.parent / "AIMLLab4-mlflow-backup.tar.gz"
with tarfile.open(backup, "w:gz") as tar:
    tar.add(state, arcname="mlflow-data")
print(json.dumps(status, ensure_ascii=False, indent=2))
print("SEND THIS FILE:", archive, f"({archive.stat().st_size / 1024**2:.1f} MiB)")
print("KEEP LOCALLY (upload only if requested):", backup, f"({backup.stat().st_size / 1024**2:.1f} MiB)")
print("Model binaries stay in the stopped MLflow backup; results ZIP includes sources, raw data, splits, model metadata, metrics, logs and screenshots.")
if not all(checks.values()):
    print("INCOMPLETE EVIDENCE — see false checks above; do not describe them as passed.")
