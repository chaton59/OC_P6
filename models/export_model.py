from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

try:
	from src.mlflow_config import DEFAULT_EXPERIMENT_NAME
except Exception:  # pragma: no cover - fallback si import impossible
	DEFAULT_EXPERIMENT_NAME = "OC_P6_Credit_Scoring"

# Nom du modèle enregistré et stage cible
MODEL_NAME = "LightGBM"
MODEL_STAGE = "Production"


def resolve_tracking_uri() -> str:
	env_uri = os.getenv("MLFLOW_TRACKING_URI")
	if env_uri:
		return env_uri
	local_store = Path("mlruns")
	if local_store.exists():
		return local_store.resolve().as_uri()
	return mlflow.get_tracking_uri()


tracking_uri = resolve_tracking_uri()
mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()
model_uri = None

# 1) Essaye le Model Registry avec stage (si présent)
try:
	latest_versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
	if latest_versions:
		model_version = latest_versions[0].version
		model_uri = f"models:/{MODEL_NAME}/{model_version}"
except Exception:
	model_uri = None

# 2) Sinon, prend la dernière version enregistrée (tous stages)
if model_uri is None:
	try:
		versions = client.search_model_versions(f"name='{MODEL_NAME}'")
		if versions:
			latest = max(versions, key=lambda v: int(v.version))
			model_uri = f"models:/{MODEL_NAME}/{latest.version}"
	except Exception:
		model_uri = None

# 3) Sinon, fallback sur le dernier run de l'expérience
if model_uri is None:
	experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
	try:
		experiment = mlflow.get_experiment_by_name(experiment_name)
	except Exception:
		experiment = None
	if experiment:
		try:
			runs = mlflow.search_runs(
				[experiment.experiment_id],
				order_by=["start_time DESC"],
				max_results=1,
			)
		except Exception:
			runs = None
		if runs is not None and not runs.empty:
			run_id = runs.loc[0, "run_id"]
			model_uri = f"runs:/{run_id}/model"

# 4) Fallback filesystem si la base MLflow est partiellement corrompue
if model_uri is None:
	mlruns_dir = Path("mlruns")
	if mlruns_dir.exists():
		mlmodel_files = sorted(
			mlruns_dir.rglob("MLmodel"), key=lambda p: p.stat().st_mtime, reverse=True
		)
		if mlmodel_files:
			model_uri = str(mlmodel_files[0].parent)

if model_uri is None:
	raise RuntimeError(
		"Aucun modèle trouvé. Vérifie MLFLOW_TRACKING_URI, le Model Registry, "
		"ou l'expérience MLflow."
	)

# Charge et sauvegarde en fichier simple
model = mlflow.lightgbm.load_model(model_uri)
output_path = Path("models") / "lightgbm.txt"
output_path.parent.mkdir(parents=True, exist_ok=True)

# LGBMClassifier (sklearn API) → utilise le booster interne
# Booster natif → save_model directement
import lightgbm as lgb

if isinstance(model, lgb.Booster):
	model.save_model(str(output_path))
elif hasattr(model, "booster_"):
	model.booster_.save_model(str(output_path))
else:
	# Dernier recours : sérialisation joblib
	import joblib
	output_path = output_path.with_suffix(".pkl")
	joblib.dump(model, str(output_path))

print(f"Modèle exporté depuis {model_uri} vers {output_path}")