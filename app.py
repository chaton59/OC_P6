"""Gradio app for Credit Scoring using an MLflow LightGBM model."""

import json
from typing import Any, Dict

# Compatibility shim: HF Spaces may install a `huggingface_hub` that no longer
# exports `HfFolder` (used by older Gradio 4.x oauth). Try to import and patch
# the real `huggingface_hub` when available; only create a minimal shim if the
# package is absent so we don't shadow the real implementation.
import os
try:
    import huggingface_hub as _hf  # prefer the real package when available
except Exception:
    _hf = None

if _hf is not None:
    # Patch only missing symbols to preserve real package behaviour
    if not hasattr(_hf, 'HfFolder'):
        class HfFolder:
            @staticmethod
            def get_token():
                return os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
        _hf.HfFolder = HfFolder
    if not hasattr(_hf, 'whoami'):
        def whoami(token=None):
            return {}
        _hf.whoami = whoami
else:
    import sys, types
    _mod = types.ModuleType('huggingface_hub')
    class HfFolder:
        @staticmethod
        def get_token():
            return os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    def whoami(token=None):
        return {}
    _mod.HfFolder = HfFolder
    _mod.whoami = whoami
    sys.modules['huggingface_hub'] = _mod

import re as _re

import gradio as gr
import mlflow
import mlflow.lightgbm
import pandas as pd
import numpy as np
from pathlib import Path

# Optional joblib for preprocessor persistence; fall back to None if unavailable
try:
	import joblib
except Exception:
	joblib = None

# Lightweight transformer to accept "raw" payloads (categorical strings, booleans)
from src.preprocessing import RawToModelTransformer


# Load the model once at startup for efficiency (lazy loading for tests).
MODEL = None

def _load_model():
	"""Lazy-load the model on first use.

	Behavior:
	- Try local LightGBM model file `models/lightgbm.txt` first (fastest, works in Docker/HF).
	- If that fails, try the MLflow Model Registry as fallback (for local dev with MLflow server).
	"""
	global MODEL
	if MODEL is None:
		import lightgbm as lgb

		# 1) Local model file (primary — portable for Docker / HF Spaces)
		candidate_paths = [
			Path(__file__).resolve().parent / "models" / "lightgbm.txt",
			Path.cwd() / "models" / "lightgbm.txt",
		]
		env_path = os.environ.get("LOCAL_MODEL_PATH")
		if env_path:
			candidate_paths.insert(0, Path(env_path))

		for p in candidate_paths:
			if p.exists():
				try:
					MODEL = lgb.Booster(model_file=str(p))
					print(f"Loaded local LightGBM model from {p}")
					return MODEL
				except Exception as err:
					print(f"Warning: failed to load {p}: {err}")

		# 2) Fallback: MLflow Model Registry (for local dev)
		try:
			MODEL = mlflow.lightgbm.load_model("models:/LightGBM/Production")
			print("Loaded model from MLflow registry")
			return MODEL
		except Exception as mlflow_err:
			raise RuntimeError(
				f"No local model found at {[str(p) for p in candidate_paths]} "
				f"and MLflow registry failed: {mlflow_err}. "
				"Place the model at `models/lightgbm.txt` or set LOCAL_MODEL_PATH."
			) from mlflow_err

	return MODEL


# Preprocessor (accept "raw" input and map to model features)
PREPROCESSOR = None

def _load_preprocessor():
	"""Load or instantiate the RawToModelTransformer.

	If a serialized preprocessor exists at `models/preprocessor.joblib` it is loaded.
	Otherwise an instance of `RawToModelTransformer` is created and (if possible)
	saved for future runs.
	"""
	global PREPROCESSOR
	if PREPROCESSOR is not None:
		return PREPROCESSOR

	path = Path("models") / "preprocessor.joblib"
	if path.exists() and joblib is not None:
		try:
			PREPROCESSOR = joblib.load(path)
			return PREPROCESSOR
		except Exception:
			# continue to create a fresh instance
			PREPROCESSOR = None

	# create a fresh transformer (it will infer feature names from CSV)
	PREPROCESSOR = RawToModelTransformer()
	# try to persist for later
	if joblib is not None:
		try:
			path.parent.mkdir(parents=True, exist_ok=True)
			joblib.dump(PREPROCESSOR, path)
		except Exception:
			# non-fatal: continue without persistence
			pass

	return PREPROCESSOR

def _validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
	"""Basic validation on input payload.

	Raises:
		ValueError: If the payload is invalid.
	"""
	if not isinstance(payload, dict):
		raise ValueError("Le JSON doit être un objet (clé/valeur).")

	if not payload:
		raise ValueError("Le JSON est vide.")

	for key, value in payload.items():
		if value is None:
			raise ValueError(f"La valeur de '{key}' est manquante.")

		if isinstance(value, (list, dict)):
			raise ValueError(f"La valeur de '{key}' doit être scalaire.")

	return payload


def _parse_json_line(json_line: str) -> pd.DataFrame:
	"""Parse a single JSON line into a one-row DataFrame."""
	try:
		raw = json.loads(json_line)
	except json.JSONDecodeError as exc:
		raise ValueError("JSON invalide. Vérifie la syntaxe.") from exc

	payload = _validate_payload(raw)

	# Build a single-row DataFrame and sanitize common problematic inputs:
	# - convert empty strings to NaN so numeric coercion / imputation works
	# - convert string booleans to actual booleans ("True"/"False")
	df = pd.DataFrame([payload])
	df = df.replace({"": np.nan, "True": True, "False": False})

	# Force all columns to numeric dtypes (LightGBM rejects object/str columns).
	# Booleans become 1/0, strings that are still present become NaN.
	for col in df.columns:
		df[col] = pd.to_numeric(df[col], errors='coerce')

	# Try to apply a lightweight preprocessor to accept "raw" payloads
	# The transformer maps categorical strings (ex. NAME_CONTRACT_TYPE) to the
	# one-hot columns expected by the trained model. On any failure we keep the
	# original dataframe and rely on column reindexing later.
	try:
		pre = _load_preprocessor()
		if pre is not None:
			# transform handles both full preprocessed rows and raw rows
			df = pre.transform(df)
	except Exception:
		# Non-fatal: continue with the original df (alignment step will fill missing)
		pass

	return df


def _get_model_feature_names(model) -> list | None:
	"""Try to obtain the model's expected feature names.

	Tries common LightGBM / sklearn attributes first, then falls back to
	reading the header of `data/processed/features_train.csv`.
	Returns a list of column names or None if not found.
	"""
	# 1) common LightGBM / sklearn attributes
	try:
		fn = getattr(model, "feature_name", None)
		if callable(fn):
			names = list(fn())
			if names:
				return names
	except Exception:
		pass

	names = getattr(model, "feature_name_", None)
	if isinstance(names, (list, tuple)):
		return list(names)

	# LightGBM scikit-learn wrapper exposes `booster_`
	try:
		if hasattr(model, "booster_") and getattr(model.booster_, "feature_name", None):
			return list(model.booster_.feature_name())
	except Exception:
		pass

	# 2) Fallback to header from the preprocessed training CSV
	try:
		header_path = Path("data/processed/features_train.csv")
		if header_path.exists():
			df_header = pd.read_csv(header_path, nrows=0)
			cols = [c for c in df_header.columns if c not in ("SK_ID_CURR", "TARGET")]
			# Apply same sanitization as training notebook (spaces → _, non-alnum → _)
			cols = [_re.sub(r'[^a-zA-Z0-9_]', '_', c.replace(' ', '_')) for c in cols]
			if cols:
				return cols
	except Exception:
		pass

	return None


def _predict(json_line: str, threshold: float = 0.4) -> str:
	"""Predict default probability and return a formatted response."""
	try:
		df = _parse_json_line(json_line)

		model = _load_model()

		# Align input columns to the model's expected features (fill missing with 0).
		expected = _get_model_feature_names(model)
		if expected:
			# keep only the expected columns and fill missing ones with 0
			df = df.reindex(columns=expected, fill_value=0)

		# Final safety: ensure every column is numeric (LightGBM requirement)
		for col in df.columns:
			df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

		try:
			proba = float(model.predict_proba(df)[:, 1][0])
		except AttributeError:
			# Fallback for models exposing predict() returning probabilities.
			proba = float(model.predict(df)[0])

		if not 0.0 <= proba <= 1.0:
			raise ValueError("La probabilité prédite est hors de l'intervalle [0, 1].")

		score = int(proba * 1000)
		decision = "Accordé" if proba < threshold else "Refusé"

		return (
			f"Score: {score}\n"
			f"Probabilité de défaut: {proba:.4f}\n"
			f"Décision: {decision}"
		)

	except ValueError as exc:
		return f"Erreur: {exc}"
	except KeyError as exc:
		return f"Erreur: colonne manquante ({exc})."
	except TypeError as exc:
		return f"Erreur: type invalide ({exc})."
	except Exception as exc:  # noqa: BLE001
		return f"Erreur inattendue: {exc}"


def build_demo() -> gr.Blocks:
	"""Build and return the Gradio Blocks demo."""
	with gr.Blocks(title="Credit Scoring API") as demo:
		gr.Markdown(
			"# Credit Scoring API\n"
			"Saisis une seule ligne JSON avec les variables d'entrée.\n"
			"Le modèle LightGBM retourne une probabilité de défaut, un score, et une décision."
		)

		with gr.Row():
			input_json = gr.Textbox(
				label="JSON (ligne unique)",
				lines=12,
				max_lines=30,
				placeholder='{"feature_1": 0.5, "feature_2": 123, "feature_3": "A"}',
			)

		output_text = gr.Textbox(
			label="Résultat",
			lines=5,
		)

		predict_btn = gr.Button("Prédire")
		predict_btn.click(
			fn=_predict,
			inputs=[input_json],
			outputs=[output_text],
		)

		gr.Examples(
			examples=[
				# Placeholder examples; replace with real feature JSON when ready.
				'{"feature_1": 0.2, "feature_2": 45, "feature_3": "B"}',
				'{"feature_1": 0.9, "feature_2": 120, "feature_3": "A"}',
			],
			inputs=input_json,
			label="Exemples",
		)

		gr.Markdown(
			"**Note:** Le seuil de décision est fixé à 0.4 par défaut."
		)

	return demo


demo = build_demo()

if __name__ == "__main__":
	demo.launch(
		server_name="0.0.0.0",
		server_port=int(os.environ.get("PORT", 7860)),
	)
