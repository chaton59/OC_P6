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

import gradio as gr
import mlflow
import mlflow.lightgbm
import pandas as pd


# Load the model once at startup for efficiency (lazy loading for tests).
# If the "Production" stage is not available, MLflow will fall back to the latest version.
MODEL_URI = "models:/LightGBM/Production"
MODEL = None

def _load_model():
	"""Lazy-load the model on first use."""
	global MODEL
	if MODEL is None:
		try:
			MODEL = mlflow.lightgbm.load_model(MODEL_URI)
		except Exception as e:
			raise RuntimeError(f"Failed to load model from {MODEL_URI}: {e}") from e
	return MODEL


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
	df = pd.DataFrame([payload])

	# TODO: Apply preprocessing if needed (e.g., load a preprocessor from src/ or joblib)
	# from src.preprocessing import preprocessor
	# df = preprocessor.transform(df)
	# or
	# import joblib
	# preprocessor = joblib.load("path/to/preprocessor.joblib")
	# df = preprocessor.transform(df)

	return df


def _predict(json_line: str, threshold: float = 0.4) -> str:
	"""Predict default probability and return a formatted response."""
	try:
		df = _parse_json_line(json_line)

		model = _load_model()
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
	demo.launch()
