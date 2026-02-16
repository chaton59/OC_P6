"""Gradio app for Credit Scoring using an MLflow LightGBM model."""

import json
from typing import Any, Dict

import gradio as gr
import lightgbm as lgb
import pandas as pd


# Load the model once at startup for efficiency.
# Use a local model file for portability in Docker/Hugging Face deployments.
MODEL = lgb.Booster(model_file="models/lightgbm.txt")


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

		try:
			proba = float(MODEL.predict_proba(df)[:, 1][0])
		except AttributeError:
			# Fallback for models exposing predict() returning probabilities.
			proba = float(MODEL.predict(df)[0])

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
	# server_name="0.0.0.0" required for Docker/HF Spaces (listen on all interfaces)
	demo.launch(server_name="0.0.0.0", server_port=7860)
