"""Gradio app for Credit Scoring using an MLflow LightGBM model."""

import json
from typing import Any, Dict
# EXPLICATION : Imports nécessaires pour le logging structuré JSON
import logging
import time
from datetime import datetime
# EXPLICATION : Path pour gestion robuste des chemins de logs (multi-plateforme)
from pathlib import Path

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
		# EXPLICATION : None est accepté (LightGBM gère nativement les NaN)
		if value is not None and isinstance(value, (list, dict)):
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

	# EXPLICATION : Sanitiser les noms de colonnes pour matcher ceux attendus par le modèle.
	# Le modèle a été entraîné avec des noms sanitisés (espaces → _, caractères spéciaux → _).
	# Sans cette étape, des colonnes comme "BURO_CREDIT_ACTIVE_Bad debt_MEAN" ne matchent pas
	# "BURO_CREDIT_ACTIVE_Bad_debt_MEAN" → fill_value=0 → prédictions faussées (tout Accordé).
	df.columns = [_re.sub(r'[^a-zA-Z0-9_]', '_', c.replace(' ', '_')) for c in df.columns]

	# Force all columns to numeric dtypes (LightGBM rejects object/str columns).
	# Booleans become 1/0, strings that are still present become NaN.
	for col in df.columns:
		df[col] = pd.to_numeric(df[col], errors='coerce')

	# Try to apply a lightweight preprocessor to accept "raw" payloads
	# The transformer maps categorical strings (ex. NAME_CONTRACT_TYPE) to the
	# one-hot columns expected by the trained model. On any failure we keep the
	# original dataframe and rely on column reindexing later.
	#
	# IMPORTANT: Skip preprocessor if input is already processed data (e.g. from
	# features_train.csv / reference.csv). Detect this by checking how many input
	# columns match expected model features. If >50% match, data is already
	# processed — running the preprocessor would replace NaN with median values,
	# destroying the signal that LightGBM uses for missing-value splits.
	try:
		pre = _load_preprocessor()
		if pre is not None:
			expected_feats = set(pre.get_feature_names_out()) if hasattr(pre, 'get_feature_names_out') else set()
			overlap = len(set(df.columns) & expected_feats)
			if expected_feats and overlap / len(expected_feats) > 0.5:
				# Data is already processed — skip preprocessor to avoid double processing
				pass
			else:
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


# EXPLICATION : Fonction helper pour logger chaque prédiction avec tous les champs requis
# IMPORTANT : Écrit DIRECTEMENT dans le fichier (pas de FileHandler)
# pour éviter les problèmes d'interférence avec Gradio/autres loggers
def log_prediction(input_raw: str, input_features: dict, output_proba: float, 
                   output_decision: str, execution_time_ms: float, error: str = None):
	"""Log une prédiction au format JSON structuré dans logs/predictions.jsonl."""
	try:
		# EXPLICATION : Crée le dossier logs si n'existe pas
		_log_dir = Path("logs")
		_log_dir.mkdir(parents=True, exist_ok=True)
		
		# EXPLICATION : Construit l'entrée JSON
		log_entry = {
			"timestamp": datetime.utcnow().isoformat() + "Z",
			"input_raw": input_raw,
			"input_features": input_features,
			"output_proba": round(output_proba, 4) if output_proba is not None else None,
			"output_decision": output_decision,
			"execution_time_ms": round(execution_time_ms, 1),
			"error": error,
			"model_version": "models:/LightGBM/Production",
			"threshold": 0.4
		}
		
		# EXPLICATION : Écrit DIRECTEMENT dans le fichier (robuste à Gradio)
		# Mode "a" = append, newline assuré après chaque log
		log_line = json.dumps(log_entry, ensure_ascii=False) + "\n"
		log_file = _log_dir / "predictions.jsonl"
		
		with open(log_file, "a", encoding="utf-8") as f:
			f.write(log_line)
			f.flush()  # Force l'écriture immédiate (important pour le suivi en temps réel)
		
		# EXPLICATION : Aussi afficher dans la console pour Docker/HF Spaces
		print(f"[LOG] {log_line.strip()}")
		
	except Exception as exc:
		# EXPLICATION : N'échoue pas silencieusement si le logging échoue
		print(f"[ERROR] Logging échoué : {exc}", flush=True)


def _predict(json_line: str, threshold: float = 0.4) -> str:
	"""Predict default probability and return a formatted response."""
	# EXPLICATION : Capture du temps de début pour calculer execution_time_ms
	start_time = time.perf_counter()
	
	try:
		df = _parse_json_line(json_line)

		model = _load_model()

		# Align input columns to the model's expected features.
		# EXPLICATION: fill_value=np.nan (pas 0) pour que LightGBM utilise ses
		# splits natifs sur les valeurs manquantes — c'est ainsi qu'il a été entraîné.
		expected = _get_model_feature_names(model)
		if expected:
			df = df.reindex(columns=expected, fill_value=np.nan)

		# Final safety: ensure every column is numeric (LightGBM requirement)
		# NaN are preserved — LightGBM handles them natively.
		for col in df.columns:
			df[col] = pd.to_numeric(df[col], errors='coerce')

		try:
			proba = float(model.predict_proba(df)[:, 1][0])
		except AttributeError:
			# Fallback for models exposing predict() returning probabilities.
			proba = float(model.predict(df)[0])

		if not 0.0 <= proba <= 1.0:
			raise ValueError("La probabilité prédite est hors de l'intervalle [0, 1].")

		score = int(proba * 1000)
		decision = "Accordé" if proba < threshold else "Refusé"

		# EXPLICATION : Log de la prédiction réussie avec toutes les métriques
		execution_time_ms = (time.perf_counter() - start_time) * 1000
		input_features = json.loads(json_line)  # reconvertir pour le log
		log_prediction(
			input_raw=json_line,
			input_features=input_features,
			output_proba=proba,
			output_decision=decision,
			execution_time_ms=execution_time_ms,
			error=None
		)

		return (
			f"Score: {score}\n"
			f"Probabilité de défaut: {proba:.4f}\n"
			f"Décision: {decision}"
		)

	except ValueError as exc:
		# EXPLICATION : Log de l'erreur avec temps d'exécution et message d'erreur
		execution_time_ms = (time.perf_counter() - start_time) * 1000
		try:
			input_features = json.loads(json_line)
		except:
			input_features = {}
		log_prediction(
			input_raw=json_line,
			input_features=input_features,
			output_proba=None,
			output_decision="Erreur",
			execution_time_ms=execution_time_ms,
			error=f"ValueError: {exc}"
		)
		return f"Erreur: {exc}"
	except KeyError as exc:
		execution_time_ms = (time.perf_counter() - start_time) * 1000
		try:
			input_features = json.loads(json_line)
		except:
			input_features = {}
		log_prediction(
			input_raw=json_line,
			input_features=input_features,
			output_proba=None,
			output_decision="Erreur",
			execution_time_ms=execution_time_ms,
			error=f"KeyError: {exc}"
		)
		return f"Erreur: colonne manquante ({exc})."
	except TypeError as exc:
		execution_time_ms = (time.perf_counter() - start_time) * 1000
		try:
			input_features = json.loads(json_line)
		except:
			input_features = {}
		log_prediction(
			input_raw=json_line,
			input_features=input_features,
			output_proba=None,
			output_decision="Erreur",
			execution_time_ms=execution_time_ms,
			error=f"TypeError: {exc}"
		)
		return f"Erreur: type invalide ({exc})."
	except Exception as exc:  # noqa: BLE001
		execution_time_ms = (time.perf_counter() - start_time) * 1000
		try:
			input_features = json.loads(json_line)
		except:
			input_features = {}
		log_prediction(
			input_raw=json_line,
			input_features=input_features,
			output_proba=None,
			output_decision="Erreur",
			execution_time_ms=execution_time_ms,
			error=f"Exception: {exc}"
		)
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
				placeholder='{"SK_ID_CURR":100002,"CODE_GENDER":0,"FLAG_OWN_CAR":0,"FLAG_OWN_REALTY":0,"CNT_CHILDREN":0,"AMT_INCOME_TOTAL":202500,"AMT_CREDIT":406597.5,"AMT_ANNUITY":24700.5,"AMT_GOODS_PRICE":351000,"REGION_POPULATION_RELATIVE":0.018801,"DAYS_BIRTH":-9461,"DAYS_EMPLOYED":-637,"DAYS_REGISTRATION":-3648,"DAYS_ID_PUBLISH":-2120,"OWN_CAR_AGE":"","FLAG_MOBIL":1,"FLAG_EMP_PHONE":1,"FLAG_WORK_PHONE":0,"FLAG_CONT_MOBILE":1,"FLAG_PHONE":1,"FLAG_EMAIL":0,"CNT_FAM_MEMBERS":1,"REGION_RATING_CLIENT":2,"REGION_RATING_CLIENT_W_CITY":2,"HOUR_APPR_PROCESS_START":10,"REG_REGION_NOT_LIVE_REGION":0,"REG_REGION_NOT_WORK_REGION":0,"LIVE_REGION_NOT_WORK_REGION":0,"REG_CITY_NOT_LIVE_CITY":0,"REG_CITY_NOT_WORK_CITY":0,"LIVE_CITY_NOT_WORK_CITY":0,"EXT_SOURCE_1":0.0830369673913225,"EXT_SOURCE_2":0.2629485927471776,"EXT_SOURCE_3":0.1393757800997895,"APARTMENTS_AVG":0.0247,"BASEMENTAREA_AVG":0.0369,"YEARS_BEGINEXPLUATATION_AVG":0.9722,"YEARS_BUILD_AVG":0.6192,"COMMONAREA_AVG":0.0143,"ELEVATORS_AVG":0,"ENTRANCES_AVG":0.069,"FLOORSMAX_AVG":0.0833,"FLOORSMIN_AVG":0.125,"LANDAREA_AVG":0.0369,"LIVINGAPARTMENTS_AVG":0.0202,"LIVINGAREA_AVG":0.019,"NONLIVINGAPARTMENTS_AVG":0,"NONLIVINGAREA_AVG":0,"APARTMENTS_MODE":0.0252,"BASEMENTAREA_MODE":0.0383,"YEARS_BEGINEXPLUATATION_MODE":0.9722,"YEARS_BUILD_MODE":0.6341,"COMMONAREA_MODE":0.0144,"ELEVATORS_MODE":0,"ENTRANCES_MODE":0.069,"FLOORSMAX_MODE":0.0833,"FLOORSMIN_MODE":0.125,"LANDAREA_MODE":0.0377,"LIVINGAPARTMENTS_MODE":0.022,"LIVINGAREA_MODE":0.0198,"NONLIVINGAPARTMENTS_MODE":0,"NONLIVINGAREA_MODE":0,"APARTMENTS_MEDI":0.025,"BASEMENTAREA_MEDI":0.0369,"YEARS_BEGINEXPLUATATION_MEDI":0.9722,"YEARS_BUILD_MEDI":0.6243,"COMMONAREA_MEDI":0.0144,"ELEVATORS_MEDI":0,"ENTRANCES_MEDI":0.069,"FLOORSMAX_MEDI":0.0833,"FLOORSMIN_MEDI":0.125,"LANDAREA_MEDI":0.0375,"LIVINGAPARTMENTS_MEDI":0.0205,"LIVINGAREA_MEDI":0.0193,"NONLIVINGAPARTMENTS_MEDI":0,"NONLIVINGAREA_MEDI":0,"TOTALAREA_MODE":0.0149,"OBS_30_CNT_SOCIAL_CIRCLE":2,"DEF_30_CNT_SOCIAL_CIRCLE":2,"OBS_60_CNT_SOCIAL_CIRCLE":2,"DEF_60_CNT_SOCIAL_CIRCLE":2,"DAYS_LAST_PHONE_CHANGE":-1134,"FLAG_DOCUMENT_2":0,"FLAG_DOCUMENT_3":1,"FLAG_DOCUMENT_4":0,"FLAG_DOCUMENT_5":0,"FLAG_DOCUMENT_6":0,"FLAG_DOCUMENT_7":0,"FLAG_DOCUMENT_8":0,"FLAG_DOCUMENT_9":0,"FLAG_DOCUMENT_10":0,"FLAG_DOCUMENT_11":0,"FLAG_DOCUMENT_12":0,"FLAG_DOCUMENT_13":0,"FLAG_DOCUMENT_14":0,"FLAG_DOCUMENT_15":0,"FLAG_DOCUMENT_16":0,"FLAG_DOCUMENT_17":0,"FLAG_DOCUMENT_18":0,"FLAG_DOCUMENT_19":0,"FLAG_DOCUMENT_20":0,"FLAG_DOCUMENT_21":0,"AMT_REQ_CREDIT_BUREAU_HOUR":0,"AMT_REQ_CREDIT_BUREAU_DAY":0,"AMT_REQ_CREDIT_BUREAU_WEEK":0,"AMT_REQ_CREDIT_BUREAU_MON":0,"AMT_REQ_CREDIT_BUREAU_QRT":0,"AMT_REQ_CREDIT_BUREAU_YEAR":1,"NAME_CONTRACT_TYPE_Cash loans":"True","NAME_CONTRACT_TYPE_Revolving loans":"False","NAME_TYPE_SUITE_Children":"False","NAME_TYPE_SUITE_Family":"False","NAME_TYPE_SUITE_Group of people":"False","NAME_TYPE_SUITE_Other_A":"False","NAME_TYPE_SUITE_Other_B":"False","NAME_TYPE_SUITE_Spouse, partner":"False","NAME_TYPE_SUITE_Unaccompanied":"True","NAME_INCOME_TYPE_Businessman":"False","NAME_INCOME_TYPE_Commercial associate":"False","NAME_INCOME_TYPE_Pensioner":"False","NAME_INCOME_TYPE_State servant":"False","NAME_INCOME_TYPE_Student":"False","NAME_INCOME_TYPE_Unemployed":"False","NAME_INCOME_TYPE_Working":"True","NAME_EDUCATION_TYPE_Academic degree":"False","NAME_EDUCATION_TYPE_Higher education":"False","NAME_EDUCATION_TYPE_Incomplete higher":"False","NAME_EDUCATION_TYPE_Lower secondary":"False","NAME_EDUCATION_TYPE_Secondary / secondary special":"True","NAME_FAMILY_STATUS_Civil marriage":"False","NAME_FAMILY_STATUS_Married":"False","NAME_FAMILY_STATUS_Separated":"False","NAME_FAMILY_STATUS_Single / not married":"True","NAME_FAMILY_STATUS_Widow":"False","NAME_HOUSING_TYPE_Co-op apartment":"False","NAME_HOUSING_TYPE_House / apartment":"True","NAME_HOUSING_TYPE_Municipal apartment":"False","NAME_HOUSING_TYPE_Office apartment":"False","NAME_HOUSING_TYPE_Rented apartment":"False","NAME_HOUSING_TYPE_With parents":"False","OCCUPATION_TYPE_Accountants":"False","OCCUPATION_TYPE_Cleaning staff":"False","OCCUPATION_TYPE_Cooking staff":"False","OCCUPATION_TYPE_Core staff":"False","OCCUPATION_TYPE_Drivers":"False","OCCUPATION_TYPE_HR staff":"False","OCCUPATION_TYPE_High skill tech staff":"False","OCCUPATION_TYPE_IT staff":"False","OCCUPATION_TYPE_Laborers":"True","OCCUPATION_TYPE_Low-skill Laborers":"False","OCCUPATION_TYPE_Managers":"False","OCCUPATION_TYPE_Medicine staff":"False","OCCUPATION_TYPE_Private service staff":"False","OCCUPATION_TYPE_Realty agents":"False","OCCUPATION_TYPE_Sales staff":"False","OCCUPATION_TYPE_Secretaries":"False","OCCUPATION_TYPE_Security staff":"False","OCCUPATION_TYPE_Waiters/barmen staff":"False","WEEKDAY_APPR_PROCESS_START_FRIDAY":"False","WEEKDAY_APPR_PROCESS_START_MONDAY":"False","WEEKDAY_APPR_PROCESS_START_SATURDAY":"False","WEEKDAY_APPR_PROCESS_START_SUNDAY":"False","WEEKDAY_APPR_PROCESS_START_THURSDAY":"False","WEEKDAY_APPR_PROCESS_START_TUESDAY":"False","WEEKDAY_APPR_PROCESS_START_WEDNESDAY":"True","ORGANIZATION_TYPE_Advertising":"False","ORGANIZATION_TYPE_Agriculture":"False","ORGANIZATION_TYPE_Bank":"False","ORGANIZATION_TYPE_Business Entity Type 1":"False","ORGANIZATION_TYPE_Business Entity Type 2":"False","ORGANIZATION_TYPE_Business Entity Type 3":"True","ORGANIZATION_TYPE_Cleaning":"False","ORGANIZATION_TYPE_Construction":"False","ORGANIZATION_TYPE_Culture":"False","ORGANIZATION_TYPE_Electricity":"False","ORGANIZATION_TYPE_Emergency":"False","ORGANIZATION_TYPE_Government":"False","ORGANIZATION_TYPE_Hotel":"False","ORGANIZATION_TYPE_Housing":"False","ORGANIZATION_TYPE_Industry: type 1":"False","ORGANIZATION_TYPE_Industry: type 10":"False","ORGANIZATION_TYPE_Industry: type 11":"False","ORGANIZATION_TYPE_Industry: type 12":"False","ORGANIZATION_TYPE_Industry: type 13":"False","ORGANIZATION_TYPE_Industry: type 2":"False","ORGANIZATION_TYPE_Industry: type 3":"False","ORGANIZATION_TYPE_Industry: type 4":"False","ORGANIZATION_TYPE_Industry: type 5":"False","ORGANIZATION_TYPE_Industry: type 6":"False","ORGANIZATION_TYPE_Industry: type 7":"False","ORGANIZATION_TYPE_Industry: type 8":"False","ORGANIZATION_TYPE_Industry: type 9":"False","ORGANIZATION_TYPE_Insurance":"False","ORGANIZATION_TYPE_Kindergarten":"False","ORGANIZATION_TYPE_Legal Services":"False","ORGANIZATION_TYPE_Medicine":"False","ORGANIZATION_TYPE_Military":"False","ORGANIZATION_TYPE_Mobile":"False","ORGANIZATION_TYPE_Other":"False","ORGANIZATION_TYPE_Police":"False","ORGANIZATION_TYPE_Postal":"False","ORGANIZATION_TYPE_Realtor":"False","ORGANIZATION_TYPE_Religion":"False","ORGANIZATION_TYPE_Restaurant":"False","ORGANIZATION_TYPE_School":"False","ORGANIZATION_TYPE_Security":"False","ORGANIZATION_TYPE_Security Ministries":"False","ORGANIZATION_TYPE_Self-employed":"False","ORGANIZATION_TYPE_Services":"False","ORGANIZATION_TYPE_Telecom":"False","ORGANIZATION_TYPE_Trade: type 1":"False","ORGANIZATION_TYPE_Trade: type 2":"False","ORGANIZATION_TYPE_Trade: type 3":"False","ORGANIZATION_TYPE_Trade: type 4":"False","ORGANIZATION_TYPE_Trade: type 5":"False","ORGANIZATION_TYPE_Trade: type 6":"False","ORGANIZATION_TYPE_Trade: type 7":"False","ORGANIZATION_TYPE_Transport: type 1":"False","ORGANIZATION_TYPE_Transport: type 2":"False","ORGANIZATION_TYPE_Transport: type 3":"False","ORGANIZATION_TYPE_Transport: type 4":"False","ORGANIZATION_TYPE_University":"False","ORGANIZATION_TYPE_XNA":"False","FONDKAPREMONT_MODE_not specified":"False","FONDKAPREMONT_MODE_org spec account":"False","FONDKAPREMONT_MODE_reg oper account":"True","FONDKAPREMONT_MODE_reg oper spec account":"False","HOUSETYPE_MODE_block of flats":"True","HOUSETYPE_MODE_specific housing":"False","HOUSETYPE_MODE_terraced house":"False","WALLSMATERIAL_MODE_Block":"False","WALLSMATERIAL_MODE_Mixed":"False","WALLSMATERIAL_MODE_Monolithic":"False","WALLSMATERIAL_MODE_Others":"False","WALLSMATERIAL_MODE_Panel":"False","WALLSMATERIAL_MODE_Stone, brick":"True","WALLSMATERIAL_MODE_Wooden":"False","EMERGENCYSTATE_MODE_No":"True","EMERGENCYSTATE_MODE_Yes":"False","DAYS_EMPLOYED_PERC":0.0673290349857309,"INCOME_CREDIT_PERC":0.4980355265342261,"INCOME_PER_PERSON":202500,"ANNUITY_INCOME_PERC":0.12197777777777778,"PAYMENT_RATE":0.06074926678103038,"BURO_DAYS_CREDIT_MIN":"","BURO_DAYS_CREDIT_MAX":"","BURO_DAYS_CREDIT_MEAN":"","BURO_DAYS_CREDIT_VAR":"","BURO_DAYS_CREDIT_ENDDATE_MIN":"","BURO_DAYS_CREDIT_ENDDATE_MAX":"","BURO_DAYS_CREDIT_ENDDATE_MEAN":"","BURO_DAYS_CREDIT_UPDATE_MEAN":"","BURO_CREDIT_DAY_OVERDUE_MAX":"","BURO_CREDIT_DAY_OVERDUE_MEAN":"","BURO_AMT_CREDIT_MAX_OVERDUE_MEAN":"","BURO_AMT_CREDIT_SUM_MAX":"","BURO_AMT_CREDIT_SUM_MEAN":"","BURO_AMT_CREDIT_SUM_SUM":"","BURO_AMT_CREDIT_SUM_DEBT_MAX":"","BURO_AMT_CREDIT_SUM_DEBT_MEAN":"","BURO_AMT_CREDIT_SUM_DEBT_SUM":"","BURO_AMT_CREDIT_SUM_OVERDUE_MEAN":"","BURO_AMT_CREDIT_SUM_LIMIT_MEAN":"","BURO_AMT_CREDIT_SUM_LIMIT_SUM":"","BURO_AMT_ANNUITY_MAX":"","BURO_AMT_ANNUITY_MEAN":"","BURO_CNT_CREDIT_PROLONG_SUM":"","BURO_MONTHS_BALANCE_MIN_MIN":"","BURO_MONTHS_BALANCE_MAX_MAX":"","BURO_MONTHS_BALANCE_SIZE_MEAN":"","BURO_MONTHS_BALANCE_SIZE_SUM":"","BURO_CREDIT_ACTIVE_Active_MEAN":"","BURO_CREDIT_ACTIVE_Bad debt_MEAN":"","BURO_CREDIT_ACTIVE_Closed_MEAN":"","BURO_CREDIT_ACTIVE_Sold_MEAN":"","BURO_CREDIT_ACTIVE_nan_MEAN":"","BURO_CREDIT_CURRENCY_currency 1_MEAN":"","BURO_CREDIT_CURRENCY_currency 2_MEAN":"","BURO_CREDIT_CURRENCY_currency 4_MEAN":"","BURO_CREDIT_CURRENCY_nan_MEAN":"","BURO_CREDIT_TYPE_Another type of loan_MEAN":"","BURO_CREDIT_TYPE_Car loan_MEAN":"","BURO_CREDIT_TYPE_Consumer credit_MEAN":"","BURO_CREDIT_TYPE_Credit card_MEAN":"","BURO_CREDIT_TYPE_Loan for business development_MEAN":"","BURO_CREDIT_TYPE_Loan for working capital replenishment_MEAN":"","BURO_CREDIT_TYPE_Microloan_MEAN":"","BURO_CREDIT_TYPE_Mortgage_MEAN":"","BURO_CREDIT_TYPE_Real estate loan_MEAN":"","BURO_CREDIT_TYPE_Unknown type of loan_MEAN":"","BURO_CREDIT_TYPE_nan_MEAN":"","BURO_STATUS_0_MEAN_MEAN":"","BURO_STATUS_1_MEAN_MEAN":"","BURO_STATUS_2_MEAN_MEAN":"","BURO_STATUS_3_MEAN_MEAN":"","BURO_STATUS_C_MEAN_MEAN":"","BURO_STATUS_X_MEAN_MEAN":"","BURO_STATUS_nan_MEAN_MEAN":"","ACTIVE_DAYS_CREDIT_MIN":"","ACTIVE_DAYS_CREDIT_MAX":"","ACTIVE_DAYS_CREDIT_MEAN":"","ACTIVE_DAYS_CREDIT_VAR":"","ACTIVE_DAYS_CREDIT_ENDDATE_MIN":"","ACTIVE_DAYS_CREDIT_ENDDATE_MAX":"","ACTIVE_DAYS_CREDIT_ENDDATE_MEAN":"","ACTIVE_DAYS_CREDIT_UPDATE_MEAN":"","ACTIVE_CREDIT_DAY_OVERDUE_MAX":"","ACTIVE_CREDIT_DAY_OVERDUE_MEAN":"","ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN":"","ACTIVE_AMT_CREDIT_SUM_MAX":"","ACTIVE_AMT_CREDIT_SUM_MEAN":"","ACTIVE_AMT_CREDIT_SUM_SUM":"","ACTIVE_AMT_CREDIT_SUM_DEBT_MAX":"","ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN":"","ACTIVE_AMT_CREDIT_SUM_DEBT_SUM":"","ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN":"","ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN":"","ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM":"","ACTIVE_AMT_ANNUITY_MAX":"","ACTIVE_AMT_ANNUITY_MEAN":"","ACTIVE_CNT_CREDIT_PROLONG_SUM":"","ACTIVE_MONTHS_BALANCE_MIN_MIN":"","ACTIVE_MONTHS_BALANCE_MAX_MAX":"","ACTIVE_MONTHS_BALANCE_SIZE_MEAN":"","ACTIVE_MONTHS_BALANCE_SIZE_SUM":"","CLOSED_DAYS_CREDIT_MIN":"","CLOSED_DAYS_CREDIT_MAX":"","CLOSED_DAYS_CREDIT_MEAN":"","CLOSED_DAYS_CREDIT_VAR":"","CLOSED_DAYS_CREDIT_ENDDATE_MIN":"","CLOSED_DAYS_CREDIT_ENDDATE_MAX":"","CLOSED_DAYS_CREDIT_ENDDATE_MEAN":"","CLOSED_CREDIT_DAY_OVERDUE_MAX":"","CLOSED_CREDIT_DAY_OVERDUE_MEAN":"","CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN":"","CLOSED_AMT_CREDIT_SUM_MAX":"","CLOSED_AMT_CREDIT_SUM_MEAN":"","CLOSED_AMT_CREDIT_SUM_SUM":"","CLOSED_AMT_CREDIT_SUM_DEBT_MAX":"","CLOSED_AMT_CREDIT_SUM_DEBT_MEAN":"","CLOSED_AMT_CREDIT_SUM_DEBT_SUM":"","CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN":"","CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN":"","CLOSED_AMT_CREDIT_SUM_LIMIT_SUM":"","CLOSED_AMT_ANNUITY_MAX":"","CLOSED_AMT_ANNUITY_MEAN":"","CLOSED_CNT_CREDIT_PROLONG_SUM":"","CLOSED_MONTHS_BALANCE_MIN_MIN":"","CLOSED_MONTHS_BALANCE_MAX_MAX":"","CLOSED_MONTHS_BALANCE_SIZE_MEAN":"","CLOSED_MONTHS_BALANCE_SIZE_SUM":"","PREV_AMT_ANNUITY_MIN":"","PREV_AMT_ANNUITY_MAX":"","PREV_AMT_ANNUITY_MEAN":"","PREV_AMT_APPLICATION_MIN":"","PREV_AMT_APPLICATION_MAX":"","PREV_AMT_APPLICATION_MEAN":"","PREV_AMT_CREDIT_MIN":"","PREV_AMT_CREDIT_MAX":"","PREV_AMT_CREDIT_MEAN":"","PREV_APP_CREDIT_PERC_MIN":"","PREV_APP_CREDIT_PERC_MAX":"","PREV_APP_CREDIT_PERC_MEAN":"","PREV_APP_CREDIT_PERC_VAR":"","PREV_AMT_DOWN_PAYMENT_MIN":"","PREV_AMT_DOWN_PAYMENT_MAX":"","PREV_AMT_DOWN_PAYMENT_MEAN":"","PREV_AMT_GOODS_PRICE_MIN":"","PREV_AMT_GOODS_PRICE_MAX":"","PREV_AMT_GOODS_PRICE_MEAN":"","PREV_HOUR_APPR_PROCESS_START_MIN":"","PREV_HOUR_APPR_PROCESS_START_MAX":"","PREV_HOUR_APPR_PROCESS_START_MEAN":"","PREV_RATE_DOWN_PAYMENT_MIN":"","PREV_RATE_DOWN_PAYMENT_MAX":"","PREV_RATE_DOWN_PAYMENT_MEAN":"","PREV_DAYS_DECISION_MIN":"","PREV_DAYS_DECISION_MAX":"","PREV_DAYS_DECISION_MEAN":"","PREV_CNT_PAYMENT_MEAN":"","PREV_CNT_PAYMENT_SUM":"","PREV_NAME_CONTRACT_TYPE_Cash loans_MEAN":"","PREV_NAME_CONTRACT_TYPE_Consumer loans_MEAN":"","PREV_NAME_CONTRACT_TYPE_Revolving loans_MEAN":"","PREV_NAME_CONTRACT_TYPE_XNA_MEAN":"","PREV_NAME_CONTRACT_TYPE_nan_MEAN":"","PREV_WEEKDAY_APPR_PROCESS_START_FRIDAY_MEAN":"","PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN":"","PREV_WEEKDAY_APPR_PROCESS_START_SATURDAY_MEAN":"","PREV_WEEKDAY_APPR_PROCESS_START_SUNDAY_MEAN":"","PREV_WEEKDAY_APPR_PROCESS_START_THURSDAY_MEAN":"","PREV_WEEKDAY_APPR_PROCESS_START_TUESDAY_MEAN":"","PREV_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_MEAN":"","PREV_WEEKDAY_APPR_PROCESS_START_nan_MEAN":"","PREV_FLAG_LAST_APPL_PER_CONTRACT_N_MEAN":"","PREV_FLAG_LAST_APPL_PER_CONTRACT_Y_MEAN":"","PREV_FLAG_LAST_APPL_PER_CONTRACT_nan_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Business development_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Buying a garage_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Buying a home_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Buying a new car_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Buying a used car_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Car repairs_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Education_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Everyday expenses_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Furniture_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Gasification / water supply_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Hobby_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Journey_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Medicine_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Other_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Payments on other loans_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Repairs_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Urgent needs_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_XAP_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_XNA_MEAN":"","PREV_NAME_CASH_LOAN_PURPOSE_nan_MEAN":"","PREV_NAME_CONTRACT_STATUS_Approved_MEAN":"","PREV_NAME_CONTRACT_STATUS_Canceled_MEAN":"","PREV_NAME_CONTRACT_STATUS_Refused_MEAN":"","PREV_NAME_CONTRACT_STATUS_Unused offer_MEAN":"","PREV_NAME_CONTRACT_STATUS_nan_MEAN":"","PREV_NAME_PAYMENT_TYPE_Cash through the bank_MEAN":"","PREV_NAME_PAYMENT_TYPE_Cashless from the account of the employer_MEAN":"","PREV_NAME_PAYMENT_TYPE_Non-cash from your account_MEAN":"","PREV_NAME_PAYMENT_TYPE_XNA_MEAN":"","PREV_NAME_PAYMENT_TYPE_nan_MEAN":"","PREV_CODE_REJECT_REASON_CLIENT_MEAN":"","PREV_CODE_REJECT_REASON_HC_MEAN":"","PREV_CODE_REJECT_REASON_LIMIT_MEAN":"","PREV_CODE_REJECT_REASON_SCO_MEAN":"","PREV_CODE_REJECT_REASON_SCOFR_MEAN":"","PREV_CODE_REJECT_REASON_VERIF_MEAN":"","PREV_CODE_REJECT_REASON_XAP_MEAN":"","PREV_CODE_REJECT_REASON_XNA_MEAN":"","PREV_CODE_REJECT_REASON_nan_MEAN":"","PREV_NAME_TYPE_SUITE_Children_MEAN":"","PREV_NAME_TYPE_SUITE_Family_MEAN":"","PREV_NAME_TYPE_SUITE_Group of people_MEAN":"","PREV_NAME_TYPE_SUITE_Other_A_MEAN":"","PREV_NAME_TYPE_SUITE_Other_B_MEAN":"","PREV_NAME_TYPE_SUITE_Spouse, partner_MEAN":"","PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN":"","PREV_NAME_TYPE_SUITE_nan_MEAN":"","PREV_NAME_CLIENT_TYPE_New_MEAN":"","PREV_NAME_CLIENT_TYPE_Refreshed_MEAN":"","PREV_NAME_CLIENT_TYPE_Repeater_MEAN":"","PREV_NAME_CLIENT_TYPE_XNA_MEAN":"","PREV_NAME_CLIENT_TYPE_nan_MEAN":"","PREV_NAME_GOODS_CATEGORY_Audio/Video_MEAN":"","PREV_NAME_GOODS_CATEGORY_Auto Accessories_MEAN":"","PREV_NAME_GOODS_CATEGORY_Clothing and Accessories_MEAN":"","PREV_NAME_GOODS_CATEGORY_Computers_MEAN":"","PREV_NAME_GOODS_CATEGORY_Construction Materials_MEAN":"","PREV_NAME_GOODS_CATEGORY_Consumer Electronics_MEAN":"","PREV_NAME_GOODS_CATEGORY_Direct Sales_MEAN":"","PREV_NAME_GOODS_CATEGORY_Education_MEAN":"","PREV_NAME_GOODS_CATEGORY_Fitness_MEAN":"","PREV_NAME_GOODS_CATEGORY_Furniture_MEAN":"","PREV_NAME_GOODS_CATEGORY_Gardening_MEAN":"","PREV_NAME_GOODS_CATEGORY_Homewares_MEAN":"","PREV_NAME_GOODS_CATEGORY_Insurance_MEAN":"","PREV_NAME_GOODS_CATEGORY_Jewelry_MEAN":"","PREV_NAME_GOODS_CATEGORY_Medical Supplies_MEAN":"","PREV_NAME_GOODS_CATEGORY_Medicine_MEAN":"","PREV_NAME_GOODS_CATEGORY_Mobile_MEAN":"","PREV_NAME_GOODS_CATEGORY_Office Appliances_MEAN":"","PREV_NAME_GOODS_CATEGORY_Other_MEAN":"","PREV_NAME_GOODS_CATEGORY_Photo / Cinema Equipment_MEAN":"","PREV_NAME_GOODS_CATEGORY_Sport and Leisure_MEAN":"","PREV_NAME_GOODS_CATEGORY_Tourism_MEAN":"","PREV_NAME_GOODS_CATEGORY_Vehicles_MEAN":"","PREV_NAME_GOODS_CATEGORY_XNA_MEAN":"","PREV_NAME_GOODS_CATEGORY_nan_MEAN":"","PREV_NAME_PORTFOLIO_Cards_MEAN":"","PREV_NAME_PORTFOLIO_Cars_MEAN":"","PREV_NAME_PORTFOLIO_Cash_MEAN":"","PREV_NAME_PORTFOLIO_POS_MEAN":"","PREV_NAME_PORTFOLIO_XNA_MEAN":"","PREV_NAME_PORTFOLIO_nan_MEAN":"","PREV_NAME_PRODUCT_TYPE_XNA_MEAN":"","PREV_NAME_PRODUCT_TYPE_walk-in_MEAN":"","PREV_NAME_PRODUCT_TYPE_x-sell_MEAN":"","PREV_NAME_PRODUCT_TYPE_nan_MEAN":"","PREV_CHANNEL_TYPE_AP+ (Cash loan)_MEAN":"","PREV_CHANNEL_TYPE_Car dealer_MEAN":"","PREV_CHANNEL_TYPE_Channel of corporate sales_MEAN":"","PREV_CHANNEL_TYPE_Contact center_MEAN":"","PREV_CHANNEL_TYPE_Country-wide_MEAN":"","PREV_CHANNEL_TYPE_Credit and cash offices_MEAN":"","PREV_CHANNEL_TYPE_Regional / Local_MEAN":"","PREV_CHANNEL_TYPE_Stone_MEAN":"","PREV_CHANNEL_TYPE_nan_MEAN":"","PREV_NAME_SELLER_INDUSTRY_Auto technology_MEAN":"","PREV_NAME_SELLER_INDUSTRY_Clothing_MEAN":"","PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN":"","PREV_NAME_SELLER_INDUSTRY_Construction_MEAN":"","PREV_NAME_SELLER_INDUSTRY_Consumer electronics_MEAN":"","PREV_NAME_SELLER_INDUSTRY_Furniture_MEAN":"","PREV_NAME_SELLER_INDUSTRY_Industry_MEAN":"","PREV_NAME_SELLER_INDUSTRY_Jewelry_MEAN":"","PREV_NAME_SELLER_INDUSTRY_MLM partners_MEAN":"","PREV_NAME_SELLER_INDUSTRY_Tourism_MEAN":"","PREV_NAME_SELLER_INDUSTRY_XNA_MEAN":"","PREV_NAME_SELLER_INDUSTRY_nan_MEAN":"","PREV_NAME_YIELD_GROUP_XNA_MEAN":"","PREV_NAME_YIELD_GROUP_high_MEAN":"","PREV_NAME_YIELD_GROUP_low_action_MEAN":"","PREV_NAME_YIELD_GROUP_low_normal_MEAN":"","PREV_NAME_YIELD_GROUP_middle_MEAN":"","PREV_NAME_YIELD_GROUP_nan_MEAN":"","PREV_PRODUCT_COMBINATION_Card Street_MEAN":"","PREV_PRODUCT_COMBINATION_Card X-Sell_MEAN":"","PREV_PRODUCT_COMBINATION_Cash_MEAN":"","PREV_PRODUCT_COMBINATION_Cash Street: high_MEAN":"","PREV_PRODUCT_COMBINATION_Cash Street: low_MEAN":"","PREV_PRODUCT_COMBINATION_Cash Street: middle_MEAN":"","PREV_PRODUCT_COMBINATION_Cash X-Sell: high_MEAN":"","PREV_PRODUCT_COMBINATION_Cash X-Sell: low_MEAN":"","PREV_PRODUCT_COMBINATION_Cash X-Sell: middle_MEAN":"","PREV_PRODUCT_COMBINATION_POS household with interest_MEAN":"","PREV_PRODUCT_COMBINATION_POS household without interest_MEAN":"","PREV_PRODUCT_COMBINATION_POS industry with interest_MEAN":"","PREV_PRODUCT_COMBINATION_POS industry without interest_MEAN":"","PREV_PRODUCT_COMBINATION_POS mobile with interest_MEAN":"","PREV_PRODUCT_COMBINATION_POS mobile without interest_MEAN":"","PREV_PRODUCT_COMBINATION_POS other with interest_MEAN":"","PREV_PRODUCT_COMBINATION_POS others without interest_MEAN":"","PREV_PRODUCT_COMBINATION_nan_MEAN":"","APPROVED_AMT_ANNUITY_MIN":"","APPROVED_AMT_ANNUITY_MAX":"","APPROVED_AMT_ANNUITY_MEAN":"","APPROVED_AMT_APPLICATION_MIN":"","APPROVED_AMT_APPLICATION_MAX":"","APPROVED_AMT_APPLICATION_MEAN":"","APPROVED_AMT_CREDIT_MIN":"","APPROVED_AMT_CREDIT_MAX":"","APPROVED_AMT_CREDIT_MEAN":"","APPROVED_APP_CREDIT_PERC_MIN":"","APPROVED_APP_CREDIT_PERC_MAX":"","APPROVED_APP_CREDIT_PERC_MEAN":"","APPROVED_APP_CREDIT_PERC_VAR":"","APPROVED_AMT_DOWN_PAYMENT_MIN":"","APPROVED_AMT_DOWN_PAYMENT_MAX":"","APPROVED_AMT_DOWN_PAYMENT_MEAN":"","APPROVED_AMT_GOODS_PRICE_MIN":"","APPROVED_AMT_GOODS_PRICE_MAX":"","APPROVED_AMT_GOODS_PRICE_MEAN":"","APPROVED_HOUR_APPR_PROCESS_START_MIN":"","APPROVED_HOUR_APPR_PROCESS_START_MAX":"","APPROVED_HOUR_APPR_PROCESS_START_MEAN":"","APPROVED_RATE_DOWN_PAYMENT_MIN":"","APPROVED_RATE_DOWN_PAYMENT_MAX":"","APPROVED_RATE_DOWN_PAYMENT_MEAN":"","APPROVED_DAYS_DECISION_MIN":"","APPROVED_DAYS_DECISION_MAX":"","APPROVED_DAYS_DECISION_MEAN":"","APPROVED_CNT_PAYMENT_MEAN":"","APPROVED_CNT_PAYMENT_SUM":"","REFUSED_AMT_ANNUITY_MIN":"","REFUSED_AMT_ANNUITY_MAX":"","REFUSED_AMT_ANNUITY_MEAN":"","REFUSED_AMT_APPLICATION_MIN":"","REFUSED_AMT_APPLICATION_MAX":"","REFUSED_AMT_APPLICATION_MEAN":"","REFUSED_AMT_CREDIT_MIN":"","REFUSED_AMT_CREDIT_MAX":"","REFUSED_AMT_CREDIT_MEAN":"","REFUSED_APP_CREDIT_PERC_MIN":"","REFUSED_APP_CREDIT_PERC_MAX":"","REFUSED_APP_CREDIT_PERC_MEAN":"","REFUSED_APP_CREDIT_PERC_VAR":"","REFUSED_AMT_DOWN_PAYMENT_MIN":"","REFUSED_AMT_DOWN_PAYMENT_MAX":"","REFUSED_AMT_DOWN_PAYMENT_MEAN":"","REFUSED_AMT_GOODS_PRICE_MIN":"","REFUSED_AMT_GOODS_PRICE_MAX":"","REFUSED_AMT_GOODS_PRICE_MEAN":"","REFUSED_HOUR_APPR_PROCESS_START_MIN":"","REFUSED_HOUR_APPR_PROCESS_START_MAX":"","REFUSED_HOUR_APPR_PROCESS_START_MEAN":"","REFUSED_RATE_DOWN_PAYMENT_MIN":"","REFUSED_RATE_DOWN_PAYMENT_MAX":"","REFUSED_RATE_DOWN_PAYMENT_MEAN":"","REFUSED_DAYS_DECISION_MIN":"","REFUSED_DAYS_DECISION_MAX":"","REFUSED_DAYS_DECISION_MEAN":"","REFUSED_CNT_PAYMENT_MEAN":"","REFUSED_CNT_PAYMENT_SUM":"","POS_MONTHS_BALANCE_MAX":"","POS_MONTHS_BALANCE_MEAN":"","POS_MONTHS_BALANCE_SIZE":"","POS_SK_DPD_MAX":"","POS_SK_DPD_MEAN":"","POS_SK_DPD_DEF_MAX":"","POS_SK_DPD_DEF_MEAN":"","POS_NAME_CONTRACT_STATUS_Active_MEAN":"","POS_NAME_CONTRACT_STATUS_Approved_MEAN":"","POS_NAME_CONTRACT_STATUS_Completed_MEAN":"","POS_NAME_CONTRACT_STATUS_Demand_MEAN":"","POS_NAME_CONTRACT_STATUS_Returned to the store_MEAN":"","POS_NAME_CONTRACT_STATUS_Signed_MEAN":"","POS_NAME_CONTRACT_STATUS_nan_MEAN":"","POS_COUNT":"","INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE":"","INSTAL_DPD_MAX":"","INSTAL_DPD_MEAN":"","INSTAL_DPD_SUM":"","INSTAL_DBD_MAX":"","INSTAL_DBD_MEAN":"","INSTAL_DBD_SUM":"","INSTAL_PAYMENT_PERC_MAX":"","INSTAL_PAYMENT_PERC_MEAN":"","INSTAL_PAYMENT_PERC_SUM":"","INSTAL_PAYMENT_PERC_VAR":"","INSTAL_PAYMENT_DIFF_MAX":"","INSTAL_PAYMENT_DIFF_MEAN":"","INSTAL_PAYMENT_DIFF_SUM":"","INSTAL_PAYMENT_DIFF_VAR":"","INSTAL_AMT_INSTALMENT_MAX":"","INSTAL_AMT_INSTALMENT_MEAN":"","INSTAL_AMT_INSTALMENT_SUM":"","INSTAL_AMT_PAYMENT_MIN":"","INSTAL_AMT_PAYMENT_MAX":"","INSTAL_AMT_PAYMENT_MEAN":"","INSTAL_AMT_PAYMENT_SUM":"","INSTAL_DAYS_ENTRY_PAYMENT_MAX":"","INSTAL_DAYS_ENTRY_PAYMENT_MEAN":"","INSTAL_DAYS_ENTRY_PAYMENT_SUM":"","INSTAL_COUNT":"","CC_MONTHS_BALANCE_MIN":"","CC_MONTHS_BALANCE_MAX":"","CC_MONTHS_BALANCE_MEAN":"","CC_MONTHS_BALANCE_SUM":"","CC_MONTHS_BALANCE_VAR":"","CC_AMT_BALANCE_MIN":"","CC_AMT_BALANCE_MAX":"","CC_AMT_BALANCE_MEAN":"","CC_AMT_BALANCE_SUM":"","CC_AMT_BALANCE_VAR":"","CC_AMT_CREDIT_LIMIT_ACTUAL_MIN":"","CC_AMT_CREDIT_LIMIT_ACTUAL_MAX":"","CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN":"","CC_AMT_CREDIT_LIMIT_ACTUAL_SUM":"","CC_AMT_CREDIT_LIMIT_ACTUAL_VAR":"","CC_AMT_DRAWINGS_ATM_CURRENT_MIN":"","CC_AMT_DRAWINGS_ATM_CURRENT_MAX":"","CC_AMT_DRAWINGS_ATM_CURRENT_MEAN":"","CC_AMT_DRAWINGS_ATM_CURRENT_SUM":"","CC_AMT_DRAWINGS_ATM_CURRENT_VAR":"","CC_AMT_DRAWINGS_CURRENT_MIN":"","CC_AMT_DRAWINGS_CURRENT_MAX":"","CC_AMT_DRAWINGS_CURRENT_MEAN":"","CC_AMT_DRAWINGS_CURRENT_SUM":"","CC_AMT_DRAWINGS_CURRENT_VAR":"","CC_AMT_DRAWINGS_OTHER_CURRENT_MIN":"","CC_AMT_DRAWINGS_OTHER_CURRENT_MAX":"","CC_AMT_DRAWINGS_OTHER_CURRENT_MEAN":"","CC_AMT_DRAWINGS_OTHER_CURRENT_SUM":"","CC_AMT_DRAWINGS_OTHER_CURRENT_VAR":"","CC_AMT_DRAWINGS_POS_CURRENT_MIN":"","CC_AMT_DRAWINGS_POS_CURRENT_MAX":"","CC_AMT_DRAWINGS_POS_CURRENT_MEAN":"","CC_AMT_DRAWINGS_POS_CURRENT_SUM":"","CC_AMT_DRAWINGS_POS_CURRENT_VAR":"","CC_AMT_INST_MIN_REGULARITY_MIN":"","CC_AMT_INST_MIN_REGULARITY_MAX":"","CC_AMT_INST_MIN_REGULARITY_MEAN":"","CC_AMT_INST_MIN_REGULARITY_SUM":"","CC_AMT_INST_MIN_REGULARITY_VAR":"","CC_AMT_PAYMENT_CURRENT_MIN":"","CC_AMT_PAYMENT_CURRENT_MAX":"","CC_AMT_PAYMENT_CURRENT_MEAN":"","CC_AMT_PAYMENT_CURRENT_SUM":"","CC_AMT_PAYMENT_CURRENT_VAR":"","CC_AMT_PAYMENT_TOTAL_CURRENT_MIN":"","CC_AMT_PAYMENT_TOTAL_CURRENT_MAX":"","CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN":"","CC_AMT_PAYMENT_TOTAL_CURRENT_SUM":"","CC_AMT_PAYMENT_TOTAL_CURRENT_VAR":"","CC_AMT_RECEIVABLE_PRINCIPAL_MIN":"","CC_AMT_RECEIVABLE_PRINCIPAL_MAX":"","CC_AMT_RECEIVABLE_PRINCIPAL_MEAN":"","CC_AMT_RECEIVABLE_PRINCIPAL_SUM":"","CC_AMT_RECEIVABLE_PRINCIPAL_VAR":"","CC_AMT_RECIVABLE_MIN":"","CC_AMT_RECIVABLE_MAX":"","CC_AMT_RECIVABLE_MEAN":"","CC_AMT_RECIVABLE_SUM":"","CC_AMT_RECIVABLE_VAR":"","CC_AMT_TOTAL_RECEIVABLE_MIN":"","CC_AMT_TOTAL_RECEIVABLE_MAX":"","CC_AMT_TOTAL_RECEIVABLE_MEAN":"","CC_AMT_TOTAL_RECEIVABLE_SUM":"","CC_AMT_TOTAL_RECEIVABLE_VAR":"","CC_CNT_DRAWINGS_ATM_CURRENT_MIN":"","CC_CNT_DRAWINGS_ATM_CURRENT_MAX":"","CC_CNT_DRAWINGS_ATM_CURRENT_MEAN":"","CC_CNT_DRAWINGS_ATM_CURRENT_SUM":"","CC_CNT_DRAWINGS_ATM_CURRENT_VAR":"","CC_CNT_DRAWINGS_CURRENT_MIN":"","CC_CNT_DRAWINGS_CURRENT_MAX":"","CC_CNT_DRAWINGS_CURRENT_MEAN":"","CC_CNT_DRAWINGS_CURRENT_SUM":"","CC_CNT_DRAWINGS_CURRENT_VAR":"","CC_CNT_DRAWINGS_OTHER_CURRENT_MIN":"","CC_CNT_DRAWINGS_OTHER_CURRENT_MAX":"","CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN":"","CC_CNT_DRAWINGS_OTHER_CURRENT_SUM":"","CC_CNT_DRAWINGS_OTHER_CURRENT_VAR":"","CC_CNT_DRAWINGS_POS_CURRENT_MIN":"","CC_CNT_DRAWINGS_POS_CURRENT_MAX":"","CC_CNT_DRAWINGS_POS_CURRENT_MEAN":"","CC_CNT_DRAWINGS_POS_CURRENT_SUM":"","CC_CNT_DRAWINGS_POS_CURRENT_VAR":"","CC_CNT_INSTALMENT_MATURE_CUM_MIN":"","CC_CNT_INSTALMENT_MATURE_CUM_MAX":"","CC_CNT_INSTALMENT_MATURE_CUM_MEAN":"","CC_CNT_INSTALMENT_MATURE_CUM_SUM":"","CC_CNT_INSTALMENT_MATURE_CUM_VAR":"","CC_SK_DPD_MIN":"","CC_SK_DPD_MAX":"","CC_SK_DPD_MEAN":"","CC_SK_DPD_SUM":"","CC_SK_DPD_VAR":"","CC_SK_DPD_DEF_MIN":"","CC_SK_DPD_DEF_MAX":"","CC_SK_DPD_DEF_MEAN":"","CC_SK_DPD_DEF_SUM":"","CC_SK_DPD_DEF_VAR":"","CC_STATUS_Active":"","CC_STATUS_Completed":"","CC_STATUS_Demand":"","CC_STATUS_Sent proposal":"","CC_STATUS_Signed":"","CC_COUNT":""}',
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
