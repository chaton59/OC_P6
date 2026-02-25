"""Preprocessor to convert "raw" input JSON into the model feature vector.

This transformer is purposely lightweight and deterministic:
- Reads the expected feature names from `data/processed/features_train.csv` when not
  provided explicitly.
- If an expected feature is present verbatim in the input it is used.
- If an expected feature looks like a one-hot column (e.g. "NAME_CONTRACT_TYPE_Cash loans")
  and the input contains the base column "NAME_CONTRACT_TYPE": "Cash loans", the
  corresponding one-hot column is set to 1, others to 0.
- Missing features are filled with `0`.

The goal is to allow the API to accept "raw" payloads (categorical strings, booleans)
and map them to the exact column names used at training time.

This transformer implements a minimal sklearn-like API (fit/transform) so it can be
pickled/joblib-dumped if desired.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


class RawToModelTransformer:
	"""Transformer that maps raw inputs to model feature vector expected names.

	This improved transformer:
	- infers expected feature names from the training CSV if not provided
	- computes a few derived features commonly used in the notebook (PAYMENT_RATE,
	  INCOME_CREDIT_PERC, INCOME_PER_PERSON, ANNUITY_INCOME_PERC, DAYS_EMPLOYED_PERC)
	- fills non-computable/unknown features with the column median from
	  `data/processed/features_train.csv` when available (better than 0)
	- maps raw categorical columns to one-hot expected columns by prefix match + sanitized
	  category names (robust to spaces/special chars)
	
	The transformer is intentionally conservative — it does not attempt to
	recreate complex aggregations (BURO_*, PREV_*, POS_*, CC_*, INSTAL_* etc.).
	"""
	@staticmethod
	def _sanitize_column_name(name: str) -> str:
		"""Sanitize a column name to match the model's feature naming convention.

		Replicates the notebook cleaning (03_LGBM.ipynb cell 6):
		  1. Replace spaces with '_'
		  2. Replace all non-alphanumeric/non-underscore chars with '_'
		Note: double underscores are NOT collapsed — the exported model
		feature names retain them.
		"""
		s = name.replace(' ', '_')
		s = re.sub(r'[^a-zA-Z0-9_]', '_', s)
		return s

	def __init__(self, expected_features: Optional[Iterable[str]] = None, fill_value: float = 0.0) -> None:
		self.fill_value = fill_value
		self.expected_features = list(expected_features) if expected_features is not None else self._read_features_from_csv()

		# Precompute imputation (median) for expected numeric features from train CSV
		self._impute_values: dict = {}
		train_path = Path("data/processed/features_train.csv")
		if train_path.exists():
			try:
				df_train = pd.read_csv(train_path, nrows=10000)
				# remove identifier/target if present
				for c in ("SK_ID_CURR", "TARGET"):
					if c in df_train.columns:
						df_train = df_train.drop(columns=[c])
				# Sanitize column names to match expected features
				df_train.columns = [self._sanitize_column_name(c) for c in df_train.columns]
				medians = df_train.median(numeric_only=True)
				for col in self.expected_features:
					if col in medians.index:
						self._impute_values[col] = float(medians.loc[col])
			except Exception:
				# ignore and keep empty imputation map
				self._impute_values = {}

	def _read_features_from_csv(self) -> List[str]:
		"""Read expected feature names from the training CSV header.

		Uses ``pd.read_csv(nrows=0)`` to correctly handle quoted column
		names that contain commas (e.g. 'Spouse, partner').
		Applies the same sanitization as the training notebook.
		"""
		p = Path("data/processed/features_train.csv")
		if not p.exists():
			return []
		try:
			df_header = pd.read_csv(p, nrows=0)
			cols = [c for c in df_header.columns if c not in ("SK_ID_CURR", "TARGET")]
			return [self._sanitize_column_name(c) for c in cols]
		except Exception:
			return []

	def fit(self, X=None, y=None):
		# Stateless transformer
		return self

	def _is_nan(self, x) -> bool:
		return pd.isna(x)

	def _sanitize_category(self, val: str) -> str:
		"""Normalize a category value to match the one-hot column suffix convention.

		Uses the same logic as ``_sanitize_column_name`` (no collapse of
		double underscores) so that e.g. 'Spouse, partner' → 'Spouse__partner'
		matches the model feature name ``NAME_TYPE_SUITE_Spouse__partner``.
		"""
		if pd.isna(val):
			return ""
		return self._sanitize_column_name(str(val).strip())

	def _compute_derived(self, row: pd.Series) -> dict:
		# Compute a few numeric derived features when base columns are available
		out = {}
		# PAYMENT_RATE = AMT_ANNUITY / AMT_CREDIT
		if 'AMT_ANNUITY' in row.index and 'AMT_CREDIT' in row.index:
			try:
				out['PAYMENT_RATE'] = float(row['AMT_ANNUITY']) / float(row['AMT_CREDIT']) if float(row['AMT_CREDIT']) != 0 else self.fill_value
			except Exception:
				out['PAYMENT_RATE'] = self.fill_value

		# INCOME_CREDIT_PERC = AMT_INCOME_TOTAL / AMT_CREDIT
		if 'AMT_INCOME_TOTAL' in row.index and 'AMT_CREDIT' in row.index:
			try:
				out['INCOME_CREDIT_PERC'] = float(row['AMT_INCOME_TOTAL']) / float(row['AMT_CREDIT']) if float(row['AMT_CREDIT']) != 0 else self.fill_value
			except Exception:
				out['INCOME_CREDIT_PERC'] = self.fill_value

		# INCOME_PER_PERSON = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
		if 'AMT_INCOME_TOTAL' in row.index and 'CNT_FAM_MEMBERS' in row.index:
			try:
				cnt = float(row['CNT_FAM_MEMBERS']) if float(row['CNT_FAM_MEMBERS']) not in (0, None) else 1.0
				out['INCOME_PER_PERSON'] = float(row['AMT_INCOME_TOTAL']) / cnt
			except Exception:
				out['INCOME_PER_PERSON'] = self.fill_value

		# ANNUITY_INCOME_PERC = AMT_ANNUITY / AMT_INCOME_TOTAL
		if 'AMT_ANNUITY' in row.index and 'AMT_INCOME_TOTAL' in row.index:
			try:
				out['ANNUITY_INCOME_PERC'] = float(row['AMT_ANNUITY']) / float(row['AMT_INCOME_TOTAL']) if float(row['AMT_INCOME_TOTAL']) != 0 else self.fill_value
			except Exception:
				out['ANNUITY_INCOME_PERC'] = self.fill_value

		# DAYS_EMPLOYED_PERC = DAYS_EMPLOYED / DAYS_BIRTH (both negative; ratio meaningful)
		if 'DAYS_EMPLOYED' in row.index and 'DAYS_BIRTH' in row.index:
			try:
				out['DAYS_EMPLOYED_PERC'] = float(row['DAYS_EMPLOYED']) / float(row['DAYS_BIRTH']) if float(row['DAYS_BIRTH']) != 0 else self.fill_value
			except Exception:
				out['DAYS_EMPLOYED_PERC'] = self.fill_value

		return out

	def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
		"""Transform a single-row (or multi-row) raw DataFrame into model features.

		Behaviour:
		- If an expected column exists in df_raw it is copied.
		- Try to compute derived numeric features from base columns.
		- Map raw categorical columns to one-hot expected columns by prefix match + sanitized value.
		- Fill any remaining expected columns with the per-column median (if known) or `fill_value`.
		"""
		if not isinstance(df_raw, pd.DataFrame):
			raise TypeError("df_raw doit être un pandas.DataFrame")

		if not self.expected_features:
			# Nothing to map to — return copy of input
			return df_raw.copy()

		# Sanitize input column names so they match model feature names
		df_raw = df_raw.copy()
		df_raw.columns = [self._sanitize_column_name(c) for c in df_raw.columns]

		out_rows = []
		for _, row in df_raw.iterrows():
			# start from an empty output dict for the expected features
			out = {feat: None for feat in self.expected_features}

			# 1) copy direct matches
			for feat in list(out.keys()):
				if feat in row.index:
					val = row[feat]
					out[feat] = int(val) if isinstance(val, (bool, np.bool_)) else (val if not self._is_nan(val) else None)

			# 2) compute derived numeric features and set if present in expected_features
			derived = self._compute_derived(row)
			for k, v in derived.items():
				if k in out:
					out[k] = v

			# 3) categorical -> one-hot mapping using base column names from raw row
			for base_col in row.index:
				if pd.isna(row[base_col]):
					continue
				# sanitize raw value once
				raw_s = self._sanitize_category(row[base_col])
				for feat in self.expected_features:
					prefix = feat.split('_')[0]
					# better check: if feature name starts with base_col + '_'
					if feat.startswith(f"{base_col}_"):
						suffix = feat[len(base_col) + 1 :]
						# compare sanitized forms
						if suffix == raw_s:
							out[feat] = 1
						elif out[feat] is None:
							# set 0 only if not already set to 1
							out[feat] = 0

			# 4) final pass: fill remaining None values with impute median or fill_value
			for feat in out:
				if out[feat] is None:
					if feat in self._impute_values:
						out[feat] = self._impute_values[feat]
					else:
						out[feat] = self.fill_value

			out_rows.append(out)

		result = pd.DataFrame(out_rows, columns=self.expected_features)

		# cast numeric-like columns to numeric
		for col in result.columns:
			try:
				result[col] = pd.to_numeric(result[col], errors='coerce').fillna(self.fill_value)
			except Exception:
				pass

		return result

	def get_feature_names_out(self) -> List[str]:
		return list(self.expected_features)


# =============================================================================
# VectorizedPreprocessor — VERSION OPTIMISÉE 4.4 (Gain 15.7x)
# Wrappeur vectorisé de RawToModelTransformer pour batch et requêtes unitaires.
# Source : notebooks/10_optimisation.ipynb — Cellule 3
# =============================================================================

class VectorizedPreprocessor:
	"""Preprocessor vectorisé pour traiter PLUSIEURS lignes en UNE seule opération.

	Gain de performance : 15.7x plus rapide que la boucle ligne par ligne
	grâce à la construction du DataFrame depuis une liste de dicts en une
	seule opération pandas (pd.DataFrame(payloads)).

	Usage dans app.py :
		prep = VectorizedPreprocessor(base_transformer)
		df = prep.transform_single(payload_dict)        # requête API unique
		df = prep.transform_batch([dict1, dict2, ...])  # batch
		df = prep.transform_one_sample(json_string)     # depuis JSON brut
	"""

	def __init__(self, base_transformer: "RawToModelTransformer") -> None:
		"""Initialise avec un transformer de base (récupère expected_features + impute)."""
		self.base_transformer = base_transformer
		# Accès direct aux attributs clés pour éviter les appels répétés
		self.expected_features = base_transformer.expected_features
		self._impute_values = base_transformer._impute_values

	def transform_batch(self, payloads: list) -> pd.DataFrame:
		"""Transforme une liste de dicts (payloads JSON) → DataFrame features.

		Étapes :
		1. Convertir liste de dicts → DataFrame en UNE opération pandas vectorisée
		2. Nettoyage standard (empty string, boolean string, numeric coercion)
		3. Appliquer le transformer de base (one-hot, médiane, derived features)
		4. Retourner DataFrame prêt pour le modèle LightGBM
		"""
		# === ÉTAPE 1 : Construction vectorisée du DataFrame (cœur du gain 15.7x) ===
		df = pd.DataFrame(payloads)

		# === ÉTAPE 2 : Nettoyage standard (same as _parse_json_line) ===
		df = df.replace({"": np.nan, "True": True, "False": False})

		# Conversion numérique (LightGBM exige des colonnes numériques)
		for col in df.columns:
			try:
				df[col] = pd.to_numeric(df[col], errors='coerce')
			except Exception:
				pass

		# === ÉTAPE 3 : Transformer de base (one-hot, dérivées, imputations) ===
		df = self.base_transformer.transform(df)

		return df

	def transform_single(self, payload: dict) -> pd.DataFrame:
		"""Transforme UN SEUL dict (payload JSON parsé) → DataFrame (1 ligne)."""
		return self.transform_batch([payload])

	def transform_one_sample(self, json_line: str) -> pd.DataFrame:
		"""Parse un JSON string et transforme → DataFrame (1 ligne).

		Point d'entrée principal dans app.py :
			df = PREPROCESSOR.transform_one_sample(json_line)
		"""
		import json as _json
		payload = _json.loads(json_line)
		return self.transform_single(payload)

	def get_feature_names_out(self) -> List[str]:
		return list(self.expected_features)
