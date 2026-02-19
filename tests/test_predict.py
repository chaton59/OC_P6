import numpy as np
import pandas as pd
import pytest

import app as app_module

try:
	from app import predict_score, model
except ImportError:
	from app import _predict as predict_score
	model = app_module.MODEL


class DummyModel:
	def __init__(self, proba: float = 0.2) -> None:
		self.proba = proba

	def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
		return np.array([[1.0 - self.proba, self.proba]])

	def predict(self, df: pd.DataFrame) -> np.ndarray:
		return np.array([self.proba])


def _series_json(payload: dict) -> str:
	# Convert a single-record payload using Series.to_json(orient="records").
	# Pandas returns a one-item list; trim brackets to get the JSON object.
	return pd.Series([payload]).to_json(orient="records")[1:-1]


def _extract_proba(response: str) -> float:
	for line in response.splitlines():
		if line.startswith("Probabilit"):
			return float(line.split(":", 1)[1].strip())
	raise AssertionError("Probability line not found in response")


@pytest.fixture()
def dummy_model(monkeypatch: pytest.MonkeyPatch) -> DummyModel:
	# Patch the global model so tests are fast and independent of disk artifacts.
	dummy = DummyModel(proba=0.23)
	monkeypatch.setattr(app_module, "MODEL", dummy, raising=False)
	monkeypatch.setattr(app_module, "model", dummy, raising=False)
	return dummy


def test_predict_valid_minimal_json(dummy_model: DummyModel) -> None:
	# Valid minimal JSON should yield a probability between 0 and 1.
	payload = {
		"EXT_SOURCE_1": 0.5,
		"AMT_INCOME_TOTAL": 50000.0,
	}
	json_line = _series_json(payload)
	response = predict_score(json_line)

	assert "Erreur" not in response
	proba = _extract_proba(response)
	assert 0.0 <= proba <= 1.0


def test_predict_partial_json_missing_columns(dummy_model: DummyModel) -> None:
	# Missing columns should be handled (reindex + NaN) and still predict.
	payload = {
		"EXT_SOURCE_2": 0.1,
	}
	json_line = _series_json(payload)
	response = predict_score(json_line)

	assert "Erreur" not in response
	proba = _extract_proba(response)
	assert 0.0 <= proba <= 1.0


def test_predict_invalid_json_returns_error() -> None:
	# Bad JSON format should return an explicit error message.
	json_line = "{this is not valid json"
	response = predict_score(json_line)

	assert "Erreur" in response


def test_predict_out_of_range_value(dummy_model: DummyModel) -> None:
	# Aberrant values (e.g., negative income) should still predict for now.
	payload = {
		"AMT_INCOME_TOTAL": -1000.0,
		"EXT_SOURCE_3": 0.2,
	}
	json_line = _series_json(payload)
	response = predict_score(json_line)

	assert "Erreur" not in response
	proba = _extract_proba(response)
	assert 0.0 <= proba <= 1.0


def test_predict_accepts_raw_categorical(dummy_model: DummyModel) -> None:
	# The API should accept raw categorical fields and map them to the model's
	# one-hot columns (e.g. NAME_CONTRACT_TYPE -> NAME_CONTRACT_TYPE_Cash loans).
	payload = {
		"NAME_CONTRACT_TYPE": "Cash loans",
		"AMT_INCOME_TOTAL": 75000.0,
		"EXT_SOURCE_1": 0.3,
	}
	json_line = _series_json(payload)
	response = predict_score(json_line)

	assert "Erreur" not in response
	proba = _extract_proba(response)
	assert 0.0 <= proba <= 1.0
