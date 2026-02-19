"""Create and persist the preprocessing transformer used by the API.

Run this script after you change `data/processed/features_train.csv` to refresh the
serialized preprocessor at `models/preprocessor.joblib`.
"""
from pathlib import Path
import joblib

from src.preprocessing import RawToModelTransformer

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREPROC_PATH = MODEL_DIR / "preprocessor.joblib"

pre = RawToModelTransformer()
print(f"Inferred {len(pre.get_feature_names_out())} expected features")

joblib.dump(pre, PREPROC_PATH)
print(f"Preprocessor saved to {PREPROC_PATH.resolve()}")
