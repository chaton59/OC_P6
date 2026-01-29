"""
Configuration globale du projet.
"""

from pathlib import Path

# Chemins des données
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Chemins MLflow
MLFLOW_TRACKING_URI = "./mlruns"
EXPERIMENT_NAME = "credit_scoring"

# Paramètres du modèle
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Coûts métier
COST_FALSE_NEGATIVE = 10.0  # Coût d'accorder un crédit à un mauvais payeur
COST_FALSE_POSITIVE = 1.0    # Coût de refuser un crédit à un bon payeur
REVENUE_TRUE_POSITIVE = 5.0  # Revenu d'un crédit accordé à un bon payeur

# Seuil de décision par défaut
DEFAULT_THRESHOLD = 0.5

# Paramètres des modèles par défaut
LIGHTGBM_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.05,
    'max_depth': 7,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Colonnes à exclure du training
EXCLUDED_COLUMNS = ['SK_ID_CURR', 'TARGET']
