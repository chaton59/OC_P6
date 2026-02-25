---
title: Credit Scoring - Home Credit Default Risk
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.12"
app_file: app.py
pinned: false
---

# OC_P6 - API Scoring Credit (MLOps)

## 🚀 Demo live
https://huggingface.co/spaces/ASI-Engineer/OC_P8_prod
https://huggingface.co/spaces/ASI-Engineer/OC_P8_test

## Resultats optimisation etape 4
- Gain latence : **15.7x** (0.64 ms -> 0.04 ms par requete)
- Precision : 100 % identique
- Voir [reports/rapport_optimisation.md](reports/rapport_optimisation.md) complet

## Architecture finale
- FastAPI/Gradio + Docker (entrypoint : [app.py](app.py))
- Monitoring logs + Evidently (drift)
- Optimisation : VectorizedPreprocessor (15.7x)

## Etapes realisees
- Etape 2 : API + Docker + CI/CD
- Etape 3 : Stockage + analyse prod
- Etape 4 : Optimisation perfs (terminee)

## Apercu du projet (audit rapide)
- Donnees brutes et features : [data/raw](data/raw), [data/processed](data/processed)
- Pipeline data/model : [src/load_data.py](src/load_data.py), [src/preprocessing.py](src/preprocessing.py)
- Experiments et artefacts : [mlruns](mlruns), [models](models)
- Notebooks MLOps : [notebooks](notebooks)
- Monitoring prod : [logs/predictions.jsonl](logs/predictions.jsonl), [reports](reports)
- Tests : [tests](tests)
- Conteneurisation : [Dockerfile](Dockerfile)

## Structure du projet
```
OC_P6/
├── app.py
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── requirements-inference.txt
├── data/
│   ├── raw/
│   └── processed/
├── logs/
│   └── predictions.jsonl
├── mlruns/
├── models/
│   ├── export_model.py
│   ├── export_preprocessor.py
│   ├── lightgbm.txt
│   └── preprocessor.joblib
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preparation_features.ipynb
│   ├── 03_LGBM.ipynb
│   ├── 04_regression.ipynb
│   ├── 05_model_interpretation.ipynb
│   ├── 06_analyse_logs.ipynb
│   ├── 07_detect_data_drift.ipynb
│   ├── 08_analyze_logs_2.ipynb
│   ├── 09_profiling.ipynb
│   └── 10_optimisation.ipynb
├── reference/
│   ├── reference.csv
│   └── simulate_production_calls.py
├── reports/
│   ├── data_drift_report.html
│   ├── monitoring_study.md
│   └── plots/
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── mlflow_config.py
│   └── preprocessing.py
└── tests/
  ├── conftest.py
  ├── test_predict.py
  └── test_preprocessing.py
```

## Installation (UV recommande)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## Donnees
Source : Kaggle Home Credit Default Risk.
Placer les fichiers dans [data/raw](data/raw) :
- application_train.csv
- application_test.csv
- bureau.csv
- bureau_balance.csv
- credit_card_balance.csv
- installments_payments.csv
- POS_CASH_balance.csv
- previous_application.csv

## Notebooks (resume)
- Exploration : [notebooks/01_exploration.ipynb](notebooks/01_exploration.ipynb)
- Feature engineering : [notebooks/02_preparation_features.ipynb](notebooks/02_preparation_features.ipynb)
- Modelling LGBM + MLflow : [notebooks/03_LGBM.ipynb](notebooks/03_LGBM.ipynb)
- Baseline regression : [notebooks/04_regression.ipynb](notebooks/04_regression.ipynb)
- Interpretation : [notebooks/05_model_interpretation.ipynb](notebooks/05_model_interpretation.ipynb)
- Monitoring et drift : [notebooks/06_analyse_logs.ipynb](notebooks/06_analyse_logs.ipynb), [notebooks/07_detect_data_drift.ipynb](notebooks/07_detect_data_drift.ipynb)
- Profiling et optimisation : [notebooks/09_profiling.ipynb](notebooks/09_profiling.ipynb), [notebooks/10_optimisation.ipynb](notebooks/10_optimisation.ipynb)

## Comment tester localement
```bash
uv sync
uv run python app.py
```

Option Docker :
```bash
docker build -t oc_p6:latest .
docker run --rm -it -p 7860:7860 oc_p6:latest
```

## Usage API (local ou HF Space)
Exemple JSON minimal :
```json
{"SK_ID_CURR": 100001, "AMT_INCOME_TOTAL": 202500.0, "AMT_CREDIT": 80000.0, "CODE_GENDER": "M", "DAYS_BIRTH": -12000}
```

Requete vers la Space de production :
```bash
curl -s -X POST "https://huggingface.co/spaces/ASI-Engineer/OC_P8_prod/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"data":["{\"SK_ID_CURR\":100001,\"AMT_INCOME_TOTAL\":202500.0,\"AMT_CREDIT\":80000.0,\"CODE_GENDER\":\"M\",\"DAYS_BIRTH\":-12000}"]}'
```

## Monitoring et data drift
- Rapport monitoring : [reports/monitoring_study.md](reports/monitoring_study.md)
- Rapport drift Evidently : [reports/data_drift_report.html](reports/data_drift_report.html)
- Plots latence et scores : [reports/plots](reports/plots)
- Simulation d'appels prod : [reference/simulate_production_calls.py](reference/simulate_production_calls.py)

## Tests
```bash
uv run pytest
```

**Date** : 25 fevrier 2026  
**Statut** : Projet termine OK, pret pour soutenance
