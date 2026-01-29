# Projet Credit Scoring - Home Credit Default Risk

## 📋 Description

Projet de prédiction du risque de défaut de paiement pour Home Credit. Ce projet utilise des techniques de machine learning pour prédire la probabilité qu'un client ne rembourse pas son crédit, avec un focus sur l'optimisation du coût métier et l'interprétabilité des modèles.

## 🏗️ Structure du projet

```
projet_credit_scoring/
├── data/
│   ├── raw/                  # Données brutes (non versionnées)
│   │   └── .gitkeep
│   └── processed/            # Datasets prétraités (non versionnés)
│       └── .gitkeep
├── notebooks/                # Notebooks d'exploration et expérimentation
│   ├── 01_exploration.ipynb
│   ├── 02_preparation_features.ipynb
│   ├── 03_modeling_mlflow.ipynb
│   ├── 04_hyperopt_threshold.ipynb
│   └── 05_interpretability.ipynb
├── src/                      # Code Python réutilisable
│   ├── __init__.py
│   ├── data/
│   │   ├── load_data.py
│   │   ├── clean_and_merge.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   ├── utils/
│   │   ├── metrics.py        # Fonction de coût métier
│   │   └── mlflow_helpers.py
│   └── config.py             # Configuration globale
├── models/                   # Modèles sauvegardés localement
├── mlruns/                   # Tracking MLFlow
├── experiments/              # Artefacts MLFlow
├── tests/                    # Tests unitaires
│   └── test_preprocessing.py
├── pyproject.toml            # Configuration projet et dépendances (UV)
├── .python-version           # Version Python (3.12)
├── .gitignore
├── README.md
└── serve_model.py            # Script de serving MLFlow
```

## 🚀 Installation

Ce projet utilise **[UV](https://docs.astral.sh/uv/)** pour la gestion des dépendances.

### Installation avec UV (recommandé)

```bash
# Installer UV si pas déjà fait
curl -LsSf https://astral.sh/uv/install.sh | sh

# Synchroniser l'environnement et installer les dépendances
uv sync

# Activer l'environnement virtuel
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows
```

### Installer les dépendances de développement

```bash
uv sync --extra dev
```

### Ajouter une nouvelle dépendance

```bash
uv add nom-du-package
```

## 📊 Données

Les données proviennent du concours Kaggle [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).

Téléchargez les fichiers suivants et placez-les dans `data/raw/`:
- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `bureau_balance.csv`
- `credit_card_balance.csv`
- `installments_payments.csv`
- `POS_CASH_balance.csv`
- `previous_application.csv`

## 🎯 Utilisation

### 1. Exploration des données

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

### 2. Préparation des features

```bash
jupyter notebook notebooks/02_preparation_features.ipynb
```

### 3. Modélisation avec MLflow

```bash
# Lancer l'UI MLflow (optionnel)
mlflow ui

# Puis ouvrir le notebook
jupyter notebook notebooks/03_modeling_mlflow.ipynb
```

### 4. Optimisation des hyperparamètres

```bash
jupyter notebook notebooks/04_hyperopt_threshold.ipynb
```

### 5. Interprétabilité

```bash
jupyter notebook notebooks/05_interpretability.ipynb
```

uv run pytest

# Avec couverture détaillée
uv run pytest
pytest tests/

# Avec couverture
pytest tests/ --cov=src --cov-report=html
```

## 📈 MLflow

uv run Le projet utilise MLflow pour le tracking des expériences.

```bash
# Lancer l'interface MLflow
mlflow ui

# Puis ouvrir http://localhost:5000
```

## 🔧 Serving du modèle
uv run 
Pour servir un modèle en production:

```bash
python serve_model.py --model-uri models:/credit_scoring/Production --port 5001
```

Endpoints disponibles:
- `GET /health` - Vérifier le statut
- `POST /predict` - Faire des prédictions
- `POST /predict_proba` - Obtenir les probabilités

## 📝 Configuration

Les paramètres principaux sont dans [src/config.py](src/config.py):
- Coûts métier (faux positifs/négatifs)
- Chemins des données
- Paramètres des modèles par défaut

## 🤝 Contribution

Les contributions sont les bienvenues! Merci de:
1. Créer une branche pour votre feature
2. Écrire des tests pour votre code
3. Respecter le style de code (black, flake8)

## 📄 Licence

Ce projet est à usage éducatif.
