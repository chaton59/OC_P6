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

# Projet Credit Scoring - Home Credit Default Risk

## 📋 Description

Projet de prédiction du risque de défaut de paiement pour Home Credit. Ce projet utilise des techniques de machine learning pour prédire la probabilité qu'un client ne rembourse pas son crédit, avec un focus sur l'optimisation du coût métier et l'interprétabilité des modèles.

## 🏗️ Structure du projet

```
OC_P6/
├── data/
│   ├── raw/                          # Données brutes (non versionnées)
│   │   ├── application_train.csv
│   │   ├── application_test.csv
│   │   ├── bureau.csv
│   │   ├── bureau_balance.csv
│   │   ├── credit_card_balance.csv
│   │   ├── installments_payments.csv
│   │   ├── POS_CASH_balance.csv
│   │   └── previous_application.csv
│   └── processed/                    # Datasets prétraités (non versionnés)
│       ├── features_full.csv
│       ├── features_train.csv
│       └── features_test.csv
├── notebooks/                        # Notebooks d'apprentissage
│   ├── 01_exploration.ipynb         # EDA complète
│   └── 02_preparation_features.ipynb # Feature Engineering
├── src/                              # Code Python réutilisable
│   ├── __init__.py
│   └── data/
│       └── load_data.py              # Fonction de chargement des données
├── projet/                           # Documents de mission
│   ├── mission.txt
│   └── etapes.txt
├── pyproject.toml                    # Configuration projet et dépendances (UV)
├── .gitignore                        # Protège les données
├── README.md
└── uv.lock                           # Lock des dépendances
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

**Contenu :**
- Chargement et première inspection des données
- Analyse de la variable cible (déséquilibre des classes)
- Analyse des valeurs manquantes
- Exploration des corrélations
- Détection d'anomalies (DAYS_EMPLOYED = 365243)
- Analyse des variables EXT_SOURCE (prédicteurs clés)

### 2. Préparation des features (Feature Engineering)

```bash
jupyter notebook notebooks/02_preparation_features.ipynb
```

**Contenu :**
- Chargement et fusion de 7 tables de données
- Nettoyage des données (valeurs aberrantes, sentinelles)
- Encodage des variables catégorielles (One-Hot encoding)
- Création de features par agrégation (min, max, mean, sum, var)
- Features spécifiques :
  - Ratios et pourcentages (ex: INCOME_CREDIT_PERC, PAYMENT_RATE)
  - Comportement de paiement (DPD, DBD)
  - Crédits actifs vs fermés
  - Demandes approuvées vs refusées
- Séparation train/test
- Sauvegarde des datasets préparés

**Output :** 
- `data/processed/features_full.csv` (~800+ features)
- `data/processed/features_train.csv`
- `data/processed/features_test.csv`

### 3. Modélisation LightGBM

```bash
jupyter notebook notebooks/03_LGBM.ipynb
```

**Contenu :**
- Modélisation LightGBM avec validation croisée stratifiée
- Gestion du déséquilibre des classes
- Optimisation d’hyperparamètres (Optuna)
- Optimisation du seuil métier (coût FN vs FP)
- Tracking des expérimentations avec MLflow

### 4. Régression logistique (baseline + déséquilibre)

```bash
jupyter notebook notebooks/04_regression.ipynb
```

**Contenu :**
- Modèle baseline
- Gestion du déséquilibre (class_weight, SMOTE)
- Comparaison des versions
- Logging MLflow des métriques et modèles

### 5. Interprétation des modèles

```bash
jupyter notebook notebooks/05_model_interpretation.ipynb
```

**Contenu :**
- Importance globale des features (LightGBM)
- Importance par permutation
- Interprétation locale via SHAP

### 6. Évaluation finale

```bash
jupyter notebook notebooks/06_final_evaluation.ipynb
```

**Contenu :**
- Évaluation finale sur jeu de test (ou split de validation si test indisponible)
- Optimisation du seuil métier
- Comparaison synthétique des modèles

## 📝 Approche

Ce projet suit l'approche du kernel Kaggle **"LightGBM with Simple Features"** de [jsaguiar](https://www.kaggle.com/jsaguiar), qui a obtenu d'excellents résultats sur cette compétition.

**Stratégie :**
- Modulabilité : une fonction pour chaque table de données
- Agrégations statistiques sur les données groupées
- Création de ratios et pourcentages entre variables importantes
- Features spécifiques pour différents profils (crédits actifs/fermés, demandes approuvées/refusées)

**Approche de modélisation prévue :**
1. Feature Selection : identifier les features les plus importantes
2. Modélisation : LightGBM avec validation croisée (K-Fold)
3. Optimisation : tuning des hyperparamètres
4. Évaluation : métriques (ROC-AUC, coûts métier)
5. Prédictions : générer les prédictions pour le test set
## Déploiement

### Test

- Space de test : https://huggingface.co/spaces/ASI-Engineer/OC_P8_test
- Branche utilisée : `dev` (builds déclenchés automatiquement au push)

### Production

- Space de production : https://huggingface.co/spaces/ASI-Engineer/OC_P8_prod
- Branche utilisée : `main`

### Lancement local avec Docker

Construire l'image puis lancer le conteneur :

```bash
docker build -t oc_p6:latest .
docker run --rm -it -p 7860:7860 oc_p6:latest
```

### Lancement local (le plus simple)

Utilisez le script d'aide qui crée un virtualenv, installe les dépendances et démarre l'API :

```bash
bash scripts/setup_dev.sh
```

> Si vous préférez exécuter manuellement :
>
> python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python app.py


### Tester l'API

L'API attend une *ligne JSON* (une seule observation). Exemple JSON minimal :

```json
{"SK_ID_CURR": 100001, "AMT_INCOME_TOTAL": 202500.0, "AMT_CREDIT": 80000.0, "CODE_GENDER": "M", "DAYS_BIRTH": -12000}
```

Exemple de requête (POST) vers la Space de production :

```bash
curl -s -X POST "https://huggingface.co/spaces/ASI-Engineer/OC_P8_prod/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"data":["{\"SK_ID_CURR\":100001,\"AMT_INCOME_TOTAL\":202500.0,\"AMT_CREDIT\":80000.0,\"CODE_GENDER\":\"M\",\"DAYS_BIRTH\":-12000}"]}'
```

La réponse contient : `Score`, `Probabilité de défaut` et `Décision`.

### Seuil de décision

- **Seuil par défaut : 0.4** (si probabilité de défaut ≥ 0.4 → **Refusé**)

---

## 🎯 Étape 3 - Monitoring Production & Data Drift

### Objectif
Mettre en place un système de monitoring léger en production pour détecter les dérives de données et surveiller les performances opérationnelles de l'API.

### Résultats clés
✅ **Logging structuré JSON** implémenté dans l'API (`app.py`)  
✅ **200+ appels simulés** en production avec `reference/simulate_production_calls.py`  
✅ **Analyse opérationnelle** : latence moyenne & distribution des scores  
✅ **Detection Data Drift** avec Evidently : **25 features impactées**  
✅ **Feature critique** : `AMT_INCOME_TOTAL` (Drift Score = 0.304)

### Contenu

- **[📊 Rapport complet](reports/monitoring_study.md)** - Analyse complète avec visualisations et recommandations
- **[📈 Dashboard Drift](reports/data_drift_report.html)** - Rapport Evidently (drift détecté/non détecté par feature)
- **[🕐 Latence API](reports/plots/latence_histogram.html)** - Distribution du temps d'exécution
- **[📉 Distribution Probabilités](reports/plots/proba_histogram.html)** - Scores de risque en production

### Architecture

```
Logs (JSONL) ──> Pandas + Plotly ──> HTML Reports
                      │
                      └──> Evidently ──> Data Drift Report
```

**Stockage**: Fichier JSONL léger (< 1 Mo pour 10k appels)  
**Alertes futures**: Seuil drift > 0.25 → Slack/Email  
**Amélioration**: Intégration NannyML ou MLflow pour prediction drift

### PoC Status
✅ **Local PoC: 100% terminé**

---

## 🤝 Contribution

Les contributions sont les bienvenues! Merci de:
1. Créer une branche pour votre feature
2. Respecter le style de code
3. Mettre à jour la documentation

## 📄 Licence

Ce projet est à usage éducatif.

## 🎓 Status d'apprentissage

**Phase actuelle :** Exploration et Feature Engineering  
**Prochaines phases :** Modélisation et Optimisation

Ce projet est conçu comme un parcours d'apprentissage en machine learning appliqué au credit scoring.# Test push credentials
# Test 2
