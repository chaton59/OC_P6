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

Ce projet est conçu comme un parcours d'apprentissage en machine learning appliqué au credit scoring.