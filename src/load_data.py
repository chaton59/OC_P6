"""
Module pour charger les données brutes du projet Home Credit.
"""

import pandas as pd
from pathlib import Path
from typing import Dict
import os


class DataContainer(dict):
    """
    Conteneur de données permettant l'accès par clé (dict-like) et par attribut.
    
    Usage:
        data = DataContainer({'df1': pd.DataFrame(), 'df2': pd.DataFrame()})
        data.df1  # Accès par attribut
        data['df1']  # Accès par clé
    """
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'DataContainer' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value):
        self[name] = value


def _find_project_root() -> Path:
    """
    Trouve la racine du projet de manière robuste.
    Stratégie :
    1. Si __file__ existe (script .py) → on remonte comme avant.
    2. Sinon (notebook), on part du répertoire courant et on cherche un marqueur
       classique de projet : le dossier 'data/raw' contenant 'application_train.csv'.
    Cela évite les erreurs de contexte d'exécution.
    """
    try:
        # Cas classique : exécuté comme module .py
        return Path(__file__).resolve().parent.parent.parent
    except (NameError, RuntimeError):
        # Cas notebook / interactive
        current = Path.cwd()
        # On remonte jusqu'à trouver le dossier contenant data/raw/application_train.csv
        for p in [current] + list(current.parents):
            candidate = p / "data" / "raw" / "application_train.csv"
            if candidate.exists():
                return p
        
        # Fallback: cherche un dossier nommé OC_P6 avec data/raw dedans
        for p in [current] + list(current.parents):
            candidate = p / "data" / "raw" / "application_train.csv"
            if candidate.exists():
                return p
            # Cherche aussi dans OC_P6 s'il est un sous-dossier
            oc_p6 = p / "OC_P6"
            if oc_p6.exists():
                candidate = oc_p6 / "data" / "raw" / "application_train.csv"
                if candidate.exists():
                    return oc_p6
        
        raise FileNotFoundError("Impossible de trouver la racine du projet. Vérifie la structure des dossiers.")


BASE_DIR = _find_project_root()


def load_raw_data(data_dir: str | None = None) -> DataContainer:
    """
    Charge toutes les données brutes.
    
    Retourne un conteneur permettant l'accès par attribut et par clé :
        raw_data = load_raw_data()
        raw_data.application_train  # Accès par attribut
        raw_data['application_train']  # Accès par clé
    """
    if data_dir is None:
        # First try to use provided BASE_DIR
        if not (BASE_DIR / "data" / "raw" / "application_train.csv").exists():
            # If BASE_DIR doesn't have data, search from current working directory
            current = Path.cwd()
            found = False
            for p in [current] + list(current.parents):
                candidate_file = p / "data" / "raw" / "application_train.csv"
                if candidate_file.exists():
                    data_path = p / "data" / "raw"
                    found = True
                    break
            
            if not found:
                raise FileNotFoundError(
                    f"Data files not found. Searched in {BASE_DIR / 'data' / 'raw'} "
                    f"and from {current} upwards."
                )
        else:
            data_path = BASE_DIR / "data" / "raw"
    else:
        data_path = Path(data_dir)
    
    print(f"Chargement depuis : {data_path.resolve()}")  # Utile pour debug
    
    datasets = {
        'application_train': 'application_train.csv',
        'application_test': 'application_test.csv',
        'bureau': 'bureau.csv',
        'bureau_balance': 'bureau_balance.csv',
        'credit_card_balance': 'credit_card_balance.csv',
        'installments_payments': 'installments_payments.csv',
        'POS_CASH_balance': 'POS_CASH_balance.csv',
        'previous_application': 'previous_application.csv'
    }
    
    data = {}
    for name, filename in datasets.items():
        filepath = data_path / filename
        if filepath.exists():
            print(f"✓ Chargement de {filename}")
            data[name] = pd.read_csv(filepath)
        else:
            print(f"✗ Fichier manquant : {filename} (chemin : {filepath.resolve()})")
    
    return DataContainer(data)


def load_processed_data(data_dir: str = "data/processed") -> Dict[str, pd.DataFrame]:
    """
    Charge les données prétraitées.
    
    Args:
        data_dir: Chemin vers le dossier contenant les données traitées
        
    Returns:
        Dictionnaire contenant les DataFrames train et test
    """
    data_path = Path(data_dir)
    
    data = {}
    train_path = data_path / "train_processed.pkl"
    test_path = data_path / "test_processed.pkl"
    
    if train_path.exists():
        data['train'] = pd.read_pickle(train_path)
    if test_path.exists():
        data['test'] = pd.read_pickle(test_path)
    
    return data
