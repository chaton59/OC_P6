"""Simulateur d'appels de production pour remplir logs/predictions.jsonl."""

# EXPLICATION : Imports standards uniquement (aucune dépendance nouvelle)
import requests
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path

# EXPLICATION : Chargement du dataset de référence (500 lignes échantillonnées de features_train)
# Path(__file__).parent rend le chemin robuste quel que soit le répertoire courant
reference = pd.read_csv(Path(__file__).parent / "reference.csv")

# EXPLICATION : Gradio 5.x utilise une API SSE en 2 étapes :
#   1) POST /gradio_api/call/<fn_name> → retourne un event_id
#   2) GET  /gradio_api/call/<fn_name>/<event_id> → stream SSE avec le résultat
BASE_URL = "http://127.0.0.1:7860"
CALL_URL = f"{BASE_URL}/gradio_api/call/_predict"

# EXPLICATION : Tirage aléatoire de 500 lignes (avec remise si dataset < 500)
# random_state=42 pour reproductibilité, replace=True pour éviter l'erreur si reference < 500
sampled = reference.sample(n=500, replace=True, random_state=42).reset_index(drop=True)

# EXPLICATION : Boucle de 500 appels simulés (375 normaux + 125 avec drift)
for i in range(500):
    # EXPLICATION : Sélection de la ligne aléatoire pré-tirée
    row = sampled.iloc[i].to_dict()

    # EXPLICATION : Nettoyage — convertir "" et NaN en None pour JSON propre
    for k, v in row.items():
        if v == "" or pd.isna(v):
            row[k] = None

    # EXPLICATION : 25% des appels avec drift simulé (AMT_INCOME_TOTAL * 1.5)
    if i % 4 == 0:
        row["AMT_INCOME_TOTAL"] = row["AMT_INCOME_TOTAL"] * 1.5 if row["AMT_INCOME_TOTAL"] else 100000

    # EXPLICATION : Format payload attendu par l'interface Gradio (app.py)
    payload = {"data": [json.dumps(row)]}

    start = time.perf_counter()
    drift_tag = " [DRIFT]" if i % 4 == 0 else ""
    try:
        # EXPLICATION : Étape 1 — POST pour obtenir un event_id
        resp = requests.post(CALL_URL, json=payload, timeout=10)
        resp.raise_for_status()
        event_id = resp.json().get("event_id")

        # EXPLICATION : Étape 2 — GET SSE pour récupérer le résultat
        result_url = f"{CALL_URL}/{event_id}"
        sse_resp = requests.get(result_url, timeout=30, stream=True)
        sse_resp.raise_for_status()

        # EXPLICATION : Parse la réponse SSE (format "event: ...\ndata: ...\n")
        result_text = ""
        for line in sse_resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                result_text = line[len("data:"):].strip()

        duration = (time.perf_counter() - start) * 1000
        print(f"Appel {i+1}/500 - OK - Temps: {duration:.1f}ms{drift_tag}")
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        print(f"Erreur appel {i+1}: {e} ({duration:.1f}ms){drift_tag}")

    # EXPLICATION : Pause entre chaque appel pour ne pas surcharger Docker
    time.sleep(0.3)

# Sous-étape 4 terminée - 500 appels simulés (375 normal + 125 avec drift)
# Lancer avec : uv run python simulate_production_calls.py (API doit tourner sur 7860)
