# Étape 3 - Plan de monitoring production (Sous-étape 1 terminé)

## Audit du code actuel (app.py)

| **Élément** | **État actuel** | **Impact** | **Action requise** |
|-------------|-----------------|------------|-------------------|
| **Logging entrées** | ❌ Absent | Impossible de debugger les erreurs clients ou rejouer les prédictions | Ajouter logging de `input_raw` et `input_features` |
| **Logging sorties** | ❌ Absent | Pas de traçabilité des décisions métier (conformité RGPD/audit) | Logger `output_proba`, `output_decision` |
| **Horodatage** | ❌ Absent | Impossible d'analyser les tendances temporelles ou les pics de charge | Ajouter `timestamp` UTC au format ISO 8601 |
| **Temps d'exécution** | ❌ Absent | Pas de détection d'anomalies de latence (SLA non mesurable) | Instrumenter avec `time.perf_counter()` → `execution_time_ms` |
| **Logging erreurs** | ⚠️ Partiel | Erreurs capturées mais non structurées (try/except retourne strings) | Ajouter champ `error` avec message + type d'exception |
| **Version modèle** | ❌ Absent | Impossible de relier une prédiction à une version spécifique du modèle | Logger `model_version` (ex: "models:/LightGBM/Production" ou hash) |
| **Seuil de décision** | ⚠️ Hardcodé | Seuil fixe à 0.4 non tracé (risque si modification ultérieure) | Logger `threshold` pour reproductibilité |
| **Format de sortie** | ❌ Texte libre | Résultat en string non parsable (impossible pour Evidently/dashboards) | Retourner JSON structuré + logger en JSON |
| **Persistance logs** | ❌ Absent | Aucune sauvegarde des logs (pertes à chaque redémarrage) | Écrire dans fichier `logs/predictions.jsonl` (1 ligne = 1 log) |
| **Rotation logs** | ❌ Absent | Risque de saturation disque si volume élevé | Implémenter rotation (ex: RotatingFileHandler, 100 MB/fichier) |

### Résumé des risques identifiés
- 🔴 **Critique** : Aucune traçabilité des prédictions (conformité, audit, debug)
- 🟠 **Élevé** : Pas de monitoring de performance (latence, disponibilité)
- 🟡 **Moyen** : Impossible de détecter le drift de données/modèle sans logs structurés

---

## Champs à logger (définition finale)

```json
{
  "timestamp": "2026-02-20T01:23:45.678Z",          // UTC, pour analyse temporelle
  "input_raw": "{\"SK_ID_CURR\": 100001, ...}",    // JSON brut reçu (pour debug + replay)
  "input_features": { "SK_ID_CURR": 100001, ... }, // dict parsé (pour drift Evidently)
  "output_proba": 0.3721,                          // proba défaut (clé pour drift outputs)
  "output_decision": "Accordé",                    // ou "Refusé" (seuil 0.4)
  "execution_time_ms": 142,                        // latence totale (anomalies ops)
  "error": null,                                   // str ou null (taux d'erreur)
  "model_version": "models:/LightGBM/Production",  // pour tracking version
  "threshold": 0.4                                 // pour reproductibilité
}
```

### Justification des champs

| **Champ** | **Utilité monitoring** | **Utilité métier** |
|-----------|------------------------|-------------------|
| `timestamp` | Détection pics de charge, analyses temporelles | Audit réglementaire (RGPD : droit d'accès aux décisions automatisées) |
| `input_raw` | Debug erreurs de parsing, rejeu de prédictions | Conformité (preuve de la requête client originale) |
| `input_features` | Drift detection avec Evidently (distribution des features) | Analyse des profils clients refusés |
| `output_proba` | Drift du modèle (distribution des scores), calibration | KPI métier (taux de refus, seuil optimal) |
| `output_decision` | Taux d'acceptation/refus, A/B testing seuils | Reporting direction (volume d'octrois) |
| `execution_time_ms` | SLA (P95, P99), alertes si latence > 500ms | Impact UX (timeout frontend) |
| `error` | Taux d'erreur (alerte si > 1%), debug par type | Fiabilité service (disponibilité) |
| `model_version` | Tracking de performance par version (rollback si dégradation) | Traçabilité réglementaire |
| `threshold` | Reproductibilité des décisions, historisation des changements | Audit métier (justification des refus) |

---

## Prochaines étapes

### ✅ Sous-étape 2 terminée (20-02-2026)
- [x] Fonction `log_prediction()` implémentée dans `app.py`
- [x] `_predict()` instrumentée avec `time.perf_counter()`
- [x] Logging structuré JSON (9 champs) vers `logs/predictions.jsonl`
- [x] Gestion des erreurs robuste (try/except avec logging)

### ✅ Sous-étape 3 terminée (20-02-2026)
- [x] Dossier `logs/` créé automatiquement avec `pathlib.Path`
- [x] `.gitignore` mis à jour (`logs/`, `*.jsonl`)
- [x] `Dockerfile` avec `VOLUME ["/app/logs"]`
- [x] `uv.lock` régénéré pour Python 3.11
- [x] **Docker build réussi** + container testé sur port 7860

### 🔄 Intégration future (Étape 4)
1. **Tester avec prédictions réelles** :
   - Envoyer JSON test via Gradio → vérifier `logs/predictions.jsonl`
   - Valider format JSON Lines (lecture avec `pd.read_json(..., lines=True)`)

2. **Monitoring en production** :
   - Utiliser logs pour générer rapports Evidently (drift détection)
   - Créer dashboard Grafana (latence, taux d'erreur, volume)
   - Configurer alertes (latence > 500ms, erreur > 1%, drift détecté)

3. **Optimisations** :
   - Ajouter rotation des logs (100 MB max par fichier)
   - Intégrer ELK Stack ou Loki pour centraliser les logs
   - Automatiser l'export vers S3/GCS pour archivage long terme
