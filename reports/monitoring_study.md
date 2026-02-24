# Étape 3 - Étude Monitoring Production & Data Drift (PoC local terminé)

## Résumé exécutif
- Logging structuré JSON implémenté dans l'API
- 200+ appels simulés en production
- Analyse ops : latence, taux erreur
- Détection data drift avec Evidently → 25 features impactées (dont AMT_INCOME_TOTAL)

## Visualisations clés

### 1. Analyse opérationnelle (Plots générés automatiquement)

**Latence d'exécution** - Distribution du temps de réponse de l'API
- [HTML interactif](plots/latence_histogram.html) | [Image PNG](plots/latence_histogram.png)

**Distribution des Probabilités de Défaut** - Scores produits par le modèle
- [HTML interactif](plots/proba_histogram.html) | [Image PNG](plots/proba_histogram.png)

### 2. Rapport Data Drift (Evidently)
- Drift détecté sur 3.52% des features (25/711)
- Feature critique : **AMT_INCOME_TOTAL** (Drift Score = 0.304 – modéré)
- Lien : [data_drift_report.html](data_drift_report.html)

## Conclusions sur la dérive des données
- Le drift simulé sur AMT_INCOME_TOTAL a été parfaitement détecté.
- Impact métier : risque de mauvaises décisions crédit si non surveillé.
- Recommandation : relancer l'analyse toutes les 24h en production.

## Points de vigilance (conforme au brief)
- **Stockage** : fichier JSONL léger (< 1 Mo pour 10k appels). Rotation à ajouter si > 100 Mo.
- **RGPD** : logs contiennent données clients → anonymisation ou suppression après 30 jours.
- **Coût** : PoC local = 0 €. En cloud : utiliser S3 + Athena ou PostgreSQL.
- **Alertes futures** : seuil drift > 0.25 → notification Slack/Email.
- **Amélioration** : passer à NannyML ou intégration MLflow pour prediction drift.

## Comment reproduire
1. `uv run gradio app.py`
2. `uv run python simulate_production_calls.py`
3. Ouvrir `notebooks/07_detect_data_drift.ipynb` → Run All
4. Ouvrir `reports/monitoring_study.md`

---
