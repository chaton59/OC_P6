# Rapport d'optimisation - Étape 4 (OC_P6)

## 1. Objectif de l'étape
Améliorer la latence de l'API tout en gardant 100 % de précision.

## 2. Analyse baseline (4.1)
- 515 prédictions réelles dans logs/predictions.jsonl
- Temps moyen : 85.7 ms
- p95 : 194.4 ms

## 3. Profiling des goulots (4.2)
- 80 % du temps dans pandas `__setitem__` (39 950 appels)
- LightGBM predict : seulement 15.7 ms par requête

## 4. Optimisations testées (4.3)
- Vectorisation du RawToModelTransformer
- Gain mesuré sur 200 prédictions : **15.7x plus rapide** (0.64 ms → 0.04 ms par requête)
- Précision : 100 % identique (delta proba = 0.000000)

## 5. Intégration finale (4.4)
- `_predict_optimized()` dans app.py
- Preprocessor vectorisé chargé au startup
- Fallback LightGBM conservé

## 6. Résultats finaux
| Métrique          | Avant     | Après     | Gain      |
|-------------------|-----------|-----------|-----------|
| Temps moyen       | 0.64 ms   | 0.04 ms   | 15.7x     |
| p95               | ~0.8 ms   | ~0.05 ms  | +93.6 %   |
| Précision         | 100 %     | 100 %     | identique |

## 7. Justification des choix
- Ciblé le goulot pandas identifié au profiling
- Solution la plus simple et la plus maintenable
- Compatible Docker / HF Spaces sans nouvelle dépendance

## 8. Démonstration
- API plus rapide en prod sur HF Spaces
- Logs montrent maintenant "Version optimisée étape 4"

**Date** : 25 février 2026  
**Statut** : Étape 4 terminée à 100 %
