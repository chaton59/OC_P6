#!/usr/bin/env python3
"""
Script de sommaire du projet - Affiche un résumé pour la soutenance
"""

import os
from pathlib import Path

def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)

def print_tree(path, prefix="", max_depth=3, current_depth=0, ignore_dirs={'.venv', '__pycache__', '.git', 'mlruns', 'node_modules'}):
    """Affiche l'arborescence du projet"""
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        return
    
    # Filtrer les répertoires à ignorer
    items = [i for i in items if i not in ignore_dirs and not i.startswith('.')]
    
    for i, item in enumerate(items):
        item_path = os.path.join(path, item)
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item}")
        
        if os.path.isdir(item_path) and current_depth < max_depth - 1:
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(item_path, next_prefix, max_depth, current_depth + 1, ignore_dirs)

def main():
    root_path = Path('/home/valentin/Env_Python/OC_P6')
    
    print("\n" + "🎓 PROJET OC P6 - CREDIT SCORING - RÉSUMÉ SOUTENANCE".center(80))
    print("=" * 80)
    
    print_section("📝 ÉTAPE 3 - MONITORING PRODUCTION & DATA DRIFT")
    print("""
✅ OBJECTIFS ATTEINTS:
   1. Logging structuré JSON implémenté dans l'API (app.py)
   2. 500+ appels simulés en production (reference/simulate_production_calls.py)
   3. Plots de latence et distribution des scores (Plotly + Matplotlib)
   4. Détection data drift avec Evidently (25 features impactées)
   5. Rapport complet et recommandations MLOps

📊 LIVRABLES:
   ✓ reports/monitoring_study.md        - Rapport synthétique complet
   ✓ reports/data_drift_report.html     - Dashboard Evidently (Drift detection)
   ✓ reports/plots/latence_histogram.*  - Analyse latence API (HTML + PNG)
   ✓ reports/plots/proba_histogram.*    - Distribution probabilités (HTML + PNG)
   ✓ README.md (Section 3)              - Documentation mise à jour
    """)
    
    print_section("📁 STRUCTURE FINALE DU PROJET")
    print_tree(root_path, max_depth=3)
    
    print_section("🔍 MÉTRIQUES CLÉS DU PoC")
    print("""
   • Appels API simulés: 500+
   • Latence moyenne: ~200-300ms
   • Features avec drift détecté: 25/711 (3.52%)
   • Feature critique: AMT_INCOME_TOTAL (Drift Score = 0.304)
   • Seuil décision produit: 0.4
   • Stockage JSONL: < 1 Mo pour 10k appels
    """)
    
    print_section("✅ POINTS DE VIGILANCE (BRIEF CONFORME)")
    print("""
   ✓ Stockage: JSONL léger, rotation à ajouter si > 100 Mo
   ✓ RGPD: Anonymisation / suppression logs après 30j
   ✓ Coût: PoC local = 0€, cloud = S3 + Athena
   ✓ Alertes: Seuil drift > 0.25 → Slack/Email
   ✓ Améliorations: NannyML ou MLflow pour prediction drift
    """)
    
    print_section("🚀 COMMENT REPRODUIRE")
    print("""
   # 1. Lancer l'API
   $ uv run gradio app.py

   # 2. Simuler les appels production
   $ uv run python reference/simulate_production_calls.py

   # 3. Générer les plots
   $ cd reports && python generate_plots.py

   # 4. Analyser le drift
   $ jupyter notebook notebooks/07_detect_data_drift.ipynb
    """)
    
    print_section("📊 VISUALISATIONS DISPONIBLES")
    plots_dir = root_path / 'reports' / 'plots'
    if plots_dir.exists():
        print(f"\n   Fichiers générés dans {plots_dir}:")
        for f in sorted(plots_dir.glob('*')):
            size = f.stat().st_size / (1024*1024)  # MB
            print(f"   • {f.name:<40} ({size:.2f} MB)")
    
    print_section("✨ STATUS")
    print("""
   ✅ ÉTAPE 3 - MONITORING PRODUCTION & DATA DRIFT: TERMINÉE
   
   Local PoC: 100% ✅
   Prêt pour présentation: ✅
   Documentation complète: ✅
    """)
    
    print("\n" + "=" * 80)
    print("  Généré automatiquement pour la soutenance")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
