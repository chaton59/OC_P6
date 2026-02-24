#!/usr/bin/env python3
"""
Génération des plots PNG statiques (fallback si kaleido non disponible)
Utilise matplotlib pour une version simple et universelle
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# EXPLICATION : Chemins de fichiers
LOGS_PATH = Path('/home/valentin/Env_Python/OC_P6/logs/predictions.jsonl')
OUTPUT_DIR = Path('/home/valentin/Env_Python/OC_P6/reports/plots')

def load_logs(file_path):
    """EXPLICATION : Charge les logs JSONL"""
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)

def create_latency_png(df):
    """
    EXPLICATION : Crée un PNG de latence avec matplotlib
    Style pro pour présentation
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    # EXPLICATION : Histogramme avec couleur bleu ciel
    n, bins, patches = ax.hist(df['execution_time_ms'], bins=30, 
                                color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # EXPLICATION : Ligne verticale pour la moyenne
    mean_latency = df['execution_time_ms'].mean()
    ax.axvline(mean_latency, color='red', linestyle='--', linewidth=2, label=f'Moyenne: {mean_latency:.1f} ms')
    
    # EXPLICATION : Annotations et labels
    ax.set_xlabel('Latence d\'exécution (ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Nombre d\'appels', fontsize=13, fontweight='bold')
    ax.set_title('📊 Analyse Opérationnelle - Distribution Latence\nPoC: 200+ appels API simulés', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right')
    
    # EXPLICATION : Stats en bas à droite
    stats_text = f'N = {len(df)}\nMin = {df["execution_time_ms"].min():.1f} ms\nMax = {df["execution_time_ms"].max():.1f} ms'
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'latence_histogram.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ PNG Latence sauvegardé: {output_path}")
    plt.close()

def create_proba_png(df):
    """
    EXPLICATION : Crée un PNG de distribution des probabilités
    Montre le seuil de décision
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    # EXPLICATION : Histogramme avec couleur vert
    n, bins, patches = ax.hist(df['output_proba'], bins=40, 
                                color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # EXPLICATION : Seuil de décision (0.4)
    threshold = 0.4
    ax.axvline(threshold, color='orange', linestyle='--', linewidth=2.5, label=f'Seuil: {threshold}')
    
    # EXPLICATION : Zones de décision
    ax.axvspan(0, threshold, alpha=0.1, color='green', label='Zone Approuvé')
    ax.axvspan(threshold, 1, alpha=0.1, color='red', label='Zone Refusé')
    
    # EXPLICATION : Annotations et labels
    ax.set_xlabel('Score de probabilité (0=Sain, 1=Défaut)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Nombre de clients', fontsize=13, fontweight='bold')
    ax.set_title('📈 Distribution des Probabilités de Défaut\nScores du modèle LightGBM', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right')
    
    # EXPLICATION : Stats en bas à gauche
    below_threshold = (df['output_proba'] < threshold).sum()
    above_threshold = (df['output_proba'] >= threshold).sum()
    stats_text = f'Approuvé: {below_threshold} ({100*below_threshold/len(df):.1f}%)\nRefusé: {above_threshold} ({100*above_threshold/len(df):.1f}%)'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'proba_histogram.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"✓ PNG Proba sauvegardé: {output_path}")
    plt.close()

def main():
    """EXPLICATION : Génère les PNG statiques"""
    print("📸 Génération des plots PNG statiques...")
    df = load_logs(LOGS_PATH)
    print(f"✓ Chargés {len(df)} appels API")
    
    create_latency_png(df)
    create_proba_png(df)
    
    print("\n✅ PLOTs PNG générés avec succès!")

if __name__ == '__main__':
    main()
