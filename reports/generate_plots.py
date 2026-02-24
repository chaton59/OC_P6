#!/usr/bin/env python3
"""
Script de génération des plots de monitoring pour la présentation finale.
Analyse opérationnelle : latence et distribution de proba.
"""

import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# EXPLICATION : Chemin du fichier de logs structuré (JSON Lines)
LOGS_PATH = Path('/home/valentin/Env_Python/OC_P6/logs/predictions.jsonl')
OUTPUT_DIR = Path('/home/valentin/Env_Python/OC_P6/reports/plots')

def load_logs(file_path):
    """
    EXPLICATION : Charge les logs JSONL et retourne un DataFrame pandas
    Chaque ligne JSON = 1 appel API en production
    """
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)

def create_latency_plot(df):
    """
    EXPLICATION : Crée un histogramme de latence d'exécution
    execution_time_ms = temps de réponse du modèle en ms
    Utilise Plotly pour une visualisation interactive
    """
    fig = go.Figure(data=[
        go.Histogram(
            x=df['execution_time_ms'],
            nbinsx=30,
            name='Latence (ms)',
            marker_color='#1f77b4',
            hovertemplate='<b>Latence</b><br>%{x:.1f} ms<br>Fréquence: %{y}<extra></extra>'
        )
    ])
    
    # EXPLICATION : Configuration du layout pour lisibilité
    fig.update_layout(
        title={
            'text': '<b>Analyse Opérationnelle - Distribution Latence</b><br><sub>PoC: 200+ appels API simulés</sub>',
            'font': {'size': 18}
        },
        xaxis_title='Latence d\'exécution (ms)',
        yaxis_title='Nombre d\'appels',
        template='plotly_white',
        hovermode='x unified',
        showlegend=False,
        height=500,
        font=dict(size=12)
    )
    
    # EXPLICATION : Ajoute une ligne verticale pour la moyenne
    mean_latency = df['execution_time_ms'].mean()
    fig.add_vline(
        x=mean_latency,
        line_dash='dash',
        line_color='red',
        annotation_text=f'Moyenne: {mean_latency:.1f} ms',
        annotation_position='top right',
        annotation_font_size=12
    )
    
    return fig

def create_proba_distribution_plot(df):
    """
    EXPLICATION : Crée un histogramme de distribution des probabilités
    output_proba = score de risque du modèle (0.0 = crédit sain, 1.0 = défaut)
    Montre la proportion de clients par niveau de risque
    """
    fig = go.Figure(data=[
        go.Histogram(
            x=df['output_proba'],
            nbinsx=40,
            name='Distribution Proba',
            marker_color='#2ca02c',
            hovertemplate='<b>Probability</b><br>%{x:.3f}<br>Count: %{y}<extra></extra>'
        )
    ])
    
    # EXPLICATION : Configuration du layout avec annotations
    fig.update_layout(
        title={
            'text': '<b>Distribution des Probabilités de Défaut</b><br><sub>Scores du modèle LightGBM</sub>',
            'font': {'size': 18}
        },
        xaxis_title='Score de probabilité (0=Sain, 1=Défaut)',
        yaxis_title='Nombre de clients',
        template='plotly_white',
        hovermode='x unified',
        showlegend=False,
        height=500,
        font=dict(size=12),
        xaxis=dict(range=[0, 1])
    )
    
    # EXPLICATION : Ajoute une ligne verticale au seuil de décision
    threshold = df['input_features'].apply(
        lambda x: x.get('threshold', 0.4) if isinstance(x, dict) else 0.4
    ).iloc[0]
    
    fig.add_vline(
        x=threshold,
        line_dash='dash',
        line_color='orange',
        annotation_text=f'Seuil: {threshold}',
        annotation_position='top left',
        annotation_font_size=12
    )
    
    return fig

def main():
    """EXPLICATION : Fonction principale - charge données et exporte plots"""
    print("📊 Génération des plots de monitoring...")
    
    # EXPLICATION : Charge les logs en mémoire
    df = load_logs(LOGS_PATH)
    print(f"✓ Chargés {len(df)} appels API")
    
    # EXPLICATION : Crée les figures Plotly
    fig_latency = create_latency_plot(df)
    fig_proba = create_proba_distribution_plot(df)
    
    # EXPLICATION : Exporte en HTML et PNG (PNG nécessite kaleido)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plots latence
    latency_html = output_dir / 'latence_histogram.html'
    latency_png = output_dir / 'latence_histogram.png'
    fig_latency.write_html(str(latency_html))
    print(f"✓ Latence HTML: {latency_html}")
    
    try:
        fig_latency.write_image(str(latency_png), width=1000, height=500)
        print(f"✓ Latence PNG: {latency_png}")
    except Exception as e:
        print(f"⚠ PNG export échoué (kaleido?): {e}")
    
    # Plots proba distribution
    proba_html = output_dir / 'proba_histogram.html'
    proba_png = output_dir / 'proba_histogram.png'
    fig_proba.write_html(str(proba_html))
    print(f"✓ Proba HTML: {proba_html}")
    
    try:
        fig_proba.write_image(str(proba_png), width=1000, height=500)
        print(f"✓ Proba PNG: {proba_png}")
    except Exception as e:
        print(f"⚠ PNG export échoué (kaleido?): {e}")
    
    print("\n✅ Plots générés avec succès!")

if __name__ == '__main__':
    main()
