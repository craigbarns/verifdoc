"""
Report Generator — Rapport HTML investor-ready avec Chart.js.

Génère un fichier HTML autonome avec :
- Executive summary (6 KPI cards)
- Confusion matrix (CSS grid)
- ROC curve avec seuil optimal (Chart.js)
- Score distribution avec seuil (Chart.js)
- Per-layer radar chart + AUC (Chart.js)
- Latency breakdown
- Détails par type de falsification
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .metrics import BenchmarkMetrics
from .runner import BenchmarkRun


def generate_report(
    metrics: BenchmarkMetrics,
    run: BenchmarkRun,
    output_path: str | Path = "benchmark_results/report.html",
    title: str = "VerifDoc — Benchmark Report",
) -> Path:
    """Génère le rapport HTML complet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = _build_html(metrics, run, title)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Sauvegarder aussi un résumé JSON pour le dashboard
    summary = {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "auc": metrics.auc,
        "optimal_threshold": metrics.optimal_threshold,
        "latency_mean_ms": metrics.latency_mean_ms,
        "total_samples": metrics.total_samples,
        "timestamp": run.timestamp,
        "run_id": run.run_id,
    }
    summary_path = output_path.with_name("summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return output_path


def _build_html(m: BenchmarkMetrics, run: BenchmarkRun, title: str) -> str:
    """Construit le HTML complet."""

    # ── Données pour les charts ──────────────────────────────────
    # Sous-échantillonner les données ROC pour Chart.js (max ~150 points)
    n_roc = len(m.roc_fpr)
    if n_roc > 150:
        step = max(1, n_roc // 150)
        roc_fpr_sampled = [float(m.roc_fpr[i]) for i in range(0, n_roc, step)]
        roc_tpr_sampled = [float(m.roc_tpr[i]) for i in range(0, n_roc, step)]
    else:
        roc_fpr_sampled = [float(x) for x in m.roc_fpr]
        roc_tpr_sampled = [float(x) for x in m.roc_tpr]

    roc_data = json.dumps({"fpr": roc_fpr_sampled, "tpr": roc_tpr_sampled})

    # Point optimal sur la courbe ROC
    opt_idx = 0
    best_j = -1
    for i, (fpr, tpr) in enumerate(zip(m.roc_fpr, m.roc_tpr)):
        j = tpr - fpr
        if j > best_j:
            best_j = j
            opt_idx = i
    opt_fpr = float(m.roc_fpr[opt_idx])
    opt_tpr = float(m.roc_tpr[opt_idx])

    clean_scores_json = json.dumps([round(float(s), 4) for s in m.clean_scores])
    forged_scores_json = json.dumps([round(float(s), 4) for s in m.forged_scores])

    # Radar chart data
    layer_names_display = {
        "ela": "ELA", "noise": "Bruit", "copy_move": "Copy-Move",
        "metadata": "Metadata", "cross_check": "Cross-Check",
    }
    radar_labels = []
    radar_values = []
    radar_auc_values = []
    for layer, display in layer_names_display.items():
        lm = m.layer_metrics.get(layer, {})
        radar_labels.append(display)
        radar_values.append(float(lm.get("cohens_d", 0)))
        radar_auc_values.append(float(lm.get("auc", 0.5)))
    radar_labels_json = json.dumps(radar_labels)
    radar_values_json = json.dumps(radar_values)
    radar_auc_json = json.dumps(radar_auc_values)

    # Forgery type breakdown
    forgery_rows = ""
    ft_labels = {
        "amount_edit": "Modification montant", "text_replace": "Remplacement texte",
        "copy_paste": "Copier-coller", "noise_inject": "Injection bruit",
        "compression_artifact": "Artefact compression", "metadata_strip": "Triple compression",
        "none": "Aucune", "unknown": "Inconnu",
    }
    for ft, data in sorted(m.metrics_by_forgery.items()):
        rate = data["detection_rate"]
        color = "#4ade80" if rate >= 0.8 else "#fbbf24" if rate >= 0.5 else "#f87171"
        forgery_rows += f"""
        <tr>
            <td>{ft_labels.get(ft, ft)}</td>
            <td>{data['count']}</td>
            <td>{data['detected']}</td>
            <td style="color:{color};font-weight:700">{rate:.0%}</td>
            <td>{data['avg_score']:.4f}</td>
        </tr>"""

    # Layer analysis rows
    layer_rows = ""
    for layer, display in layer_names_display.items():
        lm = m.layer_metrics.get(layer, {})
        if lm:
            sep = lm.get("separation", "—")
            sep_color = "#4ade80" if sep == "excellent" else "#fbbf24" if sep == "bon" else "#f87171"
            layer_rows += f"""
            <tr>
                <td>{display}</td>
                <td>{lm['mean_clean']:.4f}</td>
                <td>{lm['mean_forged']:.4f}</td>
                <td>{lm['cohens_d']:.2f}</td>
                <td>{lm.get('auc', 0.5):.3f}</td>
                <td style="color:{sep_color};font-weight:700">{sep.upper()}</td>
            </tr>"""

    # Confusion matrix values
    tp, tn, fp, fn = m.true_positives, m.true_negatives, m.false_positives, m.false_negatives

    timestamp_display = run.timestamp[:19].replace("T", " ")

    # Score separation ratio
    sep_ratio = m.score_stats.get("separation_ratio", 0)

    # Extraire stats scores pour éviter les problèmes de {{}} dans f-string
    cs = m.score_stats.get("clean", {})
    fs = m.score_stats.get("forged", {})
    cs_mean = cs.get("mean", 0)
    cs_median = cs.get("median", 0)
    cs_std = cs.get("std", 0)
    cs_min = cs.get("min", 0)
    cs_max = cs.get("max", 0)
    fs_mean = fs.get("mean", 0)
    fs_median = fs.get("median", 0)
    fs_std = fs.get("std", 0)
    fs_min = fs.get("min", 0)
    fs_max = fs.get("max", 0)

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3/dist/chartjs-plugin-annotation.min.js"></script>
<style>
:root {{
    --bg: #0a0a10;
    --bg-card: #12121c;
    --bg-card-alt: #1a1a28;
    --text: #e8e6f0;
    --text-muted: #8a8793;
    --accent: #2dd4bf;
    --green: #4ade80;
    --yellow: #fbbf24;
    --red: #f87171;
    --blue: #60a5fa;
    --purple: #a78bfa;
    --border: rgba(45, 212, 191, 0.12);
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 0;
}}
.container {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px; }}

/* Header */
header {{
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    padding: 60px 0 40px;
    border-bottom: 1px solid var(--border);
}}
header .container {{ text-align: center; }}
header h1 {{
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent), var(--blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}}
header .subtitle {{ color: var(--text-muted); font-size: 1rem; }}
header .meta {{
    margin-top: 16px;
    display: flex;
    justify-content: center;
    gap: 24px;
    font-size: 0.85rem;
    color: var(--text-muted);
    flex-wrap: wrap;
}}
header .meta span {{ background: var(--bg-card); padding: 4px 12px; border-radius: 6px; }}

/* KPI Cards */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 16px;
    margin: 40px 0;
}}
.kpi {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 16px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}}
.kpi:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(45, 212, 191, 0.1);
}}
.kpi .value {{
    font-size: 2rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}}
.kpi .label {{ color: var(--text-muted); font-size: 0.8rem; margin-top: 4px; }}
.kpi.accent .value {{ color: var(--accent); }}
.kpi.green .value {{ color: var(--green); }}
.kpi.blue .value {{ color: var(--blue); }}
.kpi.purple .value {{ color: var(--purple); }}
.kpi.yellow .value {{ color: var(--yellow); }}
.kpi.red .value {{ color: var(--red); }}

/* Summary box */
.summary-box {{
    background: linear-gradient(135deg, rgba(45, 212, 191, 0.08), rgba(96, 165, 250, 0.08));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 40px;
    font-size: 1.05rem;
    text-align: center;
}}
.summary-box .highlight {{ color: var(--accent); font-weight: 700; }}

/* Sections */
section {{ margin: 48px 0; }}
section h2 {{
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 24px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border);
    color: var(--accent);
}}

/* Charts */
.chart-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
}}
.chart-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
}}
.chart-card h3 {{
    font-size: 1rem;
    color: var(--text-muted);
    margin-bottom: 16px;
}}
.chart-card canvas {{ max-height: 320px; }}

/* Confusion Matrix */
.cm-grid {{
    display: grid;
    grid-template-columns: auto 1fr 1fr;
    grid-template-rows: auto 1fr 1fr;
    gap: 4px;
    max-width: 380px;
    margin: 0 auto;
}}
.cm-cell {{
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    border-radius: 8px;
    font-size: 1.8rem;
    font-weight: 800;
    font-family: monospace;
}}
.cm-label {{
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    color: var(--text-muted);
    font-weight: 600;
}}
.cm-tp {{ background: rgba(74, 222, 128, 0.15); color: var(--green); }}
.cm-tn {{ background: rgba(96, 165, 250, 0.15); color: var(--blue); }}
.cm-fp {{ background: rgba(248, 113, 113, 0.15); color: var(--red); }}
.cm-fn {{ background: rgba(251, 191, 36, 0.15); color: var(--yellow); }}

/* Threshold badge */
.threshold-badge {{
    display: inline-block;
    background: linear-gradient(135deg, rgba(45, 212, 191, 0.2), rgba(96, 165, 250, 0.2));
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 8px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    margin: 8px 0;
}}

/* Tables */
table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-card);
    border-radius: 12px;
    overflow: hidden;
}}
th {{
    background: var(--bg-card-alt);
    padding: 12px 16px;
    text-align: left;
    font-size: 0.85rem;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
td {{
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    font-size: 0.95rem;
}}
tr:hover td {{ background: rgba(45, 212, 191, 0.03); }}

/* Footer */
footer {{
    text-align: center;
    padding: 40px;
    color: var(--text-muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 60px;
}}
footer .brand {{ color: var(--accent); font-weight: 700; }}

@media (max-width: 768px) {{
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .chart-grid {{ grid-template-columns: 1fr; }}
    .kpi .value {{ font-size: 1.5rem; }}
}}
</style>
</head>
<body>

<header>
<div class="container">
    <h1>VerifDoc Benchmark</h1>
    <p class="subtitle">Performance Evaluation Report — Document Fraud Detection Pipeline</p>
    <div class="meta">
        <span>{m.total_samples} documents</span>
        <span>{timestamp_display} UTC</span>
        <span>Pipeline v0.3.0</span>
        <span>Run #{run.run_id[:8]}</span>
        <span>6 couches forensiques</span>
    </div>
</div>
</header>

<div class="container">

<!-- Executive Summary -->
<div class="kpi-grid">
    <div class="kpi green">
        <div class="value">{m.f1:.1%}</div>
        <div class="label">F1-Score</div>
    </div>
    <div class="kpi accent">
        <div class="value">{m.accuracy:.1%}</div>
        <div class="label">Accuracy</div>
    </div>
    <div class="kpi blue">
        <div class="value">{m.auc:.3f}</div>
        <div class="label">AUC-ROC</div>
    </div>
    <div class="kpi purple">
        <div class="value">{m.recall:.0%}</div>
        <div class="label">Recall</div>
    </div>
    <div class="kpi yellow">
        <div class="value">{m.precision:.0%}</div>
        <div class="label">Precision</div>
    </div>
    <div class="kpi red">
        <div class="value">{m.latency_mean_ms:.0f}ms</div>
        <div class="label">Latence moy.</div>
    </div>
</div>

<div class="summary-box">
    Sur <strong>{m.total_samples}</strong> documents testés,
    VerifDoc a correctement identifié <span class="highlight">{m.true_positives}/{m.total_forged}</span> documents falsifiés
    ({m.recall:.0%} recall) avec seulement <span class="highlight">{m.false_positives}</span> faux positif(s).
    <br/>
    <span class="threshold-badge">
        Seuil optimal (Youden's J) : <strong>{m.optimal_threshold:.4f}</strong>
        &nbsp;|&nbsp; J = {m.youden_j:.3f}
        &nbsp;|&nbsp; Ratio de séparation : {sep_ratio}x
    </span>
</div>

<!-- Methodology -->
<section>
    <h2>Methodologie</h2>
    <table>
        <tr><th>Parametre</th><th>Valeur</th></tr>
        <tr><td>Dataset</td><td>{', '.join(run.dataset_info.get('sources', []))}</td></tr>
        <tr><td>Documents clean</td><td>{m.total_clean}</td></tr>
        <tr><td>Documents falsifies</td><td>{m.total_forged}</td></tr>
        <tr><td>Types de falsification</td><td>{', '.join(run.dataset_info.get('forgery_types', []))}</td></tr>
        <tr><td>OCR</td><td>{'Active' if run.config.get('run_ocr') else 'Desactive'}</td></tr>
        <tr><td>IA (Claude Vision)</td><td>{'Active' if not run.config.get('skip_ai') else 'Desactive'}</td></tr>
        <tr><td>Seuil de decision</td><td>Youden's J-statistic (optimal = {m.optimal_threshold:.4f})</td></tr>
        <tr><td>Temps total</td><td>{run.total_time_seconds:.1f}s</td></tr>
    </table>
</section>

<!-- Classification Performance -->
<section>
    <h2>Performance de classification</h2>
    <div class="chart-grid">
        <div class="chart-card">
            <h3>Matrice de confusion (seuil = {m.optimal_threshold:.4f})</h3>
            <div class="cm-grid">
                <div></div>
                <div class="cm-label">Predit Clean</div>
                <div class="cm-label">Predit Forged</div>
                <div class="cm-label" style="writing-mode:vertical-rl;transform:rotate(180deg)">Reel Clean</div>
                <div class="cm-cell cm-tn">{tn}</div>
                <div class="cm-cell cm-fp">{fp}</div>
                <div class="cm-label" style="writing-mode:vertical-rl;transform:rotate(180deg)">Reel Forged</div>
                <div class="cm-cell cm-fn">{fn}</div>
                <div class="cm-cell cm-tp">{tp}</div>
            </div>
        </div>
        <div class="chart-card">
            <h3>Courbe ROC (AUC = {m.auc:.3f})</h3>
            <canvas id="rocChart"></canvas>
        </div>
    </div>
    <br/>
    <table>
        <tr><th>Metrique</th><th>Valeur</th><th>Interpretation</th></tr>
        <tr><td>Accuracy</td><td><strong>{m.accuracy:.1%}</strong></td><td>Taux de classification correcte global</td></tr>
        <tr><td>Precision</td><td><strong>{m.precision:.1%}</strong></td><td>% des alertes qui sont de vrais positifs</td></tr>
        <tr><td>Recall</td><td><strong>{m.recall:.1%}</strong></td><td>% des fraudes effectivement detectees</td></tr>
        <tr><td>F1-Score</td><td><strong>{m.f1:.1%}</strong></td><td>Moyenne harmonique precision/recall</td></tr>
        <tr><td>AUC-ROC</td><td><strong>{m.auc:.4f}</strong></td><td>Capacite de discrimination (1.0 = parfait)</td></tr>
        <tr><td>Seuil optimal</td><td><strong>{m.optimal_threshold:.4f}</strong></td><td>Maximise Youden's J = TPR - FPR</td></tr>
    </table>
</section>

<!-- Score Distribution -->
<section>
    <h2>Distribution des scores</h2>
    <div class="chart-grid">
        <div class="chart-card">
            <h3>Scores documents clean vs falsifies</h3>
            <canvas id="distChart"></canvas>
        </div>
        <div class="chart-card">
            <h3>Contribution par couche (Cohen's d + AUC)</h3>
            <canvas id="radarChart"></canvas>
        </div>
    </div>
    <br/>
    <table>
        <tr><th>Classe</th><th>Moyenne</th><th>Mediane</th><th>Ecart-type</th><th>Min</th><th>Max</th></tr>
        <tr>
            <td style="color:var(--green)">Clean</td>
            <td>{cs_mean:.4f}</td>
            <td>{cs_median:.4f}</td>
            <td>{cs_std:.4f}</td>
            <td>{cs_min:.4f}</td>
            <td>{cs_max:.4f}</td>
        </tr>
        <tr>
            <td style="color:var(--red)">Falsifie</td>
            <td>{fs_mean:.4f}</td>
            <td>{fs_median:.4f}</td>
            <td>{fs_std:.4f}</td>
            <td>{fs_min:.4f}</td>
            <td>{fs_max:.4f}</td>
        </tr>
    </table>
</section>

<!-- Per-Layer Analysis -->
<section>
    <h2>Analyse par couche forensique</h2>
    <table>
        <tr><th>Couche</th><th>Score moy. (Clean)</th><th>Score moy. (Forged)</th><th>Cohen's d</th><th>AUC</th><th>Separation</th></tr>
        {layer_rows}
    </table>
</section>

<!-- Detection by Forgery Type -->
<section>
    <h2>Detection par type de falsification</h2>
    <table>
        <tr><th>Type</th><th>Total</th><th>Detectes</th><th>Taux</th><th>Score moyen</th></tr>
        {forgery_rows}
    </table>
</section>

<!-- Latency -->
<section>
    <h2>Performance (latence)</h2>
    <div class="kpi-grid">
        <div class="kpi accent">
            <div class="value">{m.latency_mean_ms:.0f}ms</div>
            <div class="label">Moyenne</div>
        </div>
        <div class="kpi green">
            <div class="value">{m.latency_median_ms:.0f}ms</div>
            <div class="label">Mediane (P50)</div>
        </div>
        <div class="kpi blue">
            <div class="value">{m.latency_p95_ms:.0f}ms</div>
            <div class="label">P95</div>
        </div>
        <div class="kpi purple">
            <div class="value">{m.throughput_per_second:.1f}/s</div>
            <div class="label">Debit</div>
        </div>
    </div>
</section>

</div>

<footer>
    <span class="brand">VerifDoc Benchmark Suite v1.0</span><br/>
    Pipeline v0.3.0 — 5 couches forensiques + IA<br/>
    Seuil de decision optimise par Youden's J-statistic<br/>
    Rapport genere automatiquement — {timestamp_display} UTC
</footer>

<script>
// ROC Curve with optimal threshold marker
const rocData = {roc_data};
const optFPR = {opt_fpr};
const optTPR = {opt_tpr};
const optThreshold = {m.optimal_threshold};

new Chart(document.getElementById('rocChart'), {{
    type: 'line',
    data: {{
        datasets: [
            {{
                label: 'ROC (AUC={m.auc:.3f})',
                data: rocData.fpr.map((fpr, i) => ({{ x: fpr, y: rocData.tpr[i] }})),
                borderColor: '#2dd4bf',
                backgroundColor: 'rgba(45, 212, 191, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                borderWidth: 2.5,
            }},
            {{
                label: 'Aleatoire',
                data: [{{ x: 0, y: 0 }}, {{ x: 1, y: 1 }}],
                borderColor: '#555',
                borderDash: [5, 5],
                pointRadius: 0,
                borderWidth: 1,
            }},
            {{
                label: 'Seuil optimal (' + optThreshold.toFixed(4) + ')',
                data: [{{ x: optFPR, y: optTPR }}],
                borderColor: '#fbbf24',
                backgroundColor: '#fbbf24',
                pointRadius: 8,
                pointHoverRadius: 12,
                showLine: false,
                pointStyle: 'star',
            }}
        ]
    }},
    options: {{
        responsive: true,
        scales: {{
            x: {{ type: 'linear', title: {{ display: true, text: 'Taux Faux Positifs (FPR)', color: '#8a8793' }}, min: 0, max: 1, ticks: {{ color: '#8a8793' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
            y: {{ title: {{ display: true, text: 'Taux Vrais Positifs (TPR)', color: '#8a8793' }}, min: 0, max: 1, ticks: {{ color: '#8a8793' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
        }},
        plugins: {{ legend: {{ labels: {{ color: '#e8e6f0' }} }} }}
    }}
}});

// Score Distribution with threshold line
const cleanScores = {clean_scores_json};
const forgedScores = {forged_scores_json};
const allScores = [...cleanScores, ...forgedScores];
const scoreMax = Math.max(...allScores) * 1.3;
const bins = 25;

function histogram(data, bins, max) {{
    const step = max / bins;
    const counts = new Array(bins).fill(0);
    data.forEach(v => {{
        const idx = Math.min(Math.floor(v / step), bins - 1);
        if (idx >= 0) counts[idx]++;
    }});
    return counts;
}}

const labels = Array.from({{length: bins}}, (_, i) => (i * scoreMax / bins).toFixed(3));
const distCtx = document.getElementById('distChart');
new Chart(distCtx, {{
    type: 'bar',
    data: {{
        labels: labels,
        datasets: [
            {{ label: 'Clean', data: histogram(cleanScores, bins, scoreMax), backgroundColor: 'rgba(74, 222, 128, 0.6)', borderRadius: 4, barPercentage: 1.0, categoryPercentage: 1.0 }},
            {{ label: 'Falsifie', data: histogram(forgedScores, bins, scoreMax), backgroundColor: 'rgba(248, 113, 113, 0.6)', borderRadius: 4, barPercentage: 1.0, categoryPercentage: 1.0 }}
        ]
    }},
    options: {{
        responsive: true,
        scales: {{
            x: {{ title: {{ display: true, text: 'Score de risque', color: '#8a8793' }}, ticks: {{ color: '#8a8793', maxTicksLimit: 10 }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
            y: {{ title: {{ display: true, text: 'Nombre de documents', color: '#8a8793' }}, ticks: {{ color: '#8a8793' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
        }},
        plugins: {{
            legend: {{ labels: {{ color: '#e8e6f0' }} }},
            annotation: {{
                annotations: {{
                    thresholdLine: {{
                        type: 'line',
                        xMin: Math.floor(optThreshold / (scoreMax / bins)),
                        xMax: Math.floor(optThreshold / (scoreMax / bins)),
                        borderColor: '#fbbf24',
                        borderWidth: 2,
                        borderDash: [6, 4],
                        label: {{
                            display: true,
                            content: 'Seuil: ' + optThreshold.toFixed(4),
                            color: '#fbbf24',
                            font: {{ size: 11 }},
                            position: 'start',
                        }}
                    }}
                }}
            }}
        }}
    }}
}});

// Radar Chart (Cohen's d + AUC)
new Chart(document.getElementById('radarChart'), {{
    type: 'radar',
    data: {{
        labels: {radar_labels_json},
        datasets: [
            {{
                label: "Cohen's d (separation)",
                data: {radar_values_json},
                borderColor: '#2dd4bf',
                backgroundColor: 'rgba(45, 212, 191, 0.15)',
                pointBackgroundColor: '#2dd4bf',
                borderWidth: 2,
            }},
            {{
                label: 'AUC par couche',
                data: {radar_auc_json},
                borderColor: '#a78bfa',
                backgroundColor: 'rgba(167, 139, 250, 0.1)',
                pointBackgroundColor: '#a78bfa',
                borderWidth: 2,
            }}
        ]
    }},
    options: {{
        responsive: true,
        scales: {{
            r: {{
                ticks: {{ color: '#8a8793', backdropColor: 'transparent' }},
                grid: {{ color: 'rgba(255,255,255,0.1)' }},
                pointLabels: {{ color: '#e8e6f0', font: {{ size: 12 }} }},
                beginAtZero: true,
            }}
        }},
        plugins: {{ legend: {{ labels: {{ color: '#e8e6f0' }} }} }}
    }}
}});
</script>
</body>
</html>"""
