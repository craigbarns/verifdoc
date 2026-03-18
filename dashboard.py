"""
VerifDoc — Dashboard premium (Streamlit).
Pipeline forensique parallélisé + rapport exportable.

v2 : SVG animé, st.tabs(), KPIs st.metric(), alertes groupées,
     barre de contribution par couche, progress bar temps réel.
"""

from __future__ import annotations

import base64
import html as html_lib
import io
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VerifDoc — Forensic Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DOC_LABELS = {
    "auto": "Détection auto",
    "bulletin_paie": "Bulletin de paie",
    "avis_imposition": "Avis d'imposition",
    "facture": "Facture",
    "rib": "RIB",
    "releve_bancaire": "Relevé bancaire",
    "quittance_loyer": "Quittance de loyer",
}

LAYER_LABELS = {
    "ela": "ELA",
    "noise": "Bruit / texture",
    "copy_move": "Copy-move",
    "metadata": "Métadonnées",
    "cross_check": "Validation métier",
    "ai_analysis": "Intelligence IA",
}

LAYER_ICONS = {
    "ela": "🔬",
    "noise": "📊",
    "copy_move": "🔎",
    "metadata": "📋",
    "cross_check": "✅",
    "ai_analysis": "🤖",
}


def _pil_to_b64(img) -> str | None:
    if img is None:
        return None
    buf = io.BytesIO()
    if hasattr(img, "mode") and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_pil(b64: str | None):
    if not b64:
        return None
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _biz_card_class(status: str) -> str:
    if status in ("ok", "found"):
        return "ok"
    if status in ("invalid", "not_found"):
        return "bad"
    if status in ("unavailable", "unverifiable", "unknown", "warn"):
        return "warn"
    return "neutral"


def _verdict_color(verdict: str) -> str:
    return {"clean": "#4ade80", "suspect": "#fbbf24", "forged": "#f87171"}.get(verdict, "#8a8793")


def _svg_score_ring(score_pct: float, verdict: str) -> str:
    """Génère un anneau SVG animé pour le score."""
    color = _verdict_color(verdict)
    circumference = 2 * 3.14159 * 54  # r=54
    filled = circumference * score_pct / 100
    gap = circumference - filled

    return f"""
    <svg width="160" height="160" viewBox="0 0 120 120" style="display:block;margin:0 auto;">
      <defs>
        <linearGradient id="ring-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="{color}"/>
          <stop offset="100%" stop-color="{color}88"/>
        </linearGradient>
      </defs>
      <!-- Background ring -->
      <circle cx="60" cy="60" r="54" fill="none" stroke="#1a1a26" stroke-width="8"/>
      <!-- Score ring -->
      <circle cx="60" cy="60" r="54" fill="none"
              stroke="url(#ring-grad)" stroke-width="8"
              stroke-linecap="round"
              stroke-dasharray="{filled:.1f} {gap:.1f}"
              transform="rotate(-90 60 60)"
              style="animation: vd-ring-fill 1.2s ease-out forwards;">
        <animate attributeName="stroke-dasharray"
                 from="0 {circumference:.1f}"
                 to="{filled:.1f} {gap:.1f}"
                 dur="1.2s" fill="freeze" calcMode="spline"
                 keySplines="0.4 0 0.2 1"/>
      </circle>
      <!-- Score text -->
      <text x="60" y="56" text-anchor="middle" font-family="Outfit,sans-serif"
            font-weight="800" font-size="22" fill="#e8e6e3">{score_pct:.0f}%</text>
      <text x="60" y="72" text-anchor="middle" font-family="Outfit,sans-serif"
            font-size="8" fill="#8a8793" letter-spacing="0.1em">INDICE</text>
    </svg>"""


def _contribution_bar(layers: list[dict]) -> str:
    """Barre horizontale montrant la contribution de chaque couche au score final."""
    if not layers:
        return ""
    total = sum(l.get("score", 0) * l.get("weight", 0) for l in layers)
    if total == 0:
        return ""

    segments = ""
    for layer in layers:
        contribution = layer["score"] * layer["weight"]
        pct = (contribution / total * 100) if total > 0 else 0
        if pct < 2:
            continue
        color = _verdict_color(layer.get("verdict", ""))
        name = LAYER_LABELS.get(layer["layer"], layer["layer"])
        segments += f'<div style="width:{pct:.1f}%;background:{color};height:100%;border-radius:4px;" title="{name}: {pct:.0f}%"></div>'

    return f"""
    <div style="display:flex;gap:3px;height:10px;margin:0.75rem 0 0.25rem;border-radius:6px;overflow:hidden;">
      {segments}
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#8a8793;font-family:Outfit,sans-serif;">
      <span>Contribution au score</span>
      <span>{'  '.join(f'{LAYER_ICONS.get(l["layer"],"")} {LAYER_LABELS.get(l["layer"],l["layer"])}' for l in layers if l["score"]*l["weight"]/total*100 >= 2) if total > 0 else ''}</span>
    </div>"""


def _build_html_report(
    filename: str,
    final: dict,
    results: dict,
    elapsed: float,
    doc_type_sel: str,
) -> str:
    """Rapport HTML aligné wording produit."""
    layers_html = ""
    for layer in final.get("layers", []):
        layers_html += (
            f"<tr><td>{html_lib.escape(LAYER_LABELS.get(layer['layer'], layer['layer']))}</td>"
            f"<td>{html_lib.escape(str(layer.get('verdict', '')))}</td>"
            f"<td>{html_lib.escape(str(layer.get('detail', '')))}</td></tr>"
        )
    flags = results.get("cross_check", {}).get("flags", [])
    flags_html = "".join(
        f"<li><strong>{html_lib.escape(str(f.get('type', '')))}</strong> — "
        f"{html_lib.escape(str(f.get('detail', '')))}</li>"
        for f in flags
    )
    biz = final.get("business_verification") or {}
    biz_rows = ""
    for key, title in [("siret", "SIRET"), ("entreprise", "Entreprise"), ("iban", "IBAN"), ("tva", "TVA intracom")]:
        b = biz.get(key) or {}
        biz_rows += (
            f"<tr><td><strong>{title}</strong></td><td>{html_lib.escape(b.get('label', '—'))}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <title>Rapport VerifDoc — {html_lib.escape(filename)}</title>
  <style>
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 1rem;
      color: #111; line-height: 1.5; }}
    h1 {{ font-size: 1.4rem; border-bottom: 2px solid #0d9488; padding-bottom: 0.5rem; }}
    .exec {{ background: #f0fdfa; border: 1px solid #99f6e4; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
    .exec strong {{ color: #0f766e; }}
    .meta {{ color: #666; font-size: 0.9rem; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
    th {{ background: #f4f4f5; }}
    .disclaimer {{ font-size: 0.85rem; color: #666; margin-top: 2rem; border-top: 1px solid #eee; padding-top: 1rem; }}
  </style>
</head>
<body>
  <h1>Rapport d'analyse — VerifDoc</h1>
  <p class="meta">{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} · Fichier : <strong>{html_lib.escape(filename)}</strong></p>
  <div class="exec">
    <strong>Résumé exécutif</strong><br/>
    Type de document : {html_lib.escape(final.get("doc_type_label", "—"))}<br/>
    Niveau de risque : {html_lib.escape(final.get("risk_title", ""))} — {html_lib.escape(final.get("risk_next_step", ""))}<br/>
    Confiance d'analyse : {final.get("confidence_percent", "?")}% · Indice numérique : {final.get("score_100", 0)}%<br/>
    Synthèse : {html_lib.escape(final.get("executive_anomalies", ""))}
  </div>
  <p>{html_lib.escape(final.get("recommendation", ""))}</p>
  <p class="meta">Durée : {elapsed}s · Type sélectionné : {html_lib.escape(doc_type_sel)}</p>
  <h2>Couche métier & référentiels</h2>
  <table><tbody>{biz_rows}</tbody></table>
  <h2>Couches forensiques</h2>
  <table><thead><tr><th>Couche</th><th>Signal</th><th>Détail</th></tr></thead><tbody>{layers_html}</tbody></table>
  <h2>Alertes détaillées</h2>
  <ul>{flags_html or "<li>Aucune alerte listée</li>"}</ul>
  <p class="disclaimer">{html_lib.escape(final.get("disclaimer", ""))}</p>
</body>
</html>"""


# ── CSS global (typo + thème forensic) ────────────────────────────────────────
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Outfit:wght@300;500;600;700;800&display=swap" rel="stylesheet">
<style>
    :root {
      --bg: #08080a;
      --surface: #12121a;
      --surface2: #1a1a26;
      --text: #e8e6e3;
      --muted: #8a8793;
      --accent: #2dd4bf;
      --accent-dim: #115e59;
      --warn: #fbbf24;
      --danger: #f87171;
      --ok: #4ade80;
    }
    .stApp {
      background: var(--bg);
      background-image:
        radial-gradient(ellipse 120% 80% at 10% -20%, rgba(45, 212, 191, 0.12), transparent 50%),
        radial-gradient(ellipse 80% 50% at 100% 100%, rgba(99, 102, 241, 0.08), transparent 45%),
        url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    }
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
    h1, h2, h3 { font-family: 'Instrument Serif', Georgia, serif !important; color: var(--text) !important; font-weight: 400 !important; }
    p, span, label, div { font-family: 'Outfit', sans-serif; }
    .vd-hero {
      text-align: center;
      padding: 2.5rem 1rem 2rem;
      margin-bottom: 1rem;
    }
    .vd-hero .tag {
      display: inline-block;
      font-family: 'Outfit', sans-serif;
      font-size: 0.7rem;
      letter-spacing: 0.25em;
      text-transform: uppercase;
      color: var(--accent);
      border: 1px solid var(--accent-dim);
      padding: 0.35rem 0.9rem;
      border-radius: 999px;
      margin-bottom: 1rem;
    }
    .vd-hero h1 {
      font-size: clamp(2.5rem, 6vw, 3.8rem) !important;
      line-height: 1.05;
      margin: 0 0 0.75rem;
      background: linear-gradient(135deg, #fff 0%, var(--accent) 55%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .vd-hero .sub {
      font-family: 'Outfit', sans-serif;
      font-size: 1.05rem;
      color: var(--muted);
      max-width: 520px;
      margin: 0 auto;
      font-weight: 300;
    }
    .vd-stat-row {
      display: flex;
      justify-content: center;
      gap: 2.5rem;
      flex-wrap: wrap;
      margin: 2rem 0;
      font-family: 'Outfit', sans-serif;
    }
    .vd-stat { text-align: center; }
    .vd-stat b { font-size: 1.75rem; color: var(--accent); display: block; }
    .vd-stat span { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
    .vd-card {
      background: var(--surface);
      border: 1px solid rgba(45, 212, 191, 0.15);
      border-radius: 16px;
      padding: 1.5rem;
      margin: 0.5rem 0;
      transition: border-color 0.25s ease, transform 0.2s ease;
    }
    .vd-card:hover { border-color: rgba(45, 212, 191, 0.35); transform: translateY(-2px); }
    .vd-feature-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1rem;
      margin: 2rem 0;
    }
    .vd-feature h3 { font-size: 1.25rem !important; margin-top: 0; color: var(--accent) !important; }
    .vd-feature p { color: var(--muted); font-size: 0.92rem; margin: 0; }
    .vd-verdict-box {
      border-radius: 20px;
      padding: 2rem;
      text-align: center;
      margin: 1rem 0 1.5rem;
      position: relative;
      overflow: hidden;
    }
    .vd-verdict-box.clean {
      background: linear-gradient(145deg, rgba(74, 222, 128, 0.12), var(--surface2));
      border: 1px solid rgba(74, 222, 128, 0.35);
    }
    .vd-verdict-box.suspect {
      background: linear-gradient(145deg, rgba(251, 191, 36, 0.1), var(--surface2));
      border: 1px solid rgba(251, 191, 36, 0.35);
    }
    .vd-verdict-box.forged {
      background: linear-gradient(145deg, rgba(248, 113, 113, 0.12), var(--surface2));
      border: 1px solid rgba(248, 113, 113, 0.35);
    }
    .vd-layer-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 1rem;
      border-radius: 10px;
      font-size: 0.85rem;
      margin: 0.25rem;
      font-family: 'Outfit', sans-serif;
      background: var(--surface2);
      border: 1px solid rgba(255,255,255,0.06);
    }
    .vd-disclaimer {
      font-size: 0.8rem;
      color: var(--muted);
      border-left: 3px solid var(--accent-dim);
      padding: 0.75rem 1rem;
      margin: 1.5rem 0;
      background: rgba(0,0,0,0.25);
      border-radius: 0 8px 8px 0;
    }
    .vd-exec-summary {
      background: linear-gradient(180deg, rgba(45, 212, 191, 0.08), var(--surface));
      border: 1px solid rgba(45, 212, 191, 0.25);
      border-radius: 16px;
      padding: 1.25rem 1.5rem;
      margin: 1rem 0 1.5rem;
      font-family: 'Outfit', sans-serif;
      font-size: 0.95rem;
      color: var(--text);
      line-height: 1.65;
    }
    .vd-exec-summary .label {
      font-size: 0.65rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 0.75rem;
      display: block;
    }
    .vd-exec-summary strong { color: var(--text); font-weight: 600; }
    .vd-section-title {
      font-family: 'Instrument Serif', Georgia, serif !important;
      font-size: 1.35rem !important;
      color: var(--text) !important;
      margin: 1.75rem 0 0.75rem !important;
      border-bottom: 1px solid rgba(45, 212, 191, 0.15);
      padding-bottom: 0.35rem;
    }
    .vd-business-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 0.85rem;
      margin: 0.5rem 0 1.25rem;
    }
    .vd-biz-card {
      background: var(--surface2);
      border-radius: 12px;
      padding: 1rem 1.1rem;
      font-family: 'Outfit', sans-serif;
      border-left: 4px solid var(--muted);
      transition: transform 0.2s ease;
    }
    .vd-biz-card:hover { transform: translateY(-1px); }
    .vd-biz-card.ok { border-left-color: var(--ok); }
    .vd-biz-card.warn { border-left-color: var(--warn); }
    .vd-biz-card.bad { border-left-color: var(--danger); }
    .vd-biz-card.neutral { border-left-color: rgba(138, 135, 147, 0.5); }
    .vd-biz-card .biz-k {
      font-size: 0.65rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 0.35rem;
    }
    .vd-biz-card .biz-v { font-size: 0.95rem; color: var(--text); font-weight: 500; }
    .vd-biz-card .biz-d { font-size: 0.8rem; color: var(--muted); margin-top: 0.35rem; }
    .vd-history { font-size: 0.78rem; color: var(--muted); }
    .vd-history table { width: 100%; border-collapse: collapse; }
    .vd-history td, .vd-history th { padding: 0.35rem 0.2rem; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.06); }
    div[data-testid="stSidebar"] {
      background: var(--surface) !important;
      border-right: 1px solid rgba(45, 212, 191, 0.1);
    }
    div[data-testid="stSidebar"] .stMarkdown, div[data-testid="stSidebar"] label { color: var(--text) !important; }
    .stButton > button {
      font-family: 'Outfit', sans-serif !important;
      font-weight: 600 !important;
      border-radius: 12px !important;
      background: linear-gradient(135deg, #14b8a6, #0d9488) !important;
      color: #fff !important;
      border: none !important;
      padding: 0.65rem 1.5rem !important;
      transition: box-shadow 0.3s ease, transform 0.2s ease !important;
    }
    .stButton > button:hover { box-shadow: 0 0 24px rgba(45, 212, 191, 0.35); transform: translateY(-1px); }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    .vd-results-separator {
      margin: 2.25rem auto 0.5rem;
      max-width: 920px;
      border: 0;
      border-top: 1px solid rgba(45, 212, 191, 0.28);
    }
    .vd-results-header {
      text-align: center;
      font-family: 'Outfit', sans-serif;
      font-size: 0.72rem;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      color: #2dd4bf;
      margin: 0.5rem 0 1.25rem;
    }
    div[data-testid="column"] {
      vertical-align: top;
    }
    /* Alert cards */
    .vd-alert-card {
      background: var(--surface2);
      border-radius: 10px;
      padding: 0.75rem 1rem;
      margin: 0.4rem 0;
      font-family: 'Outfit', sans-serif;
      border-left: 3px solid var(--muted);
    }
    .vd-alert-card.high { border-left-color: var(--danger); }
    .vd-alert-card.medium { border-left-color: var(--warn); }
    .vd-alert-card.low, .vd-alert-card.info { border-left-color: var(--ok); }
    .vd-alert-type {
      font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.08em; margin-bottom: 0.2rem;
    }
    .vd-alert-detail { font-size: 0.88rem; color: var(--text); }
    /* KPI metric overrides */
    div[data-testid="stMetric"] {
      background: var(--surface2);
      border-radius: 12px;
      padding: 1rem;
      border: 1px solid rgba(45, 212, 191, 0.1);
    }
    div[data-testid="stMetric"] label { color: var(--muted) !important; font-size: 0.75rem !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: var(--text) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
if "vd_history" not in st.session_state:
    st.session_state.vd_history = []

with st.sidebar:
    st.markdown("### VerifDoc")
    st.caption("Intelligence documentaire · Confiance & clarté")

    doc_type = st.selectbox(
        "Type de document (indicatif)",
        list(DOC_LABELS.keys()),
        format_func=lambda x: DOC_LABELS.get(x, x),
    )

    run_ocr = st.checkbox(
        "Couche métier + référentiels (OCR, SIRET, IBAN…)",
        value=True,
        help="Recommandé pour la différenciation produit",
    )

    st.divider()
    st.markdown("**Pipeline AI-Powered**")
    st.caption("6 couches en parallèle — ELA multi-qualité, bruit multi-échelle, copy-move ORB adaptatif, métadonnées avancées, validation métier + Intelligence IA (Claude Vision)")
    st.markdown(
        '<span style="font-size:0.7rem;color:#2dd4bf;font-family:Outfit,sans-serif;">Powered by Claude AI</span>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Historique (session)**")
    if st.session_state.vd_history:
        rows = []
        for e in st.session_state.vd_history[:12]:
            exp = "✓" if e.get("export") else "—"
            rows.append(f"| {e.get('date', '')[:16]} | {e.get('fichier', '')[:18]}… | {e.get('risque', '')[:14]} | {e.get('score', '')} | {exp} |")
        st.markdown(
            "| Date | Fichier | Niveau | Score | Export |\n|------|---------|--------|-------|--------|\n"
            + "\n".join(rows),
        )
    else:
        st.caption("Les analyses apparaîtront ici.")

    st.divider()
    st.caption("Offres · Gratuit · Pro · Business + API")

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="vd-hero">
  <div class="tag">Niveau de risque documentaire</div>
  <h1>Clarté immédiate.<br/>Décision éclairée.</h1>
  <p class="sub">Intelligence artificielle, signaux forensiques et référentiels officiels — 6 couches d'analyse parallèles pour une confiance maximale.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("##### Votre fichier")
uploaded_file = st.file_uploader(
    "PDF ou image (JPG, PNG)",
    type=["pdf", "jpg", "jpeg", "png"],
    help="Max 20 Mo",
)

if uploaded_file is None:
    st.markdown(
        """
<div class="vd-stat-row">
  <div class="vd-stat"><b>6</b><span>couches d'analyse</span></div>
  <div class="vd-stat"><b>6</b><span>en parallèle</span></div>
  <div class="vd-stat"><b>IA</b><span>Claude Vision</span></div>
  <div class="vd-stat"><b>∞</b><span>formats PDF & image</span></div>
</div>
<div class="vd-feature-grid">
  <div class="vd-card vd-feature"><h3>◆ Intelligence IA</h3><p>Claude Vision analyse le document en profondeur — détection sémantique, cohérence des données, explication en langage naturel.</p></div>
  <div class="vd-card vd-feature"><h3>◆ Signaux forensiques</h3><p>ELA multi-qualité, texture multi-échelle et copy-move ORB adaptatif.</p></div>
  <div class="vd-card vd-feature"><h3>◆ Métadonnées + Métier</h3><p>SIRET, IBAN, TVA intracom, JavaScript, chiffrement — scoring adaptatif par type de document.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

# Streamlit : après le 1er read(), le buffer est en fin de fichier.
try:
    uploaded_file.seek(0)
except Exception:
    pass
file_bytes = uploaded_file.read()
_upload_key = f"{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"
if len(file_bytes) == 0:
    cached = st.session_state.get("vd_upload_bytes")
    if isinstance(cached, dict) and cached.get("key") == _upload_key:
        file_bytes = cached.get("data") or b""
elif len(file_bytes) > 0:
    st.session_state["vd_upload_bytes"] = {"key": _upload_key, "data": file_bytes}

if len(file_bytes) == 0:
    st.error(
        "Fichier vide ou non relu. **Sélectionnez à nouveau le fichier** puis lancez l'analyse."
    )
    st.stop()

file_ext = Path(uploaded_file.name).suffix.lower()

col_doc, col_results = st.columns([1.15, 0.85])

with col_doc:
    st.markdown("#### Document")
    preview_img = None
    if file_ext in [".jpg", ".jpeg", ".png"]:
        preview_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        st.image(preview_img, caption=uploaded_file.name, use_column_width=True)
    elif file_ext == ".pdf":
        st.caption(f"{uploaded_file.name} · {len(file_bytes) / 1024:.0f} Ko")
        _preview_ok = False
        try:
            import fitz
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            doc = fitz.open(tmp_path)
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
            preview_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            st.image(preview_img, caption=f"Page 1 / {len(doc)}", use_column_width=True)
            doc.close()
            Path(tmp_path).unlink(missing_ok=True)
            _preview_ok = True
        except Exception as exc:
            import traceback
            st.warning(f"Rendu image PDF impossible : {exc}")
        # Fallback : afficher le PDF dans un iframe intégré
        if not _preview_ok:
            try:
                pdf_b64 = base64.b64encode(file_bytes).decode()
                st.markdown(
                    f'<iframe src="data:application/pdf;base64,{pdf_b64}" '
                    f'width="100%" height="600" style="border:1px solid rgba(45,212,191,0.2);border-radius:12px;"></iframe>',
                    unsafe_allow_html=True,
                )
            except Exception:
                st.info("Utilisez votre lecteur PDF pour visualiser ce fichier.")

with col_results:
    st.markdown("#### Analyse")
    st.caption(
        "Le résumé s'affiche **en pleine largeur sous cette ligne** (section « Résultats »)."
    )

    if st.button("Lancer l'analyse forensique", type="primary", use_column_width=True):
        pdf_path = None
        try:
            if file_ext == ".pdf":
                import tempfile
                from verifdoc.utils.pdf_handler import pdf_to_images

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(file_bytes)
                    pdf_path = tmp.name
                imgs = pdf_to_images(pdf_path, dpi=200, max_pages=1)
                if not imgs:
                    st.error("PDF illisible")
                    st.stop()
                image = imgs[0]
            else:
                image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception as e:
            st.error(str(e))
            st.stop()

        t0 = time.time()
        from verifdoc.pipeline import analyze_for_dashboard

        # Progress bar temps réel
        progress_bar = st.progress(0, text="⚡ Initialisation des couches…")

        def _progress_cb(done, total):
            pct = int(done / total * 100)
            progress_bar.progress(pct, text=f"⚡ {done}/{total} couches analysées…")

        out = analyze_for_dashboard(
            image=image,
            pdf_path=pdf_path if file_ext == ".pdf" else None,
            doc_type=doc_type,
            run_ocr=run_ocr,
            progress_callback=_progress_cb,
        )

        progress_bar.progress(100, text="✅ Analyse terminée")

        elapsed = round(time.time() - t0, 2)
        final = {k: out[k] for k in out if k not in ("analysis", "visuals")}
        results = out["analysis"]
        vis = out.get("visuals") or {}
        ela_img = vis.get("ela_image")
        noise_heatmap = vis.get("heatmap")
        cm_mask = vis.get("cm_mask")

        if pdf_path:
            Path(pdf_path).unlink(missing_ok=True)

        hid = uuid.uuid4().hex[:10]
        st.session_state["vd_last"] = {
            "history_id": hid,
            "filename": uploaded_file.name,
            "final": final,
            "results": results,
            "elapsed": elapsed,
            "doc_type": doc_type,
            "ela_b64": _pil_to_b64(ela_img),
            "noise_b64": _pil_to_b64(
                Image.fromarray(noise_heatmap.astype("uint8"))
                if noise_heatmap is not None
                else None
            ),
            "cm_b64": _pil_to_b64(
                Image.fromarray(cm_mask, mode="L")
                if cm_mask is not None and cm_mask.any()
                else None
            ),
        }
        st.session_state.vd_history.insert(
            0,
            {
                "id": hid,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "fichier": uploaded_file.name,
                "risque": final.get("risk_title", ""),
                "score": f"{final.get('score_100', 0):.0f}%",
                "export": False,
            },
        )
        st.session_state.vd_history = st.session_state.vd_history[:35]

st.markdown('<hr class="vd-results-separator"/>', unsafe_allow_html=True)

if "vd_last" not in st.session_state:
    st.markdown(
        '<p class="vd-results-header">Zone résultats</p>',
        unsafe_allow_html=True,
    )
    st.info(
        "↑ **Document** à gauche · **Lancer l'analyse** à droite — puis le **résumé exécutif**, "
        "les cartes métier et le laboratoire visuel apparaissent **ici**, centrés."
    )

# ══════════════════════════════════════════════════════════════════════════════
# AFFICHAGE RÉSULTATS PERSISTANTS
# ══════════════════════════════════════════════════════════════════════════════
if "vd_last" in st.session_state:
    st.markdown(
        '<p class="vd-results-header">Résultats d\'analyse</p>',
        unsafe_allow_html=True,
    )
    L = st.session_state["vd_last"]
    final = L["final"]
    results = L["results"]
    verdict = final.get("verdict", "suspect")
    score_pct = final.get("score_100", 0)
    conf = final.get("confidence_percent", 0)
    risk_title = final.get("risk_title", "")
    risk_next = final.get("risk_next_step", "")
    doc_lbl = final.get("doc_type_label", "—")
    exec_line = html_lib.escape(final.get("executive_anomalies", ""))

    _t, _m, _a = "#e8e6e3", "#a8a4b0", "#2dd4bf"

    # ── KPIs en st.metric() ────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Type de document", doc_lbl)
    with kpi2:
        st.metric("Niveau de risque", risk_title)
    with kpi3:
        st.metric("Indice", f"{score_pct:.0f}%")
    with kpi4:
        st.metric("Confiance", f"{conf:.0f}%")

    # ── Résumé exécutif ────────────────────────────────────────────────────
    st.markdown(
        f"""
<div class="vd-exec-summary" style="color:{_t};">
  <span class="label">Résumé exécutif</span>
  <strong>Synthèse</strong> — {exec_line}<br/>
  <strong>Recommandation</strong> — {html_lib.escape(final.get("recommendation", ""))}
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Verdict box avec SVG animé ─────────────────────────────────────────
    ring_pct = min(100, max(0, float(score_pct)))
    svg_ring = _svg_score_ring(ring_pct, verdict)
    contrib_bar = _contribution_bar(final.get("layers", []))

    st.markdown(
        f"""
<div class="vd-verdict-box {verdict}">
  {svg_ring}
  <h2 style="margin:0.5rem 0 0.25rem; font-size:1.85rem !important; color:{_t} !important;">{html_lib.escape(risk_title)}</h2>
  <p style="color:{_a}; margin:0 0 0.75rem; font-size:1.05rem; font-weight:500;">{html_lib.escape(risk_next)}</p>
  {contrib_bar}
  <p style="font-size:0.8rem; color:{_m}; margin-top:1rem;">{html_lib.escape(L["filename"])} · {L["elapsed"]}s</p>
</div>
<div class="vd-disclaimer" style="color:{_m};">{html_lib.escape(final.get("disclaimer", ""))}</div>
""",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════
    # TABS : organisation par sections
    # ══════════════════════════════════════════════════════════════════════
    tab_ai, tab_biz, tab_forensic, tab_visual, tab_data = st.tabs([
        "🤖 Intelligence IA",
        "🏢 Couche métier",
        "🔬 Forensique",
        "🖼️ Laboratoire visuel",
        "📦 Données brutes",
    ])

    # ── Tab 0 : Intelligence IA ──────────────────────────────────────────
    with tab_ai:
        ai_res = results.get("ai_analysis", {})
        ai_available = ai_res.get("ai_available", False)

        if not ai_available or ai_res.get("verdict") == "skipped":
            st.markdown(
                f"""
<div class="vd-card" style="text-align:center;padding:2rem;">
  <h3 style="color:#2dd4bf !important;">🤖 Intelligence IA — Non configurée</h3>
  <p style="color:#8a8793;font-size:0.95rem;">
    Pour activer l'analyse IA par Claude Vision, configurez la variable d'environnement :
  </p>
  <code style="background:#1a1a26;padding:0.5rem 1rem;border-radius:8px;color:#2dd4bf;font-size:0.9rem;">
    export ANTHROPIC_API_KEY="sk-ant-..."
  </code>
  <p style="color:#8a8793;font-size:0.85rem;margin-top:1rem;">
    L'IA ajoute une analyse sémantique profonde : détection de cohérence, explication en langage naturel,
    identification des anomalies invisibles aux algorithmes classiques.
  </p>
</div>""",
                unsafe_allow_html=True,
            )
        elif ai_res.get("verdict") == "error":
            st.error(f"Erreur IA : {ai_res.get('detail', 'Erreur inconnue')}")
        else:
            # IA a fonctionné — afficher les résultats
            ai_score = ai_res.get("score", 0)
            ai_verdict = ai_res.get("verdict", "unknown")
            ai_explanation = ai_res.get("ai_explanation", "")
            ai_doc_type = ai_res.get("ai_doc_type", "inconnu")
            ai_doc_conf = ai_res.get("ai_doc_type_confidence", 0)
            ai_conf = ai_res.get("ai_confidence", 0)
            ai_tokens = ai_res.get("ai_tokens", {})
            ai_latency = ai_res.get("ai_latency_ms", 0)
            ai_anomalies = ai_res.get("ai_visual_anomalies", [])
            ai_forgery = ai_res.get("ai_forgery_indicators", [])
            ai_consistency = ai_res.get("ai_data_consistency", {})

            _v_color = _verdict_color(ai_verdict)

            # KPIs IA
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Score IA", f"{ai_score * 100:.0f}%")
            with c2:
                st.metric("Type détecté (IA)", ai_doc_type.replace("_", " ").title())
            with c3:
                st.metric("Confiance IA", f"{ai_conf * 100:.0f}%")
            with c4:
                st.metric("Latence", f"{ai_latency}ms")

            # Explication IA en langage naturel
            st.markdown(
                f"""
<div class="vd-exec-summary" style="border-color:{_v_color}44;">
  <span class="label" style="color:{_v_color};">Analyse Intelligence Artificielle</span>
  <div style="font-size:0.95rem;line-height:1.7;color:#e8e6e3;">
    {html_lib.escape(ai_explanation)}
  </div>
</div>""",
                unsafe_allow_html=True,
            )

            # Cohérence des données
            st.markdown('<p class="vd-section-title">Cohérence des données</p>', unsafe_allow_html=True)
            dc1, dc2, dc3 = st.columns(3)
            def _consistency_icon(val):
                if val is True:
                    return "✅"
                if val is False:
                    return "❌"
                return "➖"
            with dc1:
                st.markdown(f"{_consistency_icon(ai_consistency.get('calculations_valid'))} **Calculs**")
            with dc2:
                st.markdown(f"{_consistency_icon(ai_consistency.get('dates_coherent'))} **Dates**")
            with dc3:
                st.markdown(f"{_consistency_icon(ai_consistency.get('amounts_plausible'))} **Montants**")
            if ai_consistency.get("detail"):
                st.caption(ai_consistency["detail"])

            # Anomalies visuelles
            if ai_anomalies:
                st.markdown(f'<p class="vd-section-title">Anomalies visuelles ({len(ai_anomalies)})</p>', unsafe_allow_html=True)
                for a in ai_anomalies:
                    sev = a.get("severity", "medium")
                    sev_color = _verdict_color("forged") if sev == "high" else (_verdict_color("suspect") if sev == "medium" else _verdict_color("clean"))
                    st.markdown(
                        f'<div class="vd-alert-card {sev}">'
                        f'<div class="vd-alert-type" style="color:{sev_color};">{html_lib.escape(a.get("zone", ""))}</div>'
                        f'<div class="vd-alert-detail">{html_lib.escape(a.get("detail", ""))}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Indicateurs de falsification
            if ai_forgery:
                st.markdown(f'<p class="vd-section-title">Indicateurs de falsification ({len(ai_forgery)})</p>', unsafe_allow_html=True)
                for f in ai_forgery:
                    sev = f.get("severity", "medium")
                    sev_color = _verdict_color("forged") if sev == "high" else (_verdict_color("suspect") if sev == "medium" else _verdict_color("clean"))
                    st.markdown(
                        f'<div class="vd-alert-card {sev}">'
                        f'<div class="vd-alert-type" style="color:{sev_color};">{html_lib.escape(f.get("type", ""))}</div>'
                        f'<div class="vd-alert-detail">{html_lib.escape(f.get("detail", ""))}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Coût
            in_tok = ai_tokens.get("input", 0)
            out_tok = ai_tokens.get("output", 0)
            cost_est = (in_tok * 3 + out_tok * 15) / 1_000_000  # Sonnet pricing
            st.caption(f"Tokens : {in_tok} in / {out_tok} out · Coût estimé : ${cost_est:.4f}")

    # ── Tab 1 : Couche métier ──────────────────────────────────────────────
    with tab_biz:
        biz = final.get("business_verification") or {}
        b_siret = biz.get("siret") or {}
        b_ent = biz.get("entreprise") or {}
        b_iban = biz.get("iban") or {}
        b_tva = biz.get("tva") or {}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="vd-biz-card {_biz_card_class(b_siret.get("status", ""))}">'
                f'<div class="biz-k">SIRET</div><div class="biz-v">{html_lib.escape(b_siret.get("label", "—"))}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="vd-biz-card {_biz_card_class(b_ent.get("status", ""))}">'
                f'<div class="biz-k">Entreprise (référentiel)</div><div class="biz-v">{html_lib.escape(b_ent.get("label", "—"))}</div>'
                + (
                    f'<div class="biz-d">{html_lib.escape(str(b_ent.get("detail", "")))}</div>'
                    if b_ent.get("detail")
                    else ""
                )
                + "</div>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="vd-biz-card {_biz_card_class(b_iban.get("status", ""))}">'
                f'<div class="biz-k">IBAN</div><div class="biz-v">{html_lib.escape(b_iban.get("label", "—"))}</div>'
                + (
                    f'<div class="biz-d">{html_lib.escape(str(b_iban.get("detail", "")))}</div>'
                    if b_iban.get("detail")
                    else ""
                )
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="vd-biz-card {_biz_card_class(b_tva.get("status", ""))}">'
                f'<div class="biz-k">TVA intracommunautaire</div><div class="biz-v">{html_lib.escape(b_tva.get("label", "—"))}</div></div>',
                unsafe_allow_html=True,
            )

        # Alertes groupées par sévérité
        cross = results.get("cross_check", {})
        flags = cross.get("flags", [])
        if flags:
            st.markdown('<p class="vd-section-title">Alertes</p>', unsafe_allow_html=True)
            # Grouper par sévérité
            high_flags = [f for f in flags if f.get("severity") == "high"]
            med_flags = [f for f in flags if f.get("severity") == "medium"]
            info_flags = [f for f in flags if f.get("severity") in ("info", "low")]

            for group, label_group in [(high_flags, "Critiques"), (med_flags, "Modérées"), (info_flags, "Informatives")]:
                if not group:
                    continue
                st.markdown(f"**{label_group}** ({len(group)})")
                for f in group:
                    sev = f.get("severity", "info")
                    st.markdown(
                        f'<div class="vd-alert-card {sev}">'
                        f'<div class="vd-alert-type" style="color:{_verdict_color("forged") if sev=="high" else (_verdict_color("suspect") if sev=="medium" else _verdict_color("clean"))}">{html_lib.escape(f.get("type", ""))}</div>'
                        f'<div class="vd-alert-detail">{html_lib.escape(f.get("detail", ""))}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        if cross.get("fields_extracted"):
            with st.expander("Champs extraits par OCR", expanded=False):
                for k, v in cross["fields_extracted"].items():
                    st.markdown(f"**{k}** `{v}`")

    # ── Tab 2 : Forensique ─────────────────────────────────────────────────
    with tab_forensic:
        st.markdown(
            '<p class="vd-section-title">Signaux par couche</p>',
            unsafe_allow_html=True,
        )
        pills = ""
        for layer in final.get("layers", []):
            v = layer.get("verdict", "")
            sc = f"{layer['score'] * 100:.0f}%" if layer.get("score") is not None else "—"
            name = LAYER_LABELS.get(layer["layer"], layer["layer"])
            icon = LAYER_ICONS.get(layer["layer"], "")
            _col = _verdict_color(v)
            weight_pct = f"{layer.get('weight', 0) * 100:.0f}%"
            pills += (
                f'<span class="vd-layer-pill" style="border-left:3px solid {_col};color:#e8e6e3;">'
                f'{icon} <strong>{html_lib.escape(name)}</strong> {sc}'
                f'<span style="color:#8a8793;font-size:0.7rem;margin-left:0.5rem;">poids {weight_pct}</span>'
                f'</span>'
            )
        st.markdown(f'<div style="margin:1rem 0;">{pills}</div>', unsafe_allow_html=True)

        # Détail par couche dans des expanders
        for layer in final.get("layers", []):
            name = LAYER_LABELS.get(layer["layer"], layer["layer"])
            icon = LAYER_ICONS.get(layer["layer"], "")
            v = layer.get("verdict", "")
            sc = layer.get("score", 0)
            detail = layer.get("detail", "")
            with st.expander(f"{icon} {name} — {v} ({sc*100:.0f}%)"):
                st.markdown(f"**Détail** : {detail}")
                st.markdown(f"**Poids** : {layer.get('weight', 0)*100:.0f}%")
                # Afficher les données spécifiques si disponibles
                layer_data = results.get(layer["layer"], {})
                if layer["layer"] == "ela" and "quality_scores" in layer_data:
                    st.markdown("**Scores multi-qualité** :")
                    for q, qs in layer_data["quality_scores"].items():
                        st.markdown(f"- Qualité {q} : `{qs:.4f}`")
                if layer["layer"] == "metadata" and "metadata" in layer_data:
                    meta_info = layer_data["metadata"]
                    if meta_info.get("fonts"):
                        st.markdown(f"**Polices détectées** : {', '.join(meta_info['fonts'][:10])}")
                    if meta_info.get("page_count"):
                        st.markdown(f"**Pages** : {meta_info['page_count']}")

    # ── Tab 3 : Laboratoire visuel ─────────────────────────────────────────
    with tab_visual:
        ela_img = _b64_to_pil(L.get("ela_b64"))
        noise_img = _b64_to_pil(L.get("noise_b64"))
        cm_img = _b64_to_pil(L.get("cm_b64"))
        cm_res = results.get("copy_move") or {}
        cm_matches = int(cm_res.get("match_count") or 0)
        cm_sc = float(cm_res.get("score") or 0)
        cm_meaningful = cm_matches >= 10 or cm_sc >= 0.12

        c1, c2, c3 = st.columns(3)
        with c1:
            if ela_img:
                st.image(ela_img, caption="ELA — recompression multi-qualité")
            else:
                st.info("Pas de carte ELA disponible")
        with c2:
            if noise_img:
                st.image(noise_img, caption="Texture / wavelet multi-échelle")
            else:
                st.info("Pas de heatmap bruit disponible")
        with c3:
            if cm_img and cm_meaningful:
                st.image(cm_img, caption=f"Copy-move — {cm_matches} correspondances ORB")
            else:
                st.markdown(
                    f"""
<div style="background:#1a1a26;border:1px solid rgba(45,212,191,0.2);border-radius:12px;padding:1.25rem;min-height:120px;">
<div style="color:#2dd4bf;font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.5rem;">Copy-move</div>
<div style="color:#e8e6e3;font-size:0.92rem;">Aucune zone dupliquée <strong>significative</strong> détectée.</div>
<div style="color:#8a8793;font-size:0.8rem;margin-top:0.6rem;">Correspondances ORB : {cm_matches} — comportement normal.</div>
</div>
""",
                    unsafe_allow_html=True,
                )

        # Hotspots ELA
        ela_data = results.get("ela", {})
        hotspots = ela_data.get("hotspots", [])
        if hotspots:
            with st.expander(f"🔴 Hotspots ELA ({len(hotspots)} zones)", expanded=False):
                for i, h in enumerate(hotspots[:5]):
                    st.markdown(
                        f"**Zone {i+1}** — Position ({h['x']}, {h['y']}) · "
                        f"Taille {h['width']}×{h['height']}px · "
                        f"Intensité `{h['intensity']:.2f}`"
                    )

    # ── Tab 4 : Données brutes ─────────────────────────────────────────────
    with tab_data:
        if "metadata" in results and isinstance(results["metadata"].get("metadata"), dict):
            with st.expander("Métadonnées fichier", expanded=True):
                st.json(results["metadata"]["metadata"])

        ext_verifs = (results.get("cross_check") or {}).get("external_verifications", {})
        if ext_verifs:
            with st.expander("Référentiels externes (JSON)"):
                for key, verif in ext_verifs.items():
                    st.json({key: verif})

        with st.expander("Résultat complet (JSON)"):
            st.json(final)

    # ── Export ──────────────────────────────────────────────────────────────
    report_html = _build_html_report(
        L["filename"],
        final,
        results,
        L["elapsed"],
        DOC_LABELS.get(L["doc_type"], L["doc_type"]),
    )
    c_dl, c_arch = st.columns([2, 1])
    with c_dl:
        st.download_button(
            "📄 Télécharger le rapport HTML",
            data=report_html.encode("utf-8"),
            file_name=f"verifdoc_rapport_{Path(L['filename']).stem}.html",
            mime="text/html",
            use_column_width=True,
        )
    with c_arch:
        if st.button("Rapport exporté ✓", help="Marque cette analyse comme archivée dans l'historique", use_column_width=True):
            hid = L.get("history_id")
            if hid:
                for e in st.session_state.vd_history:
                    if e.get("id") == hid:
                        e["export"] = True
            st.success("Export enregistré dans l'historique de session.")
            st.rerun()
