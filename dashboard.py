"""
VerifDoc — Dashboard premium (Streamlit).
Pipeline forensique parallélisé + rapport exportable.
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
    "avis_imposition": "Avis d’imposition",
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
  <h1>Rapport d’analyse — VerifDoc</h1>
  <p class="meta">{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} · Fichier : <strong>{html_lib.escape(filename)}</strong></p>
  <div class="exec">
    <strong>Résumé exécutif</strong><br/>
    Type de document : {html_lib.escape(final.get("doc_type_label", "—"))}<br/>
    Niveau de risque : {html_lib.escape(final.get("risk_title", ""))} — {html_lib.escape(final.get("risk_next_step", ""))}<br/>
    Confiance d’analyse : {final.get("confidence_percent", "?")}% · Indice numérique : {final.get("score_100", 0)}%<br/>
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
    .vd-card:hover { border-color: rgba(45, 212, 191, 0.35); }
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
    .vd-ring-wrap {
      width: 140px;
      height: 140px;
      margin: 0 auto 1rem;
      position: relative;
    }
    .vd-ring {
      width: 100%;
      height: 100%;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: 'Outfit', sans-serif;
      font-weight: 800;
      font-size: 1.75rem;
      color: var(--text);
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
    }
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
    }
    .stButton > button:hover { box-shadow: 0 0 24px rgba(45, 212, 191, 0.35); }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
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
    st.markdown("**Analyse**")
    st.caption("4 signaux forensiques en parallèle + métadonnées + validation métier")

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
  <p class="sub">Signaux forensiques, cohérence métier et référentiels officiels — présentés pour la confiance, la rapidité et l’intégration équipe.</p>
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
  <div class="vd-stat"><b>5</b><span>couches d’analyse</span></div>
  <div class="vd-stat"><b>4</b><span>en parallèle</span></div>
  <div class="vd-stat"><b>∞</b><span>formats PDF & image</span></div>
</div>
<div class="vd-feature-grid">
  <div class="vd-card vd-feature"><h3>◆ Signaux forensiques</h3><p>ELA, texture et copy-move pour estimer la cohérence visuelle du fichier.</p></div>
  <div class="vd-card vd-feature"><h3>◆ Métadonnées</h3><p>Traçabilité PDF / image : chaîne de création et incohérences temporelles.</p></div>
  <div class="vd-card vd-feature"><h3>◆ Couche métier</h3><p>SIRET (référentiel), IBAN, TVA intracom — la différenciation équipe & conformité.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

file_bytes = uploaded_file.read()
file_ext = Path(uploaded_file.name).suffix.lower()

col_doc, col_results = st.columns([1, 1])

with col_doc:
    st.markdown("#### Document")
    preview_img = None
    if file_ext in [".jpg", ".jpeg", ".png"]:
        preview_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        st.image(preview_img, caption=uploaded_file.name, use_container_width=True)
    elif file_ext == ".pdf":
        st.caption(f"{uploaded_file.name} · {len(file_bytes) / 1024:.0f} Ko")
        try:
            import fitz
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            doc = fitz.open(tmp_path)
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
            preview_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            st.image(preview_img, caption=f"Page 1 / {len(doc)}", use_container_width=True)
            doc.close()
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            st.warning("Aperçu PDF indisponible")

with col_results:
    st.markdown("#### Analyse")

    if st.button("Lancer l’analyse forensique", type="primary", use_container_width=True):
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

        with st.spinner("⚡ Couches parallèles + OCR…"):
            out = analyze_for_dashboard(
                image=image,
                pdf_path=pdf_path if file_ext == ".pdf" else None,
                doc_type=doc_type,
                run_ocr=run_ocr,
            )

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

def _biz_card_class(status: str) -> str:
    if status in ("ok", "found"):
        return "ok"
    if status in ("invalid", "not_found"):
        return "bad"
    if status in ("unavailable", "unverifiable", "unknown"):
        return "warn"
    return "neutral"


# Affichage résultat persistant
if "vd_last" in st.session_state:
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

    ring_pct = min(100, max(0, float(score_pct)))

    st.markdown(
        f"""
<div class="vd-exec-summary">
  <span class="label">Résumé exécutif</span>
  <strong>Type de document détecté</strong> — {html_lib.escape(doc_lbl)}<br/>
  <strong>Niveau de risque</strong> — {html_lib.escape(risk_title)}<br/>
  <strong>Confiance d’analyse</strong> — {conf:.0f}%<br/>
  <strong>Synthèse</strong> — {exec_line}
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="vd-verdict-box {verdict}">
  <div class="vd-ring-wrap">
    <div style="width:140px;height:140px;border-radius:50%;margin:0 auto;background:conic-gradient(#f87171 0% {ring_pct:.1f}%, #1a1a26 0);display:flex;align-items:center;justify-content:center;">
      <div style="width:102px;height:102px;border-radius:50%;background:#12121a;display:flex;align-items:center;justify-content:center;font-family:Outfit,sans-serif;font-weight:800;font-size:1.5rem;color:#e8e6e3;">{score_pct:.0f}%</div>
    </div>
  </div>
  <h2 style="margin:0.5rem 0 0.25rem; font-size:1.85rem !important;">{html_lib.escape(risk_title)}</h2>
  <p style="color:var(--accent); margin:0 0 0.75rem; font-size:1.05rem; font-weight:500;">{html_lib.escape(risk_next)}</p>
  <p style="max-width:520px; margin:0 auto; color:var(--muted); font-size:0.95rem;">{html_lib.escape(final.get("recommendation", ""))}</p>
  <p style="font-size:0.8rem; color:var(--muted); margin-top:1rem;">{html_lib.escape(L["filename"])} · {L["elapsed"]}s</p>
</div>
<div class="vd-disclaimer">{html_lib.escape(final.get("disclaimer", ""))}</div>
""",
        unsafe_allow_html=True,
    )

    biz = final.get("business_verification") or {}
    st.markdown(
        '<p class="vd-section-title">Couche métier & référentiels</p>',
        unsafe_allow_html=True,
    )
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

    st.markdown(
        '<p class="vd-section-title">Signaux forensiques par couche</p>',
        unsafe_allow_html=True,
    )
    pills = ""
    for layer in final.get("layers", []):
        v = layer.get("verdict", "")
        col = "var(--ok)" if v == "clean" else ("var(--warn)" if v == "suspect" else "var(--danger)")
        sc = f"{layer['score'] * 100:.0f}%" if layer.get("score") is not None else "—"
        name = LAYER_LABELS.get(layer["layer"], layer["layer"])
        pills += f'<span class="vd-layer-pill" style="border-left:3px solid {col};"><strong>{html_lib.escape(name)}</strong> {sc}</span>'
    st.markdown(f'<div style="margin:1rem 0;">{pills}</div>', unsafe_allow_html=True)

    cross = results.get("cross_check", {})
    if cross.get("fields_extracted"):
        with st.expander("Champs extraits", expanded=False):
            for k, v in cross["fields_extracted"].items():
                st.markdown(f"**{k}** `{v}`")

    flags = cross.get("flags", [])
    if flags:
        with st.expander(f"Alertes ({len(flags)})", expanded=True):
            for f in flags:
                sev = f.get("severity", "")
                icon = "🔴" if sev == "high" else ("🟡" if sev == "medium" else "🟢")
                st.markdown(f"{icon} **{f.get('type', '')}** — {f.get('detail', '')}")

    ext_verifs = cross.get("external_verifications", {})
    if ext_verifs:
        with st.expander("Données brutes référentiels (JSON)"):
            for key, verif in ext_verifs.items():
                st.json({key: verif})

    ela_img = _b64_to_pil(L.get("ela_b64"))
    noise_img = _b64_to_pil(L.get("noise_b64"))
    cm_img = _b64_to_pil(L.get("cm_b64"))
    show_lab = ela_img or noise_img or cm_img
    if show_lab:
        st.markdown(
            '<p class="vd-section-title">Laboratoire visuel</p>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            if ela_img:
                st.image(ela_img, caption="ELA — recompression")
        with c2:
            if noise_img:
                st.image(noise_img, caption="Texture / wavelet")
        with c3:
            if cm_img:
                st.image(cm_img, caption="Copy-move")

    if "metadata" in results and isinstance(results["metadata"].get("metadata"), dict):
        with st.expander("Métadonnées fichier"):
            st.json(results["metadata"]["metadata"])

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
            "Télécharger le rapport HTML",
            data=report_html.encode("utf-8"),
            file_name=f"verifdoc_rapport_{Path(L['filename']).stem}.html",
            mime="text/html",
            use_container_width=True,
        )
    with c_arch:
        if st.button("Rapport exporté ✓", help="Marque cette analyse comme archivée dans l’historique", use_container_width=True):
            hid = L.get("history_id")
            if hid:
                for e in st.session_state.vd_history:
                    if e.get("id") == hid:
                        e["export"] = True
            st.success("Export enregistré dans l’historique de session.")
            st.rerun()
