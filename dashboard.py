"""
Dashboard Streamlit VerifDoc — Interface de démonstration.

Lance avec : streamlit run dashboard.py
"""

import streamlit as st
import io
import time
import numpy as np
from PIL import Image
from pathlib import Path

# ── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VerifDoc — Détection de Fraude Documentaire",
    page_icon="🔍",
    layout="wide",
)

# ── CSS custom ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .score-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .score-clean { background: #d4edda; border: 2px solid #28a745; }
    .score-suspect { background: #fff3cd; border: 2px solid #ffc107; }
    .score-forged { background: #f8d7da; border: 2px solid #dc3545; }
    .layer-card {
        padding: 1rem;
        border-radius: 8px;
        background: #f8f9fa;
        border-left: 4px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .layer-clean { border-left-color: #28a745; }
    .layer-suspect { border-left-color: #ffc107; }
    .layer-forged { border-left-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔍 VerifDoc</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Détection de fraude documentaire par analyse forensique IA — '
    'Bulletins de paie · Avis d\'imposition · RIB · Factures</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Paramètres")

    doc_type = st.selectbox(
        "Type de document",
        ["auto", "bulletin_paie", "avis_imposition"],
        format_func=lambda x: {
            "auto": "🔄 Détection automatique",
            "bulletin_paie": "📄 Bulletin de paie",
            "avis_imposition": "📋 Avis d'imposition",
        }.get(x, x),
    )

    run_ocr = st.checkbox("Activer OCR + validation croisée", value=False, help="Plus lent mais vérifie la cohérence des données")

    st.divider()
    st.markdown("### 📊 Pipeline d'analyse")
    st.markdown("""
    1. **ELA** — Error Level Analysis
    2. **Bruit** — Analyse wavelet
    3. **Copy-Move** — Détection ORB+RANSAC
    4. **Métadonnées** — PDF/EXIF
    5. **Cross-check** — Validation métier
    """)

    st.divider()
    st.markdown("### 💰 Tarifs")
    st.markdown("""
    - **Gratuit** : 20 vérifs/mois
    - **Pro** (99€) : 200 vérifs/mois
    - **Business** (299€) : Illimité + API
    """)

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📁 Déposez votre document ici",
    type=["pdf", "jpg", "jpeg", "png"],
    help="Formats acceptés : PDF, JPG, PNG — Max 20 MB",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_ext = Path(uploaded_file.name).suffix.lower()

    # Afficher le document uploadé
    col_doc, col_results = st.columns([1, 1])

    with col_doc:
        st.markdown("### 📄 Document uploadé")
        if file_ext in [".jpg", ".jpeg", ".png"]:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_container_width=True)
        elif file_ext == ".pdf":
            st.info(f"📑 PDF : {uploaded_file.name} ({len(file_bytes) / 1024:.0f} KB)")
            # Convertir la première page pour preview
            try:
                import fitz
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                doc = fitz.open(tmp_path)
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                st.image(image, caption=f"Page 1 / {len(doc)}", use_container_width=True)
                doc.close()
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                image = None
                st.warning("Aperçu PDF non disponible")

    # ── Lancer l'analyse ─────────────────────────────────────────────────────
    with col_results:
        st.markdown("### 🔬 Résultats de l'analyse")

        if st.button("🚀 Analyser le document", type="primary", use_container_width=True):
            # Préparer l'image
            if file_ext == ".pdf":
                try:
                    import tempfile
                    from verifdoc.utils.pdf_handler import pdf_to_images
                    from verifdoc.analyzers.metadata import analyze_pdf_metadata

                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(file_bytes)
                        pdf_path = tmp.name

                    images = pdf_to_images(pdf_path, dpi=200, max_pages=1)
                    if images:
                        image = images[0]
                    pdf_metadata = analyze_pdf_metadata(pdf_path)
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    st.stop()
            else:
                image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                pdf_path = None
                pdf_metadata = None

            # Barre de progression
            progress = st.progress(0, text="Initialisation...")

            from verifdoc.analyzers import ela, noise, copy_move, metadata
            from verifdoc.scoring import compute_final_score

            results = {}
            start = time.time()

            # Couche 1 : ELA
            progress.progress(10, text="🔍 Couche 1/5 — Error Level Analysis...")
            try:
                ela_result = ela.analyze(image)
                ela_img = ela_result.pop("ela_image", None)
                results["ela"] = ela_result
            except Exception as e:
                results["ela"] = {"score": None, "error": str(e)}
                ela_img = None

            # Couche 2 : Bruit
            progress.progress(30, text="📊 Couche 2/5 — Analyse du bruit...")
            try:
                noise_result = noise.analyze(image)
                noise_heatmap = noise_result.pop("heatmap", None)
                results["noise"] = noise_result
            except Exception as e:
                results["noise"] = {"score": None, "error": str(e)}
                noise_heatmap = None

            # Couche 3 : Copy-Move
            progress.progress(50, text="🔄 Couche 3/5 — Détection copy-move...")
            try:
                cm_result = copy_move.analyze(image)
                cm_mask = cm_result.pop("mask", None)
                results["copy_move"] = cm_result
            except Exception as e:
                results["copy_move"] = {"score": None, "error": str(e)}
                cm_mask = None

            # Couche 4 : Métadonnées
            progress.progress(70, text="📋 Couche 4/5 — Analyse métadonnées...")
            if pdf_metadata:
                results["metadata"] = pdf_metadata
            else:
                try:
                    results["metadata"] = metadata.analyze(image)
                except Exception as e:
                    results["metadata"] = {"score": None, "error": str(e)}

            # Couche 5 : OCR (optionnel)
            if run_ocr:
                progress.progress(85, text="📝 Couche 5/5 — OCR + validation...")
                try:
                    from verifdoc.analyzers import ocr as ocr_module, cross_check
                    ocr_result = ocr_module.extract_text(image)
                    results["cross_check"] = cross_check.analyze(ocr_result, doc_type=doc_type)
                except Exception as e:
                    results["cross_check"] = {"score": None, "error": str(e)}
            else:
                results["cross_check"] = {"score": 0.0, "verdict": "skipped", "detail": "OCR désactivé"}

            progress.progress(100, text="✅ Analyse terminée !")
            elapsed = round(time.time() - start, 2)
            time.sleep(0.3)
            progress.empty()

            # ── Affichage des résultats ──────────────────────────────────────
            final = compute_final_score(results)

            # Score principal
            verdict = final["verdict"]
            score_pct = final["score_100"]
            css_class = f"score-{verdict}"

            verdict_emoji = {"clean": "✅", "suspect": "⚠️", "forged": "🚨"}.get(verdict, "❓")
            verdict_label = {"clean": "AUTHENTIQUE", "suspect": "SUSPECT", "forged": "FALSIFIÉ"}.get(verdict, "INCONNU")

            st.markdown(
                f'<div class="score-card {css_class}">'
                f'<h1>{verdict_emoji} {verdict_label}</h1>'
                f'<h2>Score de risque : {score_pct}%</h2>'
                f'<p>{final["recommendation"]}</p>'
                f'<small>Analyse en {elapsed}s — {final["analyzers_run"]} couches</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Détail par couche
            st.markdown("#### 📊 Détail par couche d'analyse")

            for layer in final["layers"]:
                v = layer["verdict"]
                icon = {"clean": "✅", "suspect": "⚠️", "forged": "🚨"}.get(v, "➖")
                layer_name = {
                    "ela": "ELA (Error Level Analysis)",
                    "noise": "Analyse du bruit",
                    "copy_move": "Détection copy-move",
                    "metadata": "Métadonnées",
                    "cross_check": "Validation croisée",
                }.get(layer["layer"], layer["layer"])

                css = f"layer-{v}"
                score_str = f"{layer['score'] * 100:.0f}%" if layer["score"] is not None else "N/A"

                st.markdown(
                    f'<div class="layer-card {css}">'
                    f'{icon} <strong>{layer_name}</strong> — Score : {score_str}<br/>'
                    f'<small>{layer["detail"]}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Images forensiques
            if ela_img or noise_heatmap or cm_mask is not None:
                st.markdown("#### 🖼️ Visualisations forensiques")
                vis_cols = st.columns(3)

                with vis_cols[0]:
                    if ela_img:
                        st.image(ela_img, caption="Carte ELA", use_container_width=True)

                with vis_cols[1]:
                    if noise_heatmap is not None:
                        st.image(noise_heatmap, caption="Heatmap bruit (wavelet)", use_container_width=True)

                with vis_cols[2]:
                    if cm_mask is not None and cm_mask.any():
                        st.image(cm_mask, caption="Masque copy-move", use_container_width=True)

            # Métadonnées
            if "metadata" in results and "metadata" in results["metadata"]:
                with st.expander("📋 Métadonnées du document"):
                    st.json(results["metadata"]["metadata"])

            # Cleanup
            if file_ext == ".pdf" and pdf_path:
                Path(pdf_path).unlink(missing_ok=True)

else:
    # État vide
    st.markdown("---")
    st.markdown(
        "### 👆 Uploadez un document pour commencer l'analyse\n\n"
        "**Formats supportés :** PDF, JPG, JPEG, PNG\n\n"
        "**Documents pris en charge :**\n"
        "- Bulletins de paie\n"
        "- Avis d'imposition\n"
        "- RIB / Relevés bancaires\n"
        "- Factures\n"
        "- Tout document scanné ou PDF"
    )
