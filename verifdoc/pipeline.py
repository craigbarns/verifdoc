"""
Pipeline principal VerifDoc — Orchestre les 5 couches d'analyse.

Usage:
    from verifdoc.pipeline import analyze_document
    result = analyze_document("mon_document.pdf")
"""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from .analyzers import ela, noise, copy_move, metadata, cross_check
from .analyzers import ocr as ocr_module
from .scoring import compute_final_score
from .utils.pdf_handler import pdf_to_images, is_pdf


def analyze_document(
    file_path: str | Path,
    doc_type: str = "auto",
    run_ocr: bool = True,
    languages: list[str] | None = None,
) -> dict:
    """Analyse complète d'un document.

    Args:
        file_path: Chemin vers le fichier (PDF ou image).
        doc_type: "bulletin_paie", "avis_imposition", ou "auto".
        run_ocr: Activer l'OCR + validation croisée.
        languages: Langues OCR (défaut: ["fr", "en"]).

    Returns:
        dict complet avec score final, verdict, détails par couche.
    """
    file_path = Path(file_path)
    start_time = time.time()

    if not file_path.exists():
        return {"error": f"Fichier non trouvé: {file_path}"}

    # --- Étape 0 : Charger le document ---
    pdf_path = None
    if is_pdf(file_path):
        pdf_path = file_path
        images = pdf_to_images(file_path, dpi=200, max_pages=3)
        if not images:
            return {"error": "Impossible de convertir le PDF en images"}
        # Analyser la première page (la plus pertinente)
        image = images[0]
    else:
        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as e:
            return {"error": f"Image illisible: {e}"}

    results = {}

    # --- Couche 1 : ELA ---
    try:
        results["ela"] = ela.analyze(image)
        # Retirer l'objet PIL pour la sérialisation JSON
        results["ela"].pop("ela_image", None)
    except Exception as e:
        results["ela"] = {"score": None, "error": str(e)}

    # --- Couche 2 : Analyse du bruit ---
    try:
        result_noise = noise.analyze(image)
        result_noise.pop("heatmap", None)
        results["noise"] = result_noise
    except Exception as e:
        results["noise"] = {"score": None, "error": str(e)}

    # --- Couche 3 : Copy-Move ---
    try:
        result_cm = copy_move.analyze(image)
        result_cm.pop("mask", None)
        results["copy_move"] = result_cm
    except Exception as e:
        results["copy_move"] = {"score": None, "error": str(e)}

    # --- Couche 4 : Métadonnées ---
    try:
        results["metadata"] = metadata.analyze(image, pdf_path=pdf_path)
    except Exception as e:
        results["metadata"] = {"score": None, "error": str(e)}

    # --- Couche 5 : OCR + Cross-check ---
    if run_ocr:
        try:
            ocr_result = ocr_module.extract_text(image, languages=languages)
            results["ocr"] = {
                "full_text": ocr_result.get("full_text", "")[:500],  # Tronquer pour l'API
                "avg_confidence": ocr_result.get("avg_confidence", 0),
                "word_count": len(ocr_result.get("words", [])),
            }
            # Cross-check
            results["cross_check"] = cross_check.analyze(ocr_result, doc_type=doc_type)
        except Exception as e:
            results["ocr"] = {"error": str(e)}
            results["cross_check"] = {"score": None, "error": str(e)}
    else:
        results["cross_check"] = {
            "score": 0.0,
            "verdict": "skipped",
            "detail": "OCR désactivé",
        }

    # --- Score final ---
    final = compute_final_score(results)

    elapsed = round(time.time() - start_time, 2)

    return {
        "file": str(file_path.name),
        "processing_time_seconds": elapsed,
        **final,
        "analysis": results,
    }


def analyze_image(
    image: Image.Image,
    doc_type: str = "auto",
    run_ocr: bool = True,
    languages: list[str] | None = None,
    pdf_path: str | Path | None = None,
) -> dict:
    """Analyse une image PIL directement (pour l'API).

    Identique à analyze_document mais prend un objet PIL en entrée.
    """
    start_time = time.time()
    results = {}

    # Couche 1 : ELA
    try:
        results["ela"] = ela.analyze(image)
        results["ela"].pop("ela_image", None)
    except Exception as e:
        results["ela"] = {"score": None, "error": str(e)}

    # Couche 2 : Bruit
    try:
        result_noise = noise.analyze(image)
        result_noise.pop("heatmap", None)
        results["noise"] = result_noise
    except Exception as e:
        results["noise"] = {"score": None, "error": str(e)}

    # Couche 3 : Copy-Move
    try:
        result_cm = copy_move.analyze(image)
        result_cm.pop("mask", None)
        results["copy_move"] = result_cm
    except Exception as e:
        results["copy_move"] = {"score": None, "error": str(e)}

    # Couche 4 : Métadonnées
    try:
        results["metadata"] = metadata.analyze(image, pdf_path=pdf_path)
    except Exception as e:
        results["metadata"] = {"score": None, "error": str(e)}

    # Couche 5 : OCR + Cross-check
    if run_ocr:
        try:
            ocr_result = ocr_module.extract_text(image, languages=languages)
            results["ocr"] = {
                "full_text": ocr_result.get("full_text", "")[:500],
                "avg_confidence": ocr_result.get("avg_confidence", 0),
                "word_count": len(ocr_result.get("words", [])),
            }
            results["cross_check"] = cross_check.analyze(ocr_result, doc_type=doc_type)
        except Exception as e:
            results["ocr"] = {"error": str(e)}
            results["cross_check"] = {"score": None, "error": str(e)}
    else:
        results["cross_check"] = {
            "score": 0.0,
            "verdict": "skipped",
            "detail": "OCR désactivé",
        }

    final = compute_final_score(results)
    elapsed = round(time.time() - start_time, 2)

    return {
        "processing_time_seconds": elapsed,
        **final,
        "analysis": results,
    }
