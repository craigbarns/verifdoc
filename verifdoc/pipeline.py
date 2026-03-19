"""
Pipeline principal VerifDoc — Orchestre les 6 couches d'analyse.

ELA, bruit, copy-move, métadonnées, OCR+cross_check et Intelligence IA
s'exécutent en parallèle pour réduire la latence.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image

from .analyzers import ela, noise, copy_move, metadata, cross_check, ai_analysis
from .analyzers import ocr as ocr_module
from .scoring import compute_final_score
from .utils.pdf_handler import pdf_to_images, is_pdf


def _task_ela(image: Image.Image, keep_visuals: bool) -> tuple[str, dict, tuple | None]:
    try:
        r = ela.analyze(image)
        extra = None
        if keep_visuals:
            ei = r.pop("ela_image", None)
            extra = ("ela_image", ei)
        else:
            r.pop("ela_image", None)
        return "ela", r, extra
    except Exception as e:
        return "ela", {"score": None, "error": str(e)}, ("ela_image", None) if keep_visuals else None


def _task_noise(image: Image.Image, keep_visuals: bool) -> tuple[str, dict, tuple | None]:
    try:
        r = noise.analyze(image)
        extra = None
        if keep_visuals:
            hm = r.pop("heatmap", None)
            extra = ("heatmap", hm)
        else:
            r.pop("heatmap", None)
        return "noise", r, extra
    except Exception as e:
        return "noise", {"score": None, "error": str(e)}, ("heatmap", None) if keep_visuals else None


def _task_copy_move(image: Image.Image, keep_visuals: bool) -> tuple[str, dict, tuple | None]:
    try:
        r = copy_move.analyze(image)
        extra = None
        if keep_visuals:
            m = r.pop("mask", None)
            extra = ("cm_mask", m)
        else:
            r.pop("mask", None)
        return "copy_move", r, extra
    except Exception as e:
        return "copy_move", {"score": None, "error": str(e)}, ("cm_mask", None) if keep_visuals else None


def _task_metadata(image: Image.Image, pdf_path: Path | str | None) -> tuple[str, dict, None]:
    try:
        return "metadata", metadata.analyze(image, pdf_path=pdf_path), None
    except Exception as e:
        return "metadata", {"score": None, "error": str(e)}, None


def _run_layers_parallel(
    image: Image.Image,
    pdf_path: Path | str | None,
    keep_visuals: bool,
    progress_callback: Any | None = None,
) -> tuple[dict[str, dict], dict[str, Any]]:
    """Exécute ELA, bruit, copy-move et métadonnées en parallèle."""
    results: dict[str, dict] = {}
    visuals: dict[str, Any] = {}
    layer_names = ["ela", "noise", "copy_move", "metadata"]
    done_count = 0

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [
            ex.submit(_task_ela, image, keep_visuals),
            ex.submit(_task_noise, image, keep_visuals),
            ex.submit(_task_copy_move, image, keep_visuals),
            ex.submit(_task_metadata, image, pdf_path),
        ]
        for fut in as_completed(futures):
            key, res, extra = fut.result()
            results[key] = res
            if extra and extra[1] is not None:
                visuals[extra[0]] = extra[1]
            done_count += 1
            if progress_callback:
                try:
                    progress_callback(done_count, len(layer_names), key)
                except Exception:
                    pass

    return results, visuals


def _run_ocr_cross(
    image: Image.Image,
    pdf_path: Path | str | None,
    doc_type: str,
    languages: list[str] | None,
    run_ocr: bool,
) -> tuple[dict, dict]:
    if not run_ocr:
        return (
            {"cross_check": {"score": 0.0, "verdict": "skipped", "detail": "OCR désactivé"}},
            {},
        )
    try:
        ocr_result = ocr_module.extract_text(
            image, languages=languages, pdf_path=pdf_path if pdf_path else None
        )
        full_text = ocr_result.get("full_text", "")
        ocr_summary = {
            "full_text": full_text[:500],  # Résumé tronqué pour l'UI
            "full_text_complete": full_text,  # Texte complet pour l'IA
            "avg_confidence": ocr_result.get("avg_confidence", 0),
            "word_count": len(ocr_result.get("words", [])),
        }
        cross = cross_check.analyze(ocr_result, doc_type=doc_type)
        return (
            {"ocr": ocr_summary, "cross_check": cross},
            {},
        )
    except Exception as e:
        return (
            {"ocr": {"error": str(e)}, "cross_check": {"score": None, "error": str(e)}},
            {},
        )


def _task_ai_analysis(image: Image.Image, ocr_text: str | None) -> tuple[str, dict, None]:
    """Worker IA — analyse sémantique via Claude Vision."""
    try:
        r = ai_analysis.analyze(image, ocr_text=ocr_text)
        return "ai_analysis", r, None
    except Exception as e:
        return "ai_analysis", {"score": None, "error": str(e), "ai_available": False}, None


def _run_all_parallel(
    image: Image.Image,
    pdf_path: Path | str | None,
    keep_visuals: bool,
    doc_type: str,
    languages: list[str] | None,
    run_ocr: bool,
    progress_callback: Any | None = None,
) -> tuple[dict[str, dict], dict[str, Any]]:
    """Exécute TOUTES les couches en parallèle (forensiques + OCR + IA).

    Jusqu'à 6 workers : ELA, bruit, copy-move, métadonnées, OCR+cross_check, Intelligence IA.
    L'IA ne s'exécute que si ANTHROPIC_API_KEY est configurée.
    """
    results: dict[str, dict] = {}
    visuals: dict[str, Any] = {}
    ai_enabled = ai_analysis._is_available()
    total_tasks = 4 + (1 if run_ocr else 0) + (1 if ai_enabled else 0)
    done_count = 0
    ocr_text_holder: list[str | None] = [None]

    def _task_ocr_cross():
        result = _run_ocr_cross(image, pdf_path, doc_type, languages, run_ocr)
        # Capturer le texte OCR complet pour l'IA
        ocr_block = result[0]
        ocr_data = ocr_block.get("ocr", {})
        ocr_text_holder[0] = ocr_data.get("full_text_complete") or ocr_data.get("full_text")
        return result

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {
            ex.submit(_task_ela, image, keep_visuals): "forensic",
            ex.submit(_task_noise, image, keep_visuals): "forensic",
            ex.submit(_task_copy_move, image, keep_visuals): "forensic",
            ex.submit(_task_metadata, image, pdf_path): "forensic",
        }
        if run_ocr:
            futures[ex.submit(_task_ocr_cross)] = "ocr"

        # Phase 1 : forensiques + OCR en parallèle
        for fut in as_completed(futures):
            task_type = futures[fut]
            if task_type == "forensic":
                key, res, extra = fut.result()
                results[key] = res
                if extra and extra[1] is not None:
                    visuals[extra[0]] = extra[1]
            elif task_type == "ocr":
                ocr_block, _ = fut.result()
                results.update(ocr_block)
            done_count += 1
            if progress_callback:
                try:
                    progress_callback(done_count, total_tasks)
                except Exception:
                    pass

        # Phase 2 : IA APRÈS l'OCR pour bénéficier du texte extrait
        if ai_enabled:
            ocr_text = ocr_text_holder[0]
            ai_fut = ex.submit(_task_ai_analysis, image, ocr_text)
            _, ai_res, _ = ai_fut.result()
            results["ai_analysis"] = ai_res
            done_count += 1
            if progress_callback:
                try:
                    progress_callback(done_count, total_tasks)
                except Exception:
                    pass

    # Nettoyer le texte complet de la réponse (pas utile côté client)
    if "ocr" in results and "full_text_complete" in results["ocr"]:
        del results["ocr"]["full_text_complete"]

    # Si OCR désactivé, ajouter le bloc skipped
    if not run_ocr and "cross_check" not in results:
        results["cross_check"] = {"score": 0.0, "verdict": "skipped", "detail": "OCR désactivé"}

    # Si IA non disponible, ajouter le bloc skipped
    if not ai_enabled and "ai_analysis" not in results:
        results["ai_analysis"] = {
            "score": None,
            "verdict": "skipped",
            "detail": "ANTHROPIC_API_KEY non configurée",
            "ai_available": False,
        }

    return results, visuals


def analyze_document(
    file_path: str | Path,
    doc_type: str = "auto",
    run_ocr: bool = True,
    languages: list[str] | None = None,
) -> dict:
    """Analyse complète d'un document (fichier PDF ou image)."""
    file_path = Path(file_path)
    start_time = time.time()

    if not file_path.exists():
        return {"error": f"Fichier non trouvé: {file_path}"}

    pdf_path = None
    if is_pdf(file_path):
        pdf_path = file_path
        images = pdf_to_images(file_path, dpi=200, max_pages=3)
        if not images:
            return {"error": "Impossible de convertir le PDF en images"}
        image = images[0]
    else:
        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as e:
            return {"error": f"Image illisible: {e}"}

    results, _ = _run_all_parallel(image, pdf_path, False, doc_type, languages, run_ocr)

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
    """Analyse une image PIL (API). Toutes les couches en parallèle."""
    start_time = time.time()
    results, _ = _run_all_parallel(image, pdf_path, False, doc_type, languages, run_ocr)

    final = compute_final_score(results)
    elapsed = round(time.time() - start_time, 2)

    return {
        "processing_time_seconds": elapsed,
        **final,
        "analysis": results,
    }


def analyze_for_dashboard(
    image: Image.Image,
    pdf_path: str | Path | None,
    doc_type: str,
    run_ocr: bool,
    languages: list[str] | None = None,
    progress_callback: Any | None = None,
) -> dict:
    """Analyse complète pour le dashboard : résultats + visuels forensiques.

    Jusqu'à 6 couches en parallèle (dont IA si configurée).
    progress_callback(done, total) pour la barre de progression.
    """
    start_time = time.time()
    results, visuals = _run_all_parallel(
        image, pdf_path, True, doc_type, languages, run_ocr, progress_callback
    )

    final = compute_final_score(results)
    elapsed = round(time.time() - start_time, 2)

    return {
        "processing_time_seconds": elapsed,
        **final,
        "analysis": results,
        "visuals": {
            "ela_image": visuals.get("ela_image"),
            "heatmap": visuals.get("heatmap"),
            "cm_mask": visuals.get("cm_mask"),
        },
    }
