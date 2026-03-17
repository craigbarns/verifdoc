"""
API REST VerifDoc — FastAPI

Endpoints:
  POST /api/v1/analyze       — Analyse complète d'un document
  POST /api/v1/analyze/quick — Analyse rapide (ELA + métadonnées seulement)
  GET  /api/v1/health        — Health check
  GET  /                     — Info API
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from ..pipeline import analyze_image
from ..analyzers import ela, metadata
from ..utils.pdf_handler import pdf_to_images, is_pdf
from ..scoring import compute_final_score

app = FastAPI(
    title="VerifDoc API",
    description="API de détection de fraude documentaire par analyse forensique IA",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — autoriser les frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Info API."""
    return {
        "name": "VerifDoc API",
        "version": "0.1.0",
        "description": "Détection de fraude documentaire par IA",
        "endpoints": {
            "analyze": "POST /api/v1/analyze",
            "quick": "POST /api/v1/analyze/quick",
            "health": "GET /api/v1/health",
            "docs": "GET /docs",
        },
    }


@app.get("/api/v1/health")
async def health():
    """Health check."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/v1/analyze")
async def analyze_full(
    file: UploadFile = File(...),
    doc_type: str = Query("auto", description="Type: auto, bulletin_paie, avis_imposition, facture, rib, releve_bancaire, quittance_loyer"),
    run_ocr: bool = Query(True, description="Activer OCR + validation croisée"),
):
    """Analyse complète d'un document (5 couches).

    Accepte : PDF, JPG, JPEG, PNG.
    Retourne : score de confiance 0-100, verdict, détail par couche.
    """
    # Valider le type de fichier
    allowed_types = [
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/jpg",
    ]
    content_type = file.content_type or ""
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if content_type not in allowed_types and ext not in [".pdf", ".jpg", ".jpeg", ".png"]:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté: {content_type}. Acceptés: PDF, JPG, PNG.",
        )

    # Lire le fichier
    content = await file.read()

    if len(content) > 20 * 1024 * 1024:  # 20 MB max
        raise HTTPException(status_code=400, detail="Fichier trop volumineux (max 20 MB)")

    pdf_path = None

    # Traiter selon le type
    if ext == ".pdf" or content[:5] == b"%PDF-":
        # Sauvegarder temporairement le PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            pdf_path = tmp.name

        try:
            images = pdf_to_images(pdf_path, dpi=200, max_pages=3)
            if not images:
                raise HTTPException(status_code=400, detail="Impossible de lire le PDF")
            image = images[0]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erreur PDF: {e}")
    else:
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image illisible: {e}")

    # Lancer l'analyse
    result = analyze_image(
        image=image,
        doc_type=doc_type,
        run_ocr=run_ocr,
        pdf_path=pdf_path,
    )
    result["file"] = filename

    # Cleanup
    if pdf_path:
        Path(pdf_path).unlink(missing_ok=True)

    return JSONResponse(content=result)


@app.post("/api/v1/analyze/quick")
async def analyze_quick(
    file: UploadFile = File(...),
):
    """Analyse rapide — ELA + métadonnées seulement.

    Plus rapide, sans OCR. Idéal pour le tri initial.
    """
    content = await file.read()
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    pdf_path_str = None

    if ext == ".pdf" or content[:5] == b"%PDF-":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            pdf_path_str = tmp.name
        try:
            images = pdf_to_images(pdf_path_str, dpi=150, max_pages=1)
            image = images[0]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erreur PDF: {e}")
    else:
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image illisible: {e}")

    results = {}

    # ELA seulement
    try:
        ela_result = ela.analyze(image)
        ela_result.pop("ela_image", None)
        results["ela"] = ela_result
    except Exception as e:
        results["ela"] = {"score": None, "error": str(e)}

    # Métadonnées
    try:
        results["metadata"] = metadata.analyze(image, pdf_path=pdf_path_str)
    except Exception as e:
        results["metadata"] = {"score": None, "error": str(e)}

    final = compute_final_score(results)

    if pdf_path_str:
        Path(pdf_path_str).unlink(missing_ok=True)

    return JSONResponse(content={
        "file": filename,
        "mode": "quick",
        **final,
        "analysis": results,
    })
