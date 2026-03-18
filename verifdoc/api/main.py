"""
API REST VerifDoc — FastAPI

Endpoints:
  POST /api/v1/analyze       — Analyse complète d'un document
  POST /api/v1/analyze/quick — Analyse rapide (ELA + métadonnées seulement)
  GET  /api/v1/health        — Health check
  GET  /                     — Info API

Sécurité :
  - Authentification par clé API (header X-API-Key)
  - Rate limiting in-memory par clé
  - CORS configuré proprement
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time as _time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from PIL import Image

from ..pipeline import analyze_image
from ..analyzers import ela, metadata
from ..utils.pdf_handler import pdf_to_images, is_pdf
from ..scoring import compute_final_score

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
API_KEYS: set[str] = set()
_raw = os.environ.get("VERIFDOC_API_KEYS", "")
if _raw:
    API_KEYS = {k.strip() for k in _raw.split(",") if k.strip()}

RATE_LIMIT_RPM = int(os.environ.get("VERIFDOC_RATE_LIMIT", "30"))  # req/min
CORS_ORIGINS = os.environ.get("VERIFDOC_CORS_ORIGINS", "").split(",")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS if o.strip()] or ["*"]

# ── Rate limiter in-memory ────────────────────────────────────────────────────
_rate_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(key: str) -> bool:
    """Retourne True si la requête est autorisée (sous la limite)."""
    now = _time.time()
    window = 60.0
    _rate_store[key] = [t for t in _rate_store[key] if now - t < window]
    if len(_rate_store[key]) >= RATE_LIMIT_RPM:
        return False
    _rate_store[key].append(now)
    return True


# ── Auth ──────────────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Depends(api_key_header)):
    """Vérifie la clé API si des clés sont configurées."""
    if not API_KEYS:
        return "anonymous"  # Pas de clés configurées = mode ouvert
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Clé API invalide ou manquante (header X-API-Key)")
    if not _check_rate_limit(api_key):
        raise HTTPException(status_code=429, detail=f"Trop de requêtes — limite : {RATE_LIMIT_RPM}/min")
    return api_key


app = FastAPI(
    title="VerifDoc API",
    description="API de détection de fraude documentaire par analyse forensique IA",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — configuré proprement (pas allow_credentials + wildcard ensemble)
_cors_credentials = "*" not in CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=_cors_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
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
    return {"status": "ok", "version": "0.2.0"}


@app.post("/api/v1/analyze")
async def analyze_full(
    file: UploadFile = File(...),
    doc_type: str = Query("auto", description="Type: auto, bulletin_paie, avis_imposition, facture, rib, releve_bancaire, quittance_loyer"),
    run_ocr: bool = Query(True, description="Activer OCR + validation croisée"),
    _key: str = Depends(verify_api_key),
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
    _key: str = Depends(verify_api_key),
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
