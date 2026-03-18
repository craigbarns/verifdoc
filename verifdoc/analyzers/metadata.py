"""
Analyse de métadonnées — Couche 4 du pipeline VerifDoc.

Examine les métadonnées PDF (Producer, Creator, ModDate)
et EXIF des images pour détecter des incohérences :
- Document créé avec un éditeur PDF suspect
- Dates de modification incohérentes
- Logiciel de retouche dans les métadonnées
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from PIL import Image

# Logiciels suspects (création/modification de PDF)
SUSPICIOUS_PRODUCERS = [
    "itextpdf", "itext", "fpdf", "tcpdf", "reportlab",
    "pdfsharp", "pdfedit", "sejda", "smallpdf", "ilovepdf",
    "pdf-xchange", "nitro", "foxit phantompdf",
    "adobe photoshop", "gimp", "canva",
]

# Producteurs légitimes courants pour les documents officiels FR
LEGITIMATE_PRODUCERS = [
    "dgfip", "impots.gouv", "caf", "cpam", "urssaf",
    "silae", "adp", "sage", "cegid", "ebp",
    "microsoft", "libreoffice", "openoffice",
]


def analyze_pdf_metadata(pdf_path: str | Path) -> dict:
    """Analyse les métadonnées d'un fichier PDF.

    Returns:
        dict avec producer, creator, dates, flags suspects.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return {"error": "PyMuPDF non installé", "flags": []}

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return {"error": f"Fichier non trouvé: {pdf_path}", "flags": []}

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        return {"error": f"PDF illisible: {e}", "flags": []}

    meta = doc.metadata or {}

    producer = (meta.get("producer") or "").strip()
    creator = (meta.get("creator") or "").strip()
    creation_date = meta.get("creationDate", "")
    mod_date = meta.get("modDate", "")
    title = (meta.get("title") or "").strip()

    flags = []
    extra_meta = {}

    # ── Vérifications avancées (avant doc.close()) ────────────────────────
    # JavaScript embarqué
    try:
        has_js = False
        for page in doc:
            for annot in page.annots() or []:
                if annot.type[0] in (19, 20):  # Widget / Screen annotations
                    has_js = True
                    break
            # Check page-level actions
            if "/JS" in (page.get_text("rawdict") or {}).get("text", ""):
                has_js = True
        # Document-level JS
        if doc.pdf_catalog():
            catalog_str = str(doc.pdf_catalog())
            if "/JavaScript" in catalog_str or "/JS" in catalog_str:
                has_js = True
        if has_js:
            flags.append({
                "type": "javascript_embedded",
                "severity": "high",
                "detail": "JavaScript embarqué détecté — risque de contenu actif malveillant",
            })
            extra_meta["has_javascript"] = True
    except Exception:
        pass

    # Fichiers embarqués (pièces jointes)
    try:
        embedded_count = doc.embfile_count()
        if embedded_count > 0:
            flags.append({
                "type": "embedded_files",
                "severity": "medium",
                "detail": f"{embedded_count} fichier(s) embarqué(s) dans le PDF",
            })
            extra_meta["embedded_files"] = embedded_count
    except Exception:
        pass

    # Chiffrement / permissions
    try:
        if doc.is_encrypted:
            flags.append({
                "type": "encrypted_pdf",
                "severity": "medium",
                "detail": "PDF chiffré — les restrictions de permissions peuvent masquer des modifications",
            })
            extra_meta["encrypted"] = True
    except Exception:
        pass

    # Analyse des polices (heuristique falsification)
    try:
        font_names = set()
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            font_list = page.get_fonts()
            for f in font_list:
                font_names.add(f[3] if len(f) > 3 else f[0])
        extra_meta["fonts"] = list(font_names)[:20]
        if len(font_names) > 12:
            flags.append({
                "type": "excessive_fonts",
                "severity": "medium",
                "detail": f"{len(font_names)} polices différentes — inhabituel pour un document officiel",
            })
    except Exception:
        pass

    # Nombre de pages
    try:
        extra_meta["page_count"] = len(doc)
    except Exception:
        pass

    doc.close()

    # Check producer suspect
    producer_lower = producer.lower()
    for s in SUSPICIOUS_PRODUCERS:
        if s in producer_lower:
            flags.append({
                "type": "suspicious_producer",
                "severity": "high",
                "detail": f"PDF créé/modifié avec '{producer}' — outil d'édition PDF",
            })
            break

    # Pas de métadonnées du tout = suspect pour un document officiel
    if not producer and not creator:
        flags.append({
            "type": "missing_metadata",
            "severity": "medium",
            "detail": "Métadonnées absentes — inhabituel pour un document officiel",
        })

    # Dates incohérentes
    d_create = _parse_pdf_date(creation_date)
    d_mod = _parse_pdf_date(mod_date)

    if d_create and d_mod:
        if d_mod < d_create:
            flags.append({
                "type": "date_inconsistency",
                "severity": "high",
                "detail": f"Date de modification ({d_mod}) antérieure à la création ({d_create})",
            })
        if d_create and d_mod:
            delta = (d_mod - d_create).days
            if delta > 365:
                flags.append({
                    "type": "late_modification",
                    "severity": "medium",
                    "detail": f"Document modifié {delta} jours après sa création",
                })

    # Score
    score = 0.0
    for f in flags:
        if f["severity"] == "high":
            score += 0.35
        elif f["severity"] == "medium":
            score += 0.15
    score = min(1.0, score)

    if score < 0.15:
        verdict = "clean"
    elif score < 0.40:
        verdict = "suspect"
    else:
        verdict = "forged"

    metadata_out = {
        "producer": producer,
        "creator": creator,
        "creation_date": str(d_create) if d_create else None,
        "modification_date": str(d_mod) if d_mod else None,
        "title": title,
    }
    metadata_out.update(extra_meta)

    return {
        "analyzer": "metadata",
        "score": round(score, 4),
        "verdict": verdict,
        "detail": f"{len(flags)} anomalie(s) dans les métadonnées" if flags else "Métadonnées cohérentes",
        "metadata": metadata_out,
        "flags": flags,
    }


def analyze_image_metadata(image: Image.Image) -> dict:
    """Analyse les métadonnées EXIF d'une image.

    Returns:
        dict avec software, dates, flags suspects.
    """
    flags = []
    exif_data = {}

    try:
        exif = image.getexif()
        if exif:
            # Tag 305 = Software
            software = exif.get(305, "")
            if software:
                exif_data["software"] = software
                sw_lower = software.lower()
                if any(s in sw_lower for s in ["photoshop", "gimp", "canva", "paint"]):
                    flags.append({
                        "type": "editing_software",
                        "severity": "high",
                        "detail": f"Image éditée avec '{software}'",
                    })

            # Tag 306 = DateTime
            date_time = exif.get(306, "")
            if date_time:
                exif_data["datetime"] = date_time
    except Exception:
        pass

    # Check format info
    if image.format:
        exif_data["format"] = image.format
    exif_data["size"] = f"{image.width}x{image.height}"
    exif_data["mode"] = image.mode

    score = 0.0
    for f in flags:
        score += 0.35 if f["severity"] == "high" else 0.15
    score = min(1.0, score)

    if score < 0.15:
        verdict = "clean"
    elif score < 0.40:
        verdict = "suspect"
    else:
        verdict = "forged"

    return {
        "analyzer": "metadata",
        "score": round(score, 4),
        "verdict": verdict,
        "detail": f"{len(flags)} anomalie(s) EXIF" if flags else "Métadonnées image cohérentes",
        "metadata": exif_data,
        "flags": flags,
    }


def _parse_pdf_date(date_str: str) -> datetime | None:
    """Parse une date PDF (format D:YYYYMMDDHHmmSS)."""
    if not date_str:
        return None
    # Nettoyer le format PDF
    clean = date_str.replace("D:", "").replace("'", "")
    # Essayer plusieurs formats
    for fmt in ["%Y%m%d%H%M%S%z", "%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d"]:
        try:
            return datetime.strptime(clean[:len(fmt.replace("%", ""))], fmt)
        except (ValueError, IndexError):
            continue
    return None


def analyze(image: Image.Image, pdf_path: str | Path | None = None) -> dict:
    """Point d'entrée unifié pour l'analyse de métadonnées.

    Si pdf_path est fourni, analyse les métadonnées PDF.
    Sinon, analyse les métadonnées EXIF de l'image.
    """
    if pdf_path:
        return analyze_pdf_metadata(pdf_path)
    return analyze_image_metadata(image)
