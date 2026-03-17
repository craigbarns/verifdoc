"""
Extraction OCR — Couche 5a du pipeline VerifDoc.

Extrait le texte des documents pour permettre la validation croisée.
Utilise EasyOCR (multi-langue, support français natif).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Cache global pour le reader EasyOCR (lourd à charger)
_reader_cache: dict = {}


def _get_reader(languages: list[str], gpu: bool = False):
    """Charge ou récupère le reader EasyOCR depuis le cache."""
    try:
        import easyocr
    except ImportError:
        return None

    key = (tuple(languages), gpu)
    if key not in _reader_cache:
        _reader_cache[key] = easyocr.Reader(languages, gpu=gpu, verbose=False)
    return _reader_cache[key]


def extract_text(
    source: str | Path | Image.Image,
    languages: list[str] | None = None,
    gpu: bool = False,
) -> dict:
    """Extrait le texte d'un document image.

    Args:
        source: Chemin image ou PIL Image.
        languages: Codes langue (défaut: ["fr", "en"]).
        gpu: Activer GPU pour EasyOCR.

    Returns:
        dict avec full_text, words (avec bbox + confidence), avg_confidence.
    """
    if languages is None:
        languages = ["fr", "en"]

    reader = _get_reader(languages, gpu)
    if reader is None:
        return {
            "full_text": "",
            "words": [],
            "avg_confidence": 0.0,
            "error": "EasyOCR non installé — pip install easyocr",
        }

    # Convertir en array numpy si PIL Image
    if isinstance(source, Image.Image):
        img_array = np.array(source.convert("RGB"))
    else:
        img_array = str(source)

    try:
        results = reader.readtext(img_array)
    except Exception as e:
        return {
            "full_text": "",
            "words": [],
            "avg_confidence": 0.0,
            "error": str(e),
        }

    words = []
    for bbox, text, conf in results:
        words.append({
            "text": text.strip(),
            "confidence": round(float(conf), 4),
            "bbox": [[int(p[0]), int(p[1])] for p in bbox],
        })

    full_text = " ".join(w["text"] for w in words if w["text"])
    avg_conf = float(np.mean([w["confidence"] for w in words])) if words else 0.0

    return {
        "full_text": full_text,
        "words": words,
        "avg_confidence": round(avg_conf, 4),
    }


def extract_fields_bulletin_paie(ocr_result: dict) -> dict:
    """Extrait les champs clés d'un bulletin de paie français.

    Recherche : nom, prénom, SIRET, salaire brut/net, période.
    """
    import re

    text = ocr_result.get("full_text", "")
    text_upper = text.upper()

    fields = {}

    # SIRET (14 chiffres)
    siret_match = re.search(r'\b(\d{3}\s?\d{3}\s?\d{3}\s?\d{5})\b', text)
    if siret_match:
        fields["siret"] = siret_match.group(1).replace(" ", "")

    # Salaire net (patterns courants)
    net_patterns = [
        r'NET\s*[AÀ]\s*PAYER\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'NET\s*IMPOSABLE\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'NET\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
    ]
    for pattern in net_patterns:
        match = re.search(pattern, text_upper)
        if match:
            amount = match.group(1).replace(" ", "").replace(",", ".")
            try:
                fields["net_a_payer"] = float(amount)
            except ValueError:
                pass
            break

    # Salaire brut
    brut_match = re.search(
        r'SALAIRE\s*BRUT\s*[:\s]*(\d[\d\s]*[.,]\d{2})', text_upper
    )
    if brut_match:
        amount = brut_match.group(1).replace(" ", "").replace(",", ".")
        try:
            fields["salaire_brut"] = float(amount)
        except ValueError:
            pass

    # Période (mois/année)
    mois_match = re.search(
        r'(JANVIER|FÉVRIER|FEVRIER|MARS|AVRIL|MAI|JUIN|JUILLET|'
        r'AOÛT|AOUT|SEPTEMBRE|OCTOBRE|NOVEMBRE|DÉCEMBRE|DECEMBRE)\s*(\d{4})',
        text_upper,
    )
    if mois_match:
        fields["periode"] = f"{mois_match.group(1)} {mois_match.group(2)}"

    return fields


def extract_fields_avis_imposition(ocr_result: dict) -> dict:
    """Extrait les champs clés d'un avis d'imposition français.

    Recherche : revenu fiscal de référence, nombre de parts, impôt.
    """
    import re

    text = ocr_result.get("full_text", "")
    text_upper = text.upper()

    fields = {}

    # Revenu fiscal de référence
    rfr_match = re.search(
        r'REVENU\s*FISCAL\s*DE\s*R[EÉ]F[EÉ]RENCE\s*[:\s]*(\d[\d\s]*)',
        text_upper,
    )
    if rfr_match:
        amount = rfr_match.group(1).replace(" ", "")
        try:
            fields["revenu_fiscal_reference"] = int(amount)
        except ValueError:
            pass

    # Nombre de parts
    parts_match = re.search(r'NOMBRE\s*DE\s*PARTS?\s*[:\s]*([\d,\.]+)', text_upper)
    if parts_match:
        try:
            fields["nombre_parts"] = float(parts_match.group(1).replace(",", "."))
        except ValueError:
            pass

    # Impôt sur le revenu
    impot_match = re.search(
        r'IMP[ÔO]T\s*SUR\s*LE\s*REVENU\s*NET\s*[:\s]*(\d[\d\s]*)',
        text_upper,
    )
    if impot_match:
        amount = impot_match.group(1).replace(" ", "")
        try:
            fields["impot_revenu"] = int(amount)
        except ValueError:
            pass

    # Année d'imposition
    annee_match = re.search(r'REVENUS\s*(\d{4})', text_upper)
    if annee_match:
        fields["annee_revenus"] = int(annee_match.group(1))

    return fields
