"""
Intelligence IA — Couche 6 du pipeline VerifDoc.

Utilise Claude Vision (Anthropic) pour analyser sémantiquement un document :
- Identification du type de document
- Détection d'anomalies visuelles (polices, alignement, arrière-plan)
- Validation de cohérence des données (calculs, montants, dates)
- Indicateurs de falsification (collage, retouche, zones suspectes)
- Explication en langage naturel

Fallback gracieux : si la clé API est absente ou l'appel échoue,
le score est None et la couche est exclue du scoring.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time

from PIL import Image

logger = logging.getLogger(__name__)

_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Taille max de l'image envoyée à Claude (pixels sur le plus grand côté)
_MAX_IMAGE_DIM = 1500

_SYSTEM_PROMPT = """Tu es un expert en détection de fraude documentaire.
Tu analyses des documents administratifs français (bulletins de paie, factures,
avis d'imposition, RIB, relevés bancaires, quittances de loyer).

Tu dois fournir une analyse structurée au format JSON STRICT (pas de markdown, pas de commentaires).
Sois factuel, précis et prudent. Ne fabule pas — si tu n'es pas sûr, dis-le."""

_USER_PROMPT_TEMPLATE = """Analyse ce document pour détecter d'éventuels signes de falsification.

{ocr_context}

Réponds UNIQUEMENT avec un JSON valide respectant ce schéma exact :
{{
  "doc_type": "bulletin_paie|facture|avis_imposition|rib|releve_bancaire|quittance_loyer|autre|inconnu",
  "doc_type_confidence": 0.0 à 1.0,
  "risk_score": 0.0 à 1.0 (0=authentique, 1=très suspect),
  "visual_anomalies": [
    {{
      "zone": "zone du document concernée",
      "type": "police_incohérente|alignement_suspect|arrière_plan_modifié|résolution_différente|artefact_collage|autre",
      "severity": "low|medium|high",
      "detail": "description précise de l'anomalie"
    }}
  ],
  "data_consistency": {{
    "calculations_valid": true ou false ou null,
    "dates_coherent": true ou false ou null,
    "amounts_plausible": true ou false ou null,
    "detail": "explication si incohérence détectée"
  }},
  "forgery_indicators": [
    {{
      "type": "type d'indicateur",
      "severity": "low|medium|high",
      "detail": "description"
    }}
  ],
  "explanation": "Résumé en français de l'analyse complète (2-4 phrases). Explique les points forts et faibles du document.",
  "confidence": 0.0 à 1.0
}}"""


def _is_available() -> bool:
    """Vérifie si la couche IA est activable."""
    return bool(_API_KEY)


def _resize_image(image: Image.Image) -> Image.Image:
    """Redimensionne l'image pour limiter les coûts API."""
    w, h = image.size
    if max(w, h) <= _MAX_IMAGE_DIM:
        return image
    ratio = _MAX_IMAGE_DIM / max(w, h)
    new_size = (int(w * ratio), int(h * ratio))
    return image.resize(new_size, Image.LANCZOS)


def _image_to_base64(image: Image.Image) -> str:
    """Convertit une image PIL en base64 JPEG."""
    img = image.convert("RGB") if image.mode != "RGB" else image
    img = _resize_image(img)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _parse_response(text: str) -> dict | None:
    """Parse la réponse JSON de Claude, avec tolérance aux blocs markdown."""
    cleaned = text.strip()
    # Retirer les blocs ```json ... ```
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Tenter d'extraire le premier objet JSON
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    return None


def analyze(image: Image.Image, ocr_text: str | None = None) -> dict:
    """Analyse IA d'un document via Claude Vision.

    Args:
        image: Image PIL du document.
        ocr_text: Texte OCR extrait (optionnel, enrichit l'analyse).

    Returns:
        dict standard {score, verdict, detail, flags, ai_raw, ...}
    """
    if not _is_available():
        return {
            "score": None,
            "verdict": "skipped",
            "detail": "Clé API Anthropic non configurée (ANTHROPIC_API_KEY)",
            "ai_available": False,
        }

    try:
        import anthropic
    except ImportError:
        return {
            "score": None,
            "verdict": "skipped",
            "detail": "SDK anthropic non installé (pip install anthropic)",
            "ai_available": False,
        }

    ocr_context = ""
    if ocr_text and len(ocr_text.strip()) > 20:
        # Limiter le texte OCR pour ne pas exploser les tokens
        truncated = ocr_text[:3000]
        ocr_context = f"Texte extrait par OCR du document :\n---\n{truncated}\n---\n"

    user_prompt = _USER_PROMPT_TEMPLATE.format(ocr_context=ocr_context)
    img_b64 = _image_to_base64(image)

    t0 = time.time()
    try:
        client = anthropic.Anthropic(api_key=_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                }
            ],
        )
    except Exception as e:
        logger.warning("Appel Claude Vision échoué : %s", e)
        return {
            "score": None,
            "verdict": "error",
            "detail": f"Erreur API Claude : {e}",
            "ai_available": True,
        }

    elapsed_ms = int((time.time() - t0) * 1000)

    # Extraire le texte de la réponse
    raw_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            raw_text += block.text

    parsed = _parse_response(raw_text)
    if parsed is None:
        logger.warning("Réponse Claude non parseable : %s", raw_text[:500])
        return {
            "score": None,
            "verdict": "error",
            "detail": "Réponse IA non structurée",
            "ai_available": True,
            "ai_raw": raw_text[:1000],
        }

    # Mapper vers le format standard VerifDoc
    risk = float(parsed.get("risk_score", 0.5))
    risk = max(0.0, min(1.0, risk))

    if risk < 0.15:
        verdict = "clean"
    elif risk < 0.40:
        verdict = "suspect"
    else:
        verdict = "forged"

    # Construire les flags depuis les anomalies et indicateurs
    flags = []
    for anomaly in parsed.get("visual_anomalies", []):
        flags.append({
            "type": f"ia_{anomaly.get('type', 'anomalie')}",
            "severity": anomaly.get("severity", "medium"),
            "detail": f"[IA] {anomaly.get('zone', '')}: {anomaly.get('detail', '')}",
            "source": "ai",
        })
    for indicator in parsed.get("forgery_indicators", []):
        flags.append({
            "type": f"ia_{indicator.get('type', 'indicateur')}",
            "severity": indicator.get("severity", "medium"),
            "detail": f"[IA] {indicator.get('detail', '')}",
            "source": "ai",
        })

    # Cohérence des données
    consistency = parsed.get("data_consistency", {})
    consistency_issues = []
    if consistency.get("calculations_valid") is False:
        consistency_issues.append("calculs incohérents")
    if consistency.get("dates_coherent") is False:
        consistency_issues.append("dates incohérentes")
    if consistency.get("amounts_plausible") is False:
        consistency_issues.append("montants suspects")

    if consistency_issues:
        flags.append({
            "type": "ia_coherence_donnees",
            "severity": "high",
            "detail": f"[IA] Incohérences détectées : {', '.join(consistency_issues)}",
            "source": "ai",
        })

    # Tokens utilisés
    input_tokens = getattr(response.usage, "input_tokens", 0)
    output_tokens = getattr(response.usage, "output_tokens", 0)

    detail = parsed.get("explanation", "Analyse IA terminée.")

    return {
        "score": risk,
        "verdict": verdict,
        "detail": detail,
        "flags": flags,
        "ai_available": True,
        "ai_doc_type": parsed.get("doc_type"),
        "ai_doc_type_confidence": parsed.get("doc_type_confidence"),
        "ai_confidence": parsed.get("confidence", 0),
        "ai_data_consistency": consistency,
        "ai_visual_anomalies": parsed.get("visual_anomalies", []),
        "ai_forgery_indicators": parsed.get("forgery_indicators", []),
        "ai_explanation": detail,
        "ai_tokens": {"input": input_tokens, "output": output_tokens},
        "ai_latency_ms": elapsed_ms,
        "ai_raw": parsed,
    }
