"""
Validation croisée métier — Couche 5b du pipeline VerifDoc.

Vérifie la cohérence interne des données extraites par OCR :
- Ratio net/brut sur bulletin de paie (cohérent ?)
- Format SIRET valide (clé de Luhn)
- Montants plausibles
- Cohérence entre documents (bulletin de paie vs avis d'imposition)
"""

from __future__ import annotations

import re


def validate_siret(siret: str) -> dict:
    """Vérifie la validité d'un numéro SIRET (algorithme de Luhn).

    Un SIRET = SIREN (9 chiffres) + NIC (5 chiffres) = 14 chiffres.
    """
    clean = siret.replace(" ", "")

    if not re.match(r'^\d{14}$', clean):
        return {
            "valid": False,
            "detail": f"Format invalide — {len(clean)} chiffres au lieu de 14",
        }

    # Algorithme de Luhn
    total = 0
    for i, digit in enumerate(clean):
        n = int(digit)
        if i % 2 == 1:  # Position paire (0-indexed impaire)
            n *= 2
            if n > 9:
                n -= 9
        total += n

    is_valid = total % 10 == 0

    return {
        "valid": is_valid,
        "siren": clean[:9],
        "nic": clean[9:],
        "detail": "SIRET valide" if is_valid else "Clé de contrôle SIRET invalide — document suspect",
    }


def validate_bulletin_paie(fields: dict) -> dict:
    """Valide la cohérence d'un bulletin de paie.

    Vérifications :
    - Ratio net/brut entre 0.60 et 0.85 (standard FR)
    - Montants dans des plages réalistes
    - SIRET valide
    """
    flags = []
    score = 0.0

    brut = fields.get("salaire_brut")
    net = fields.get("net_a_payer")
    siret = fields.get("siret")

    # Check ratio net/brut
    if brut and net and brut > 0:
        ratio = net / brut
        if ratio > 1.0:
            # Net supérieur au brut = impossible
            flags.append({
                "type": "ratio_impossible",
                "severity": "high",
                "detail": f"Net ({net}€) supérieur au brut ({brut}€) — impossible",
            })
            score += 0.50
        elif ratio < 0.55 or ratio > 0.90:
            flags.append({
                "type": "ratio_incoherent",
                "severity": "high",
                "detail": f"Ratio net/brut = {ratio:.2f} — attendu entre 0.60 et 0.85",
            })
            score += 0.35
        elif ratio < 0.60 or ratio > 0.85:
            flags.append({
                "type": "ratio_inhabituel",
                "severity": "medium",
                "detail": f"Ratio net/brut = {ratio:.2f} — légèrement inhabituel",
            })
            score += 0.15

    # Check montants réalistes (SMIC ~1400€ net, plafond raisonnable ~15000€)
    if net:
        if net < 500:
            flags.append({
                "type": "montant_bas",
                "severity": "medium",
                "detail": f"Net à payer très bas ({net}€) — vérifier",
            })
            score += 0.10
        elif net > 20000:
            flags.append({
                "type": "montant_eleve",
                "severity": "medium",
                "detail": f"Net à payer très élevé ({net}€) — vérifier",
            })
            score += 0.10

    # Check SIRET
    if siret:
        siret_result = validate_siret(siret)
        if not siret_result["valid"]:
            flags.append({
                "type": "siret_invalide",
                "severity": "high",
                "detail": siret_result["detail"],
            })
            score += 0.40

    score = min(1.0, score)

    if score < 0.15:
        verdict = "clean"
        detail = "Données du bulletin cohérentes"
    elif score < 0.40:
        verdict = "suspect"
        detail = f"{len(flags)} anomalie(s) détectée(s)"
    else:
        verdict = "forged"
        detail = f"{len(flags)} incohérence(s) majeure(s)"

    return {
        "analyzer": "cross_check",
        "score": round(score, 4),
        "verdict": verdict,
        "detail": detail,
        "fields_extracted": fields,
        "flags": flags,
    }


def validate_avis_imposition(fields: dict) -> dict:
    """Valide la cohérence d'un avis d'imposition.

    Vérifications :
    - Revenu fiscal de référence plausible
    - Nombre de parts cohérent
    - Année de revenus récente
    """
    flags = []
    score = 0.0

    rfr = fields.get("revenu_fiscal_reference")
    parts = fields.get("nombre_parts")
    annee = fields.get("annee_revenus")
    impot = fields.get("impot_revenu")

    # Check RFR plausible
    if rfr is not None:
        if rfr < 0:
            flags.append({
                "type": "rfr_negatif",
                "severity": "high",
                "detail": f"RFR négatif ({rfr}€) — impossible",
            })
            score += 0.40
        elif rfr > 500000:
            flags.append({
                "type": "rfr_tres_eleve",
                "severity": "medium",
                "detail": f"RFR très élevé ({rfr}€) — vérifier",
            })
            score += 0.10

    # Check nombre de parts
    if parts is not None:
        if parts < 1 or parts > 10:
            flags.append({
                "type": "parts_incoherentes",
                "severity": "high",
                "detail": f"Nombre de parts = {parts} — hors plage normale",
            })
            score += 0.30
        # Les parts sont toujours des multiples de 0.25 ou 0.5
        if (parts * 4) % 1 != 0:
            flags.append({
                "type": "parts_invalides",
                "severity": "high",
                "detail": f"Nombre de parts = {parts} — pas un multiple de 0.25",
            })
            score += 0.30

    # Check année
    if annee is not None:
        from datetime import datetime
        current_year = datetime.now().year
        if annee > current_year or annee < 2000:
            flags.append({
                "type": "annee_incoherente",
                "severity": "high",
                "detail": f"Année des revenus = {annee} — incohérent",
            })
            score += 0.35

    # Check cohérence impôt vs RFR
    if rfr and impot and rfr > 0:
        taux_effectif = impot / rfr
        if taux_effectif > 0.50:
            flags.append({
                "type": "taux_impot_eleve",
                "severity": "medium",
                "detail": f"Taux effectif d'imposition = {taux_effectif:.0%} — très élevé",
            })
            score += 0.15

    score = min(1.0, score)

    if score < 0.15:
        verdict = "clean"
        detail = "Données de l'avis d'imposition cohérentes"
    elif score < 0.40:
        verdict = "suspect"
        detail = f"{len(flags)} anomalie(s) détectée(s)"
    else:
        verdict = "forged"
        detail = f"{len(flags)} incohérence(s) majeure(s)"

    return {
        "analyzer": "cross_check",
        "score": round(score, 4),
        "verdict": verdict,
        "detail": detail,
        "fields_extracted": fields,
        "flags": flags,
    }


def analyze(ocr_result: dict, doc_type: str = "auto") -> dict:
    """Point d'entrée pour la validation croisée.

    Args:
        ocr_result: Résultat de l'OCR (dict avec full_text, words).
        doc_type: "bulletin_paie", "avis_imposition", ou "auto".

    Returns:
        dict avec score, verdict, flags.
    """
    from .ocr import extract_fields_bulletin_paie, extract_fields_avis_imposition

    text_upper = ocr_result.get("full_text", "").upper()

    # Auto-détection du type de document
    if doc_type == "auto":
        if any(kw in text_upper for kw in ["BULLETIN", "PAIE", "SALAIRE BRUT", "NET A PAYER", "NET À PAYER"]):
            doc_type = "bulletin_paie"
        elif any(kw in text_upper for kw in ["AVIS D'IMPOSITION", "REVENU FISCAL", "IMPOT SUR LE REVENU", "IMPÔT SUR LE REVENU"]):
            doc_type = "avis_imposition"
        else:
            return {
                "analyzer": "cross_check",
                "score": 0.0,
                "verdict": "unknown",
                "detail": "Type de document non reconnu — validation croisée non applicable",
                "doc_type": "unknown",
                "fields_extracted": {},
                "flags": [],
            }

    if doc_type == "bulletin_paie":
        fields = extract_fields_bulletin_paie(ocr_result)
        result = validate_bulletin_paie(fields)
    elif doc_type == "avis_imposition":
        fields = extract_fields_avis_imposition(ocr_result)
        result = validate_avis_imposition(fields)
    else:
        return {
            "analyzer": "cross_check",
            "score": 0.0,
            "verdict": "unknown",
            "detail": f"Type '{doc_type}' non supporté",
            "flags": [],
        }

    result["doc_type"] = doc_type
    return result
