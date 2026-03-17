"""
Validation croisée métier — Couche 5b du pipeline VerifDoc.

Types supportés :
  - Bulletin de paie : ratio net/brut, SIRET, montants
  - Avis d'imposition : RFR, parts, taux effectif
  - Facture : cohérence HT/TVA/TTC, SIRET, IBAN
  - RIB : validation IBAN (mod 97), BIC
  - Relevé bancaire : solde ancien + crédits - débits = solde nouveau
  - Quittance de loyer : loyer + charges = total
"""

from __future__ import annotations

import re


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATEURS UNITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def validate_siret(siret: str) -> dict:
    """Vérifie un SIRET via l'algorithme de Luhn."""
    clean = siret.replace(" ", "")

    if not re.match(r'^\d{14}$', clean):
        return {
            "valid": False,
            "detail": f"Format invalide — {len(clean)} chiffres au lieu de 14",
        }

    total = 0
    for i, digit in enumerate(clean):
        n = int(digit)
        if i % 2 == 1:
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


def validate_iban(iban: str) -> dict:
    """Vérifie un IBAN via modulo 97 (ISO 13616)."""
    clean = iban.replace(" ", "").upper()

    if len(clean) < 15 or len(clean) > 34:
        return {
            "valid": False,
            "detail": f"Longueur IBAN invalide ({len(clean)} caractères)",
        }

    if not re.match(r'^[A-Z]{2}\d{2}[A-Z0-9]+$', clean):
        return {
            "valid": False,
            "detail": "Format IBAN invalide",
        }

    # Vérification FR : doit faire 27 caractères
    if clean[:2] == "FR" and len(clean) != 27:
        return {
            "valid": False,
            "detail": f"IBAN français doit faire 27 caractères, trouvé {len(clean)}",
        }

    # Modulo 97
    rearranged = clean[4:] + clean[:4]
    numeric = ""
    for char in rearranged:
        if char.isdigit():
            numeric += char
        else:
            numeric += str(ord(char) - 55)

    try:
        is_valid = int(numeric) % 97 == 1
    except (ValueError, OverflowError):
        return {"valid": False, "detail": "Erreur de calcul IBAN"}

    return {
        "valid": is_valid,
        "country": clean[:2],
        "detail": "IBAN valide" if is_valid else "Clé de contrôle IBAN invalide — document suspect",
    }


def validate_bic(bic: str) -> dict:
    """Vérifie le format d'un code BIC/SWIFT."""
    clean = bic.replace(" ", "").upper()
    if re.match(r'^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$', clean):
        return {"valid": True, "detail": "BIC valide"}
    return {"valid": False, "detail": "Format BIC invalide"}


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATEURS PAR TYPE DE DOCUMENT
# ══════════════════════════════════════════════════════════════════════════════

def _make_result(analyzer: str, score: float, flags: list) -> dict:
    """Génère un résultat standardisé."""
    score = min(1.0, score)
    if score < 0.15:
        verdict = "clean"
    elif score < 0.40:
        verdict = "suspect"
    else:
        verdict = "forged"

    if not flags:
        detail = "Données du document cohérentes"
    elif verdict == "forged":
        detail = f"{len(flags)} incohérence(s) majeure(s)"
    else:
        detail = f"{len(flags)} anomalie(s) détectée(s)"

    return {
        "analyzer": analyzer,
        "score": round(score, 4),
        "verdict": verdict,
        "detail": detail,
        "flags": flags,
    }


def validate_bulletin_paie(fields: dict) -> dict:
    """Valide un bulletin de paie."""
    flags = []
    score = 0.0

    brut = fields.get("salaire_brut")
    net = fields.get("net_a_payer")
    siret = fields.get("siret")

    if brut and net and brut > 0:
        ratio = net / brut
        if ratio > 1.0:
            flags.append({"type": "ratio_impossible", "severity": "high",
                          "detail": f"Net ({net}€) supérieur au brut ({brut}€) — impossible"})
            score += 0.50
        elif ratio < 0.55 or ratio > 0.90:
            flags.append({"type": "ratio_incoherent", "severity": "high",
                          "detail": f"Ratio net/brut = {ratio:.2f} — attendu entre 0.60 et 0.85"})
            score += 0.35
        elif ratio < 0.60 or ratio > 0.85:
            flags.append({"type": "ratio_inhabituel", "severity": "medium",
                          "detail": f"Ratio net/brut = {ratio:.2f} — légèrement inhabituel"})
            score += 0.15

    if net:
        if net < 500:
            flags.append({"type": "montant_bas", "severity": "medium",
                          "detail": f"Net à payer très bas ({net}€)"})
            score += 0.10
        elif net > 20000:
            flags.append({"type": "montant_eleve", "severity": "medium",
                          "detail": f"Net à payer très élevé ({net}€)"})
            score += 0.10

    if siret:
        r = validate_siret(siret)
        if not r["valid"]:
            flags.append({"type": "siret_invalide", "severity": "high", "detail": r["detail"]})
            score += 0.40

    result = _make_result("cross_check", score, flags)
    result["fields_extracted"] = fields
    return result


def validate_avis_imposition(fields: dict) -> dict:
    """Valide un avis d'imposition."""
    flags = []
    score = 0.0

    rfr = fields.get("revenu_fiscal_reference")
    parts = fields.get("nombre_parts")
    annee = fields.get("annee_revenus")
    impot = fields.get("impot_revenu")

    if rfr is not None and rfr < 0:
        flags.append({"type": "rfr_negatif", "severity": "high", "detail": f"RFR négatif ({rfr}€)"})
        score += 0.40

    if parts is not None:
        if parts < 1 or parts > 10:
            flags.append({"type": "parts_incoherentes", "severity": "high",
                          "detail": f"Nombre de parts = {parts} — hors plage"})
            score += 0.30
        if (parts * 4) % 1 != 0:
            flags.append({"type": "parts_invalides", "severity": "high",
                          "detail": f"Nombre de parts = {parts} — pas un multiple de 0.25"})
            score += 0.30

    if annee is not None:
        from datetime import datetime
        current_year = datetime.now().year
        if annee > current_year or annee < 2000:
            flags.append({"type": "annee_incoherente", "severity": "high",
                          "detail": f"Année des revenus = {annee}"})
            score += 0.35

    if rfr and impot and rfr > 0:
        taux = impot / rfr
        if taux > 0.50:
            flags.append({"type": "taux_impot_eleve", "severity": "medium",
                          "detail": f"Taux effectif = {taux:.0%} — très élevé"})
            score += 0.15

    result = _make_result("cross_check", score, flags)
    result["fields_extracted"] = fields
    return result


def validate_facture(fields: dict) -> dict:
    """Valide une facture.

    Vérifications :
    - Cohérence HT × (1 + taux TVA) = TTC
    - Cohérence HT + TVA = TTC
    - SIRET valide
    - IBAN valide (si présent)
    - Numéro de facture présent
    """
    flags = []
    score = 0.0

    ht = fields.get("montant_ht")
    tva = fields.get("montant_tva")
    ttc = fields.get("montant_ttc")
    taux = fields.get("taux_tva")
    siret = fields.get("siret")
    iban = fields.get("iban")
    numero = fields.get("numero_facture")

    # Cohérence HT + TVA = TTC
    if ht and tva and ttc:
        expected_ttc = round(ht + tva, 2)
        diff = abs(expected_ttc - ttc)
        if diff > 1.0:  # Tolérance 1€ pour arrondis
            flags.append({"type": "ttc_incoherent", "severity": "high",
                          "detail": f"HT ({ht}€) + TVA ({tva}€) = {expected_ttc}€ ≠ TTC ({ttc}€)"})
            score += 0.40

    # Cohérence taux TVA
    if ht and taux and ttc and ht > 0:
        expected_ttc = round(ht * (1 + taux / 100), 2)
        diff = abs(expected_ttc - ttc)
        if diff > 1.0:
            flags.append({"type": "taux_tva_incoherent", "severity": "high",
                          "detail": f"HT × {taux}% = {expected_ttc}€ ≠ TTC ({ttc}€)"})
            score += 0.35

    # Taux TVA français valides : 0, 2.1, 5.5, 10, 20
    if taux is not None:
        taux_valides = [0, 2.1, 5.5, 10, 20]
        if not any(abs(taux - t) < 0.5 for t in taux_valides):
            flags.append({"type": "taux_tva_invalide", "severity": "medium",
                          "detail": f"Taux TVA {taux}% — taux français valides : 2.1%, 5.5%, 10%, 20%"})
            score += 0.20

    # Montants négatifs
    for label, val in [("HT", ht), ("TVA", tva), ("TTC", ttc)]:
        if val is not None and val < 0:
            flags.append({"type": "montant_negatif", "severity": "high",
                          "detail": f"Montant {label} négatif ({val}€)"})
            score += 0.35

    # SIRET
    if siret:
        r = validate_siret(siret)
        if not r["valid"]:
            flags.append({"type": "siret_invalide", "severity": "high", "detail": r["detail"]})
            score += 0.40

    # IBAN
    if iban:
        r = validate_iban(iban)
        if not r["valid"]:
            flags.append({"type": "iban_invalide", "severity": "high", "detail": r["detail"]})
            score += 0.35

    # Numéro de facture manquant (obligatoire en France)
    if not numero:
        flags.append({"type": "numero_manquant", "severity": "medium",
                      "detail": "Numéro de facture non détecté — obligatoire en France"})
        score += 0.10

    result = _make_result("cross_check", score, flags)
    result["fields_extracted"] = fields
    return result


def validate_rib(fields: dict) -> dict:
    """Valide un RIB.

    Vérifications :
    - IBAN valide (mod 97)
    - BIC valide
    - Clé RIB cohérente
    """
    flags = []
    score = 0.0

    iban = fields.get("iban")
    bic = fields.get("bic")
    code_banque = fields.get("code_banque")
    code_guichet = fields.get("code_guichet")
    numero_compte = fields.get("numero_compte")
    cle = fields.get("cle_rib")

    # IBAN
    if iban:
        r = validate_iban(iban)
        if not r["valid"]:
            flags.append({"type": "iban_invalide", "severity": "high", "detail": r["detail"]})
            score += 0.50
    else:
        flags.append({"type": "iban_absent", "severity": "medium", "detail": "IBAN non détecté sur le RIB"})
        score += 0.15

    # BIC
    if bic:
        r = validate_bic(bic)
        if not r["valid"]:
            flags.append({"type": "bic_invalide", "severity": "medium", "detail": r["detail"]})
            score += 0.20

    # Clé RIB (vérification par modulo 97)
    if code_banque and code_guichet and numero_compte and cle:
        try:
            concat = int(code_banque + code_guichet + numero_compte + cle)
            if concat % 97 != 0:
                flags.append({"type": "cle_rib_invalide", "severity": "high",
                              "detail": "Clé RIB invalide — ne passe pas le contrôle modulo 97"})
                score += 0.45
        except (ValueError, OverflowError):
            pass

    result = _make_result("cross_check", score, flags)
    result["fields_extracted"] = fields
    return result


def validate_releve_bancaire(fields: dict) -> dict:
    """Valide un relevé bancaire.

    Vérification : solde ancien + crédits - débits ≈ solde nouveau.
    """
    flags = []
    score = 0.0

    ancien = fields.get("solde_ancien")
    nouveau = fields.get("solde_nouveau")
    debits = fields.get("total_debits")
    credits_ = fields.get("total_credits")
    iban = fields.get("iban")

    # Cohérence des soldes
    if ancien is not None and nouveau is not None and debits is not None and credits_ is not None:
        expected = round(ancien + credits_ - debits, 2)
        diff = abs(expected - nouveau)
        if diff > 1.0:  # Tolérance 1€
            flags.append({"type": "solde_incoherent", "severity": "high",
                          "detail": f"Ancien ({ancien}€) + crédits ({credits_}€) - débits ({debits}€) = {expected}€ ≠ nouveau solde ({nouveau}€)"})
            score += 0.50

    # IBAN
    if iban:
        r = validate_iban(iban)
        if not r["valid"]:
            flags.append({"type": "iban_invalide", "severity": "high", "detail": r["detail"]})
            score += 0.35

    result = _make_result("cross_check", score, flags)
    result["fields_extracted"] = fields
    return result


def validate_quittance_loyer(fields: dict) -> dict:
    """Valide une quittance de loyer.

    Vérification : loyer + charges = total.
    """
    flags = []
    score = 0.0

    loyer = fields.get("loyer")
    charges = fields.get("charges")
    total = fields.get("total")

    # Cohérence loyer + charges = total
    if loyer is not None and charges is not None and total is not None:
        expected = round(loyer + charges, 2)
        diff = abs(expected - total)
        if diff > 1.0:
            flags.append({"type": "total_incoherent", "severity": "high",
                          "detail": f"Loyer ({loyer}€) + charges ({charges}€) = {expected}€ ≠ total ({total}€)"})
            score += 0.45

    # Montants réalistes
    if loyer is not None and loyer > 10000:
        flags.append({"type": "loyer_eleve", "severity": "medium",
                      "detail": f"Loyer très élevé ({loyer}€)"})
        score += 0.10

    if total is not None and total <= 0:
        flags.append({"type": "total_negatif", "severity": "high",
                      "detail": f"Total négatif ou nul ({total}€)"})
        score += 0.35

    result = _make_result("cross_check", score, flags)
    result["fields_extracted"] = fields
    return result


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

# Mots-clés pour auto-détection
DOC_KEYWORDS = {
    "bulletin_paie": ["BULLETIN", "PAIE", "SALAIRE BRUT", "NET A PAYER", "NET À PAYER"],
    "avis_imposition": ["AVIS D'IMPOSITION", "REVENU FISCAL", "IMPOT SUR LE REVENU", "IMPÔT SUR LE REVENU"],
    "facture": ["FACTURE", "TOTAL HT", "TOTAL TTC", "MONTANT TTC", "TVA"],
    "rib": ["RELEV", "IDENTIT", "BANCAIRE", "RIB", "IBAN", "BIC", "CODE BANQUE", "CODE GUICHET"],
    "releve_bancaire": ["RELEV", "COMPTE", "SOLDE", "ANCIEN SOLDE", "NOUVEAU SOLDE", "DÉBIT", "CRÉDIT"],
    "quittance_loyer": ["QUITTANCE", "LOYER", "CHARGES", "BAILLEUR", "LOCATAIRE"],
}

# Mapping type → (extracteur, validateur)
DOC_HANDLERS = {
    "bulletin_paie": ("extract_fields_bulletin_paie", validate_bulletin_paie),
    "avis_imposition": ("extract_fields_avis_imposition", validate_avis_imposition),
    "facture": ("extract_fields_facture", validate_facture),
    "rib": ("extract_fields_rib", validate_rib),
    "releve_bancaire": ("extract_fields_releve_bancaire", validate_releve_bancaire),
    "quittance_loyer": ("extract_fields_quittance_loyer", validate_quittance_loyer),
}


def detect_doc_type(text: str) -> str:
    """Auto-détection du type de document par mots-clés."""
    text_upper = text.upper()
    scores = {}

    for doc_type, keywords in DOC_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_upper)
        if count > 0:
            scores[doc_type] = count

    if not scores:
        return "unknown"

    # Retourner le type avec le plus de mots-clés trouvés
    return max(scores, key=scores.get)


def analyze(ocr_result: dict, doc_type: str = "auto") -> dict:
    """Point d'entrée pour la validation croisée.

    Args:
        ocr_result: Résultat de l'OCR.
        doc_type: Type de document ou "auto".
    """
    from . import ocr as ocr_module

    text = ocr_result.get("full_text", "")

    # Auto-détection
    if doc_type == "auto":
        doc_type = detect_doc_type(text)

    if doc_type == "unknown" or doc_type not in DOC_HANDLERS:
        return {
            "analyzer": "cross_check",
            "score": 0.0,
            "verdict": "unknown",
            "detail": "Type de document non reconnu — validation croisée non applicable",
            "doc_type": doc_type,
            "fields_extracted": {},
            "flags": [],
        }

    extractor_name, validator = DOC_HANDLERS[doc_type]
    extractor = getattr(ocr_module, extractor_name)
    fields = extractor(ocr_result)
    result = validator(fields)
    result["doc_type"] = doc_type

    # ── Vérifications externes (SIRET, IBAN, TVA) ────────────────────────
    try:
        from .external_verify import verify_all
        ext = verify_all(fields)
        result["external_verifications"] = ext.get("verifications", {})

        # Si l'API gouv.fr confirme le SIRET, retirer le flag Luhn
        siret_ext = ext.get("verifications", {}).get("siret", {})
        if siret_ext.get("verified") is True:
            result["flags"] = [
                f for f in result["flags"]
                if f.get("type") != "siret_invalide"
            ]
            # Recalculer le score sans le flag SIRET
            score_recalc = 0.0
            for f in result["flags"]:
                if f["severity"] == "high":
                    score_recalc += 0.35
                elif f["severity"] == "medium":
                    score_recalc += 0.15
            result["score"] = round(min(1.0, score_recalc), 4)

        # Ajouter les flags externes
        for flag in ext.get("flags", []):
            if flag["severity"] != "info":
                result["flags"].append(flag)
                result["score"] = round(min(1.0, result["score"] + 0.30), 4)
            else:
                result["flags"].append(flag)

        # Recalculer le verdict
        if result["score"] < 0.15:
            result["verdict"] = "clean"
        elif result["score"] < 0.40:
            result["verdict"] = "suspect"
        else:
            result["verdict"] = "forged"

        # Mettre à jour le détail
        error_flags = [f for f in result["flags"] if f["severity"] not in ("info",)]
        if not error_flags:
            result["detail"] = "✅ Données vérifiées — document cohérent"
        elif result["verdict"] == "forged":
            result["detail"] = f"{len(error_flags)} incohérence(s) majeure(s)"
        elif result["verdict"] == "suspect":
            result["detail"] = f"{len(error_flags)} anomalie(s) détectée(s)"
        else:
            result["detail"] = "Données du document cohérentes"
    except Exception:
        pass

    return result
