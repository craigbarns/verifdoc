"""
Moteur de scoring — Agrège les résultats de tous les analyseurs.

Wording produit : niveaux de risque (faible / modéré / élevé) + prochaine étape.
"""

from __future__ import annotations

WEIGHTS = {
    "ela": 0.22,
    "noise": 0.12,
    "copy_move": 0.12,
    "metadata": 0.12,
    "cross_check": 0.22,
    "ai_analysis": 0.20,
}

# Poids adaptés par type de document — le cross_check est pondéré plus
# fortement quand la validation métier est déterminante (bulletins, factures).
# L'IA apporte 0.25 pour les types où l'analyse sémantique est clé.
WEIGHTS_BY_DOCTYPE = {
    "bulletin_paie": {
        "ela": 0.10,
        "noise": 0.08,
        "copy_move": 0.08,
        "metadata": 0.14,
        "cross_check": 0.35,
        "ai_analysis": 0.25,
    },
    "avis_imposition": {
        "ela": 0.15,
        "noise": 0.08,
        "copy_move": 0.08,
        "metadata": 0.19,
        "cross_check": 0.25,
        "ai_analysis": 0.25,
    },
    "facture": {
        "ela": 0.10,
        "noise": 0.08,
        "copy_move": 0.08,
        "metadata": 0.19,
        "cross_check": 0.30,
        "ai_analysis": 0.25,
    },
    "rib": {
        "ela": 0.15,
        "noise": 0.08,
        "copy_move": 0.08,
        "metadata": 0.19,
        "cross_check": 0.25,
        "ai_analysis": 0.25,
    },
    "releve_bancaire": {
        "ela": 0.15,
        "noise": 0.08,
        "copy_move": 0.08,
        "metadata": 0.19,
        "cross_check": 0.25,
        "ai_analysis": 0.25,
    },
    "quittance_loyer": {
        "ela": 0.15,
        "noise": 0.10,
        "copy_move": 0.08,
        "metadata": 0.15,
        "cross_check": 0.27,
        "ai_analysis": 0.25,
    },
}

# Verdicts internes (clean / suspect / forged) — inchangés pour la logique & CSS
RISK_COPY = {
    "clean": {
        "title": "Risque faible",
        "next_step": "Validation possible",
        "level": "low",
    },
    "suspect": {
        "title": "Risque modéré",
        "next_step": "Revue manuelle recommandée",
        "level": "moderate",
    },
    "forged": {
        "title": "Risque élevé",
        "next_step": "Investigation requise",
        "level": "high",
    },
}

DOC_TYPE_LABELS = {
    "bulletin_paie": "Bulletin de paie",
    "avis_imposition": "Avis d’imposition",
    "facture": "Facture",
    "rib": "RIB",
    "releve_bancaire": "Relevé bancaire",
    "quittance_loyer": "Quittance de loyer",
    "unknown": "Non identifié",
    "auto": "—",
}


def _business_verification_summary(cross: dict) -> dict[str, dict]:
    """Synthèse lisible pour l’UI métier (SIRET, entreprise, IBAN, TVA)."""
    out = {
        "siret": {"status": "absent", "label": "Non détecté sur le document"},
        "entreprise": {"status": "n/a", "label": "Pas de SIRET extrait — entreprise non vérifiable"},
        "iban": {"status": "absent", "label": "Non extrait — contrôle structurel impossible"},
        "tva": {"status": "absent", "label": "Non détectée sur le document"},
    }
    if not cross or cross.get("verdict") == "skipped":
        out["siret"] = {"status": "skipped", "label": "Couche métier non exécutée (OCR désactivé)"}
        out["iban"] = {"status": "skipped", "label": "—"}
        out["tva"] = {"status": "skipped", "label": "—"}
        return out

    ext = cross.get("external_verifications") or {}
    fields = cross.get("fields_extracted") or {}

    # SIRET
    if fields.get("siret"):
        sv = ext.get("siret") or {}
        if sv.get("verified") is True:
            out["siret"] = {"status": "ok", "label": "Numéro valide (référentiel officiel)"}
            nom = sv.get("company_name") or ""
            out["entreprise"] = {
                "status": "found",
                "label": nom.strip() or "Entreprise trouvée",
                "detail": sv.get("address") or "",
                "status_line": sv.get("status") or "",
            }
        elif sv.get("verified") is False:
            out["siret"] = {"status": "invalid", "label": "Invalide ou introuvable (référentiel)"}
            out["entreprise"] = {"status": "not_found", "label": "Aucune entreprise active associée"}
        elif sv.get("verified") is None:
            luhn_invalid = any(
                f.get("type") == "siret_invalide" for f in cross.get("flags", [])
            )
            if luhn_invalid:
                out["siret"] = {
                    "status": "invalid",
                    "label": "Clé de contrôle SIRET invalide (Luhn) — registre non joignable",
                }
                out["entreprise"] = {
                    "status": "warn",
                    "label": "Exiger un justificatif ou un autre SIRET avant validation",
                }
            else:
                out["siret"] = {"status": "unavailable", "label": "Vérification registre indisponible (API)"}
                out["entreprise"] = {"status": "unknown", "label": "Non vérifiable pour l’instant"}
        else:
            out["siret"] = {"status": "unknown", "label": "État de vérification inconnu"}
    else:
        for f in cross.get("flags", []):
            if f.get("type") == "siret_invalide":
                out["siret"] = {"status": "invalid", "label": "Format ou clé incohérente"}
                break

    # IBAN
    if fields.get("iban"):
        iv = ext.get("iban") or {}
        if iv.get("valid") is True:
            out["iban"] = {"status": "ok", "label": "Structure cohérente (modulo 97)"}
            if iv.get("bank_name"):
                out["iban"]["detail"] = iv["bank_name"]
        elif iv.get("valid") is False:
            out["iban"] = {"status": "invalid", "label": "Structure incohérente"}
        else:
            out["iban"] = {"status": "unverifiable", "label": "Non vérifiable"}
    else:
        out["iban"] = {"status": "absent", "label": "Non extrait du document"}

    # TVA intracom
    if fields.get("tva_intra"):
        tv = ext.get("tva") or {}
        if tv.get("valid") is True:
            out["tva"] = {"status": "ok", "label": "Cohérente avec le SIREN attendu"}
        elif tv.get("valid") is False:
            out["tva"] = {"status": "invalid", "label": "Incohérente (clé TVA)"}
        else:
            out["tva"] = {"status": "unverifiable", "label": "Non vérifiable"}
    else:
        out["tva"] = {"status": "absent", "label": "Non détectée sur le document"}

    return out


def _executive_anomalies_line(results: dict, verdict: str) -> str:
    """Une ligne business : principales anomalies ou absence signalée."""
    cross = results.get("cross_check") or {}
    if cross.get("verdict") == "skipped":
        return "Couche métier non activée — activez l’OCR pour SIRET, IBAN et cohérence des montants."

    highs = [f for f in cross.get("flags", []) if f.get("severity") == "high"]
    if highs:
        types = list({f.get("type", "alerte") for f in highs[:4]})
        return f"{len(highs)} alerte(s) majeure(s) : {', '.join(types[:3])}{'…' if len(types) > 3 else ''}."

    mids = [f for f in cross.get("flags", []) if f.get("severity") == "medium"]
    if mids and verdict != "clean":
        return f"{len(mids)} point(s) à clarifier en revue manuelle."

    forensics_hot = []
    for name, res in results.items():
        if name in WEIGHTS and res.get("verdict") in ("suspect", "forged"):
            forensics_hot.append(name)
    if forensics_hot:
        return f"Signaux forensiques sur : {', '.join(forensics_hot)} — croiser avec la couche métier."

    if verdict == "clean":
        return "Aucune anomalie majeure détectée sur les couches actives — rester prudent sur les originaux."
    return "Analyse terminée — consulter le détail par couche ci-dessous."


def compute_final_score(results: dict[str, dict], doc_type: str | None = None) -> dict:
    """Calcule le score final, le niveau de risque produit et la synthèse métier.

    Args:
        results: Résultats des analyseurs.
        doc_type: Type de document détecté (optionnel). Si fourni et reconnu,
                  les poids par type sont appliqués pour un scoring plus pertinent.
    """
    # Sélectionner les poids adaptés au type de document
    if doc_type is None:
        doc_type = (results.get("cross_check") or {}).get("doc_type")

    # Si l'IA a identifié le type avec haute confiance, utiliser ce type pour les poids aussi
    _ai = results.get("ai_analysis") or {}
    _ai_dt = _ai.get("ai_doc_type")
    _ai_dt_conf = _ai.get("ai_doc_type_confidence", 0) or 0
    if _ai_dt and float(_ai_dt_conf) >= 0.80 and _ai_dt in WEIGHTS_BY_DOCTYPE:
        if doc_type not in WEIGHTS_BY_DOCTYPE or float(_ai_dt_conf) >= 0.90:
            doc_type = _ai_dt

    active_weights = WEIGHTS_BY_DOCTYPE.get(doc_type, WEIGHTS)

    weighted_sum = 0.0
    total_weight = 0.0
    layer_details = []

    for name, weight in active_weights.items():
        if name in results and results[name].get("score") is not None:
            analyzer_score = results[name]["score"]
            weighted_sum += analyzer_score * weight
            total_weight += weight

            layer_details.append({
                "layer": name,
                "score": analyzer_score,
                "weight": weight,
                "verdict": results[name].get("verdict", "unknown"),
                "detail": results[name].get("detail", ""),
            })

    if total_weight > 0:
        final_score = weighted_sum / total_weight
    else:
        final_score = 0.0

    if final_score < 0.15:
        verdict = "clean"
    elif final_score < 0.40:
        verdict = "suspect"
    else:
        verdict = "forged"

    for name, res in results.items():
        if res.get("verdict") == "forged" and res.get("score", 0) > 0.7:
            if verdict == "clean":
                verdict = "suspect"
                final_score = max(final_score, 0.20)

    # Alerte métier majeure → au minimum Risque modéré (cohérent avec le résumé exécutif)
    _critical_business = {
        "siret_invalide",
        "siret_non_verifie",
        "iban_invalide",
        "ttc_incoherent",
        "taux_tva_incoherent",
        "cle_rib_invalide",
        "ratio_impossible",
        "tva_invalide",
        "solde_incoherent",
        "total_incoherent",
        "bulletin_non_verifiable",
        "bulletin_incomplet",
        "bulletin_peu_extractible",
        "type_document_inconnu",
    }
    cross_flags = results.get("cross_check", {}).get("flags") or []
    if any(
        f.get("severity") == "high" and f.get("type") in _critical_business
        for f in cross_flags
    ):
        if verdict == "clean":
            verdict = "suspect"
            final_score = max(final_score, 0.24)
        elif verdict == "suspect":
            final_score = max(final_score, 0.26)

    rc = RISK_COPY[verdict]
    risk_title = rc["title"]
    risk_next_step = rc["next_step"]
    risk_level = rc["level"]

    if verdict == "clean":
        recommendation = (
            "Indicateurs globaux favorables. " + risk_next_step.lower() + " selon vos procédures internes."
        )
    elif verdict == "suspect":
        suspect_layers = [d["layer"] for d in layer_details if d["verdict"] in ("suspect", "forged")]
        recommendation = (
            f"Signaux hétérogènes ({', '.join(suspect_layers) or 'plusieurs couches'}). "
            f"{risk_next_step}."
        )
    else:
        forged_layers = [d["layer"] for d in layer_details if d["verdict"] == "forged"]
        recommendation = (
            f"Plusieurs indicateurs convergent ({', '.join(forged_layers) or 'analyse globale'}). "
            f"{risk_next_step}."
        )

    max_layers = 6
    run_count = len(layer_details)
    confidence = min(100.0, round(100 * run_count / max_layers))
    if results.get("cross_check", {}).get("verdict") == "unknown":
        confidence = max(0, confidence - 15)
    if any(results.get(k, {}).get("error") for k in WEIGHTS):
        confidence = max(0, confidence - 10)

    cross = results.get("cross_check") or {}
    doc_type_key = cross.get("doc_type") or "unknown"

    # Données IA enrichies
    ai_res = results.get("ai_analysis") or {}
    ai_available = ai_res.get("ai_available", False)
    ai_explanation = ai_res.get("ai_explanation", "")
    ai_doc_type = ai_res.get("ai_doc_type")
    ai_doc_type_confidence = ai_res.get("ai_doc_type_confidence", 0)
    ai_confidence = ai_res.get("ai_confidence", 0)

    # L'IA override le type de document quand sa confiance dépasse 80%
    # et que le type OCR est inconnu, "auto", ou différent
    if (
        ai_doc_type
        and ai_doc_type in DOC_TYPE_LABELS
        and ai_doc_type_confidence is not None
        and float(ai_doc_type_confidence) >= 0.80
        and doc_type_key in ("unknown", "auto", "autre", "inconnu")
    ):
        doc_type_key = ai_doc_type

    # Aussi override si l'IA a une confiance ≥ 90% et le type OCR est différent
    # (l'IA voit le document entier, elle est plus fiable pour la classification)
    if (
        ai_doc_type
        and ai_doc_type in DOC_TYPE_LABELS
        and ai_doc_type_confidence is not None
        and float(ai_doc_type_confidence) >= 0.90
        and doc_type_key != ai_doc_type
    ):
        doc_type_key = ai_doc_type

    doc_type_label = DOC_TYPE_LABELS.get(doc_type_key, doc_type_key)

    business = _business_verification_summary(cross)
    exec_anomalies = _executive_anomalies_line(results, verdict)

    return {
        "final_score": round(final_score, 3),
        "score_100": round(final_score * 100, 1),
        "verdict": verdict,
        "verdict_fr": f"{risk_title} — {risk_next_step}",
        "risk_title": risk_title,
        "risk_next_step": risk_next_step,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "layers": layer_details,
        "analyzers_run": len(layer_details),
        "confidence_percent": round(confidence, 0),
        "doc_type_detected": doc_type_key,
        "doc_type_label": doc_type_label,
        "executive_anomalies": exec_anomalies,
        "business_verification": business,
        "ai_available": ai_available,
        "ai_explanation": ai_explanation,
        "ai_doc_type": ai_doc_type,
        "ai_confidence": ai_confidence,
        "disclaimer": (
            "Indicateur d’aide à la décision — ne constitue pas une preuve juridique. "
            "Confronter à l’original et à un contrôle humain."
        ),
    }
