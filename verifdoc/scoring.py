"""
Moteur de scoring — Agrège les résultats de tous les analyseurs.

Score final pondéré :
  - ELA           : 30% (manipulation visuelle)
  - Bruit         : 15% (incohérence de texture)
  - Copy-Move     : 15% (zones dupliquées)
  - Métadonnées   : 15% (logiciel suspect, dates)
  - Cross-check   : 25% (cohérence des données métier)
"""

from __future__ import annotations

WEIGHTS = {
    "ela": 0.30,
    "noise": 0.15,
    "copy_move": 0.15,
    "metadata": 0.15,
    "cross_check": 0.25,
}

VERDICT_FR = {
    "clean": "✅ Document authentique",
    "suspect": "⚠️ Document suspect — vérification manuelle recommandée",
    "forged": "🚨 Document falsifié — manipulation détectée",
}


def compute_final_score(results: dict[str, dict]) -> dict:
    """Calcule le score final à partir des résultats des analyseurs.

    Args:
        results: dict {analyzer_name: analyzer_result}.

    Returns:
        dict avec score final, verdict, détail par couche, recommandation.
    """
    weighted_sum = 0.0
    total_weight = 0.0
    layer_details = []

    for name, weight in WEIGHTS.items():
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

    # Normaliser si pas tous les analyseurs ont tourné
    if total_weight > 0:
        final_score = weighted_sum / total_weight
    else:
        final_score = 0.0

    # Déterminer le verdict global
    if final_score < 0.15:
        verdict = "clean"
    elif final_score < 0.40:
        verdict = "suspect"
    else:
        verdict = "forged"

    # Veto : si un seul analyseur dit "forged" avec score > 0.7, on passe en suspect minimum
    for name, res in results.items():
        if res.get("verdict") == "forged" and res.get("score", 0) > 0.7:
            if verdict == "clean":
                verdict = "suspect"
                final_score = max(final_score, 0.20)

    # Recommandation
    if verdict == "clean":
        recommendation = "Le document semble authentique. Aucune action requise."
    elif verdict == "suspect":
        # Identifier les couches suspectes
        suspect_layers = [d["layer"] for d in layer_details if d["verdict"] in ("suspect", "forged")]
        recommendation = (
            f"Anomalies détectées dans : {', '.join(suspect_layers)}. "
            "Vérification manuelle recommandée avant acceptation."
        )
    else:
        forged_layers = [d["layer"] for d in layer_details if d["verdict"] == "forged"]
        recommendation = (
            f"Manipulation détectée dans : {', '.join(forged_layers)}. "
            "Refuser le document et demander un original."
        )

    return {
        "final_score": round(final_score, 3),
        "score_100": round(final_score * 100, 1),
        "verdict": verdict,
        "verdict_fr": VERDICT_FR.get(verdict, verdict),
        "recommendation": recommendation,
        "layers": layer_details,
        "analyzers_run": len(layer_details),
    }
