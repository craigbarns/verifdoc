"""
Vérifications externes — Module de vérification en temps réel.

Appelle des API publiques pour vérifier que les données extraites
correspondent à des entités réelles :

  - SIRET → API Recherche Entreprises (gouv.fr, gratuit, sans clé)
  - IBAN  → Validation structurelle (mod 97) + vérification code banque FR
  - TVA   → Format TVA intracommunautaire

C'est ça qui fait la différence avec Mindee :
Mindee extrait. VerifDoc vérifie.
"""

from __future__ import annotations

import re
import json
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


# ══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATION SIRET VIA API GOUVERNEMENTALE
# ══════════════════════════════════════════════════════════════════════════════

def verify_siret_online(siret: str) -> dict:
    """Vérifie un SIRET via l'API Recherche Entreprises (gouv.fr).

    Gratuit, sans clé API, sans limite raisonnable.
    Retourne le nom de l'entreprise, l'adresse, le statut (actif/fermé).

    Args:
        siret: Numéro SIRET (14 chiffres).

    Returns:
        dict avec verified, company_name, address, status, detail.
    """
    clean = siret.replace(" ", "")

    if not re.match(r'^\d{14}$', clean):
        return {
            "verified": False,
            "detail": f"Format SIRET invalide ({len(clean)} chiffres)",
            "source": "format_check",
        }

    siren = clean[:9]

    try:
        url = f"https://recherche-entreprises.api.gouv.fr/search?q={siren}&page=1&per_page=1"
        req = Request(url, headers={"User-Agent": "VerifDoc/1.0"})
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError) as e:
        return {
            "verified": None,
            "detail": f"API indisponible — {str(e)[:80]}",
            "source": "api_recherche_entreprises",
        }
    except Exception as e:
        return {
            "verified": None,
            "detail": f"Erreur — {str(e)[:80]}",
            "source": "api_recherche_entreprises",
        }

    results = data.get("results", [])

    if not results:
        return {
            "verified": False,
            "detail": f"SIRET {clean} introuvable dans le répertoire Sirene",
            "source": "api_recherche_entreprises",
            "company_name": None,
            "address": None,
            "status": None,
        }

    company = results[0]
    nom = company.get("nom_complet", "") or company.get("nom_raison_sociale", "")
    siege = company.get("siege", {})
    adresse = siege.get("adresse", "") or ""
    code_postal = siege.get("code_postal", "")
    commune = siege.get("libelle_commune", "")

    # Vérifier si le SIRET spécifique existe dans les établissements
    siret_found = False
    etablissement_status = None
    matching_etablissements = company.get("matching_etablissements", [])

    for etab in matching_etablissements:
        if etab.get("siret") == clean:
            siret_found = True
            etablissement_status = etab.get("etat_administratif", "")
            break

    # Si pas trouvé dans matching, vérifier le siège
    if not siret_found and siege.get("siret") == clean:
        siret_found = True
        etablissement_status = siege.get("etat_administratif", "")

    # Statut de l'unité légale
    etat = company.get("etat_administratif", "A")
    is_active = etat == "A"

    if siret_found:
        if is_active and etablissement_status in ("A", None, ""):
            status_detail = "✅ Entreprise active"
        elif etablissement_status == "F":
            status_detail = "⚠️ Établissement fermé"
        else:
            status_detail = "⚠️ Entreprise cessée"
    else:
        status_detail = "⚠️ SIREN trouvé mais SIRET spécifique non confirmé"

    full_address = f"{adresse} {code_postal} {commune}".strip()

    return {
        "verified": True if siret_found and is_active else False,
        "company_name": nom,
        "siren": siren,
        "siret": clean,
        "address": full_address if full_address else None,
        "status": status_detail,
        "etat_administratif": etat,
        "detail": f"{'✅' if siret_found and is_active else '⚠️'} {nom} — {status_detail}",
        "source": "api_recherche_entreprises",
    }


def verify_siren_online(siren: str) -> dict:
    """Vérifie un SIREN (9 chiffres) via l'API."""
    clean = siren.replace(" ", "")
    if len(clean) == 9:
        # Ajouter un NIC fictif pour utiliser la même logique
        return verify_siret_online(clean + "00000")
    return {"verified": False, "detail": "Format SIREN invalide"}


# ══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATION TVA INTRACOMMUNAUTAIRE
# ══════════════════════════════════════════════════════════════════════════════

def verify_tva_format(tva_number: str) -> dict:
    """Vérifie le format d'un numéro de TVA intracommunautaire français.

    Format FR : FR + 2 chiffres clé + 9 chiffres SIREN.
    La clé = (12 + 3 × (SIREN % 97)) % 97
    """
    clean = tva_number.replace(" ", "").upper()

    match = re.match(r'^FR(\d{2})(\d{9})$', clean)
    if not match:
        return {
            "valid": False,
            "detail": f"Format TVA invalide — attendu FRxx + 9 chiffres",
        }

    cle = int(match.group(1))
    siren = int(match.group(2))

    # Vérification de la clé
    expected_cle = (12 + 3 * (siren % 97)) % 97

    if cle != expected_cle:
        return {
            "valid": False,
            "siren": str(siren).zfill(9),
            "detail": f"Clé TVA invalide — attendu FR{expected_cle:02d}, trouvé FR{cle:02d}",
        }

    return {
        "valid": True,
        "siren": str(siren).zfill(9),
        "detail": f"N° TVA valide — SIREN {str(siren).zfill(9)}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# VÉRIFICATION IBAN AVANCÉE
# ══════════════════════════════════════════════════════════════════════════════

# Codes banque français connus
FR_BANK_CODES = {
    "30002": "Crédit Lyonnais (LCL)",
    "30003": "Société Générale",
    "30004": "BNP Paribas",
    "30006": "BNP Paribas",
    "30007": "Natixis",
    "30027": "CIC",
    "10057": "Banque Postale",
    "10096": "Banque Postale",
    "12506": "Crédit Mutuel",
    "13807": "Crédit Mutuel",
    "14707": "Crédit Mutuel",
    "15589": "Crédit Mutuel",
    "17515": "Crédit Mutuel",
    "18206": "Crédit Mutuel",
    "30047": "CIC",
    "30056": "HSBC France",
    "30066": "CIC",
    "30076": "CIC",
    "30087": "Crédit du Nord",
    "10278": "Crédit Mutuel Arkéa",
    "16275": "Banque Palatine",
    "20041": "Banque Postale",
    "30008": "Société Générale",
}


def verify_iban_advanced(iban: str) -> dict:
    """Vérification IBAN avancée avec identification de la banque."""
    clean = iban.replace(" ", "").upper()

    # Vérification format
    if len(clean) < 15 or len(clean) > 34:
        return {"valid": False, "detail": f"Longueur IBAN invalide ({len(clean)})"}

    if not re.match(r'^[A-Z]{2}\d{2}[A-Z0-9]+$', clean):
        return {"valid": False, "detail": "Format IBAN invalide"}

    # France : 27 caractères
    country = clean[:2]
    if country == "FR" and len(clean) != 27:
        return {"valid": False, "detail": f"IBAN FR doit faire 27 caractères ({len(clean)} trouvés)"}

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

    if not is_valid:
        return {"valid": False, "detail": "Clé de contrôle IBAN invalide (mod 97)"}

    # Identification banque FR
    bank_name = None
    bank_code = None
    if country == "FR" and len(clean) == 27:
        bank_code = clean[4:9]
        bank_name = FR_BANK_CODES.get(bank_code)

    return {
        "valid": True,
        "country": country,
        "bank_code": bank_code,
        "bank_name": bank_name,
        "detail": f"IBAN valide — {bank_name or 'banque ' + (bank_code or 'inconnue')}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE UNIFIÉ
# ══════════════════════════════════════════════════════════════════════════════

def verify_all(fields: dict) -> dict:
    """Vérifie tous les identifiants trouvés dans un document.

    Args:
        fields: dict de champs extraits (siret, iban, tva, etc.)

    Returns:
        dict avec les résultats de chaque vérification.
    """
    results = {}
    flags = []
    score = 0.0

    # SIRET
    siret = fields.get("siret")
    if siret:
        r = verify_siret_online(siret)
        results["siret"] = r
        if r.get("verified") is False:
            flags.append({
                "type": "siret_non_verifie",
                "severity": "high",
                "detail": r["detail"],
            })
            score += 0.40
        elif r.get("verified") is True:
            flags.append({
                "type": "siret_verifie",
                "severity": "info",
                "detail": r["detail"],
            })

    # IBAN
    iban = fields.get("iban")
    if iban:
        r = verify_iban_advanced(iban)
        results["iban"] = r
        if not r["valid"]:
            flags.append({
                "type": "iban_invalide",
                "severity": "high",
                "detail": r["detail"],
            })
            score += 0.35
        else:
            flags.append({
                "type": "iban_verifie",
                "severity": "info",
                "detail": r["detail"],
            })

    # TVA intracommunautaire
    tva = fields.get("tva_intra")
    if tva:
        r = verify_tva_format(tva)
        results["tva"] = r
        if not r["valid"]:
            flags.append({
                "type": "tva_invalide",
                "severity": "high",
                "detail": r["detail"],
            })
            score += 0.30

    return {
        "verifications": results,
        "flags": flags,
        "score": min(1.0, round(score, 4)),
    }
