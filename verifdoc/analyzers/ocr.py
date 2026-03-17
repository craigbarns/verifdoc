"""
Extraction OCR — Couche 5a du pipeline VerifDoc.

Stratégie d'extraction :
  1. PDF natif → PyMuPDF (instantané, 0 RAM supplémentaire)
  2. Image scannée → EasyOCR (fallback, plus lent)

Types supportés :
  - Bulletin de paie
  - Avis d'imposition
  - Facture
  - RIB / Relevé d'identité bancaire
  - Relevé bancaire
  - Quittance de loyer
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def extract_text_from_pdf(pdf_path: str | Path) -> dict:
    """Extrait le texte d'un PDF natif via PyMuPDF (ultra rapide)."""
    try:
        import fitz
    except ImportError:
        return {"full_text": "", "words": [], "avg_confidence": 0.0, "error": "PyMuPDF non installé"}

    try:
        doc = fitz.open(str(pdf_path))
        full_text = ""
        words = []
        for page in doc:
            full_text += page.get_text() + "\n"
            for w in page.get_text("words"):
                words.append({
                    "text": w[4].strip(),
                    "confidence": 0.99,
                    "bbox": [[int(w[0]), int(w[1])], [int(w[2]), int(w[1])],
                             [int(w[2]), int(w[3])], [int(w[0]), int(w[3])]],
                })
        doc.close()

        if full_text.strip():
            return {
                "full_text": full_text.strip(),
                "words": words,
                "avg_confidence": 0.99,
                "engine": "pymupdf",
            }
    except Exception:
        pass

    return {"full_text": "", "words": [], "avg_confidence": 0.0}


def extract_text_from_image(source: Image.Image, languages: list[str] | None = None) -> dict:
    """Extrait le texte d'une image via EasyOCR (fallback)."""
    if languages is None:
        languages = ["fr"]

    try:
        import easyocr
    except ImportError:
        return {"full_text": "", "words": [], "avg_confidence": 0.0, "error": "EasyOCR non installé"}

    try:
        # Cache le reader
        if not hasattr(extract_text_from_image, "_reader"):
            extract_text_from_image._reader = easyocr.Reader(languages, gpu=False, verbose=False)
        reader = extract_text_from_image._reader

        img_array = np.array(source.convert("RGB"))
        # Réduire la taille pour économiser la RAM
        h, w = img_array.shape[:2]
        if max(h, w) > 1500:
            scale = 1500 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_array = np.array(source.resize((new_w, new_h)).convert("RGB"))

        results = reader.readtext(img_array)
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
            "engine": "easyocr",
        }
    except Exception as e:
        return {"full_text": "", "words": [], "avg_confidence": 0.0, "error": str(e)}


def extract_text(
    source: str | Path | Image.Image,
    languages: list[str] | None = None,
    gpu: bool = False,
    pdf_path: str | Path | None = None,
) -> dict:
    """Point d'entrée principal — choisit la meilleure méthode.

    1. Si pdf_path fourni → extraction PyMuPDF (instantanée)
    2. Sinon → EasyOCR sur l'image
    """
    # Essayer PyMuPDF d'abord si on a un PDF
    if pdf_path:
        result = extract_text_from_pdf(pdf_path)
        if result["full_text"]:
            return result

    # Fallback : EasyOCR sur l'image
    if isinstance(source, Image.Image):
        return extract_text_from_image(source, languages)
    elif isinstance(source, (str, Path)):
        img = Image.open(str(source)).convert("RGB")
        return extract_text_from_image(img, languages)

    return {"full_text": "", "words": [], "avg_confidence": 0.0, "error": "Source non supportée"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_amount(text: str, patterns: list[str]) -> float | None:
    """Cherche un montant dans le texte via une liste de regex."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount = match.group(1).replace(" ", "").replace(",", ".")
            try:
                return float(amount)
            except ValueError:
                continue
    return None


def _find_siret(text: str) -> str | None:
    """Cherche un numéro SIRET (14 chiffres) dans le texte."""
    match = re.search(r'(?:SIRET|siret)\s*[:\s]*(\d[\d\s]{12,16}\d)', text)
    if match:
        clean = match.group(1).replace(" ", "")
        if len(clean) == 14:
            return clean
    # Fallback : chercher 14 chiffres groupés
    match = re.search(r'\b(\d{3}\s?\d{3}\s?\d{3}\s?\d{5})\b', text)
    if match:
        return match.group(1).replace(" ", "")
    return None


def _find_iban(text: str) -> str | None:
    """Cherche un IBAN dans le texte."""
    match = re.search(
        r'\b([A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3})\b',
        text.upper(),
    )
    if match:
        return match.group(1).replace(" ", "")
    # Format FR plus souple
    match = re.search(
        r'(?:IBAN|iban)\s*[:\s]*([A-Z]{2}\d[\d\s]{20,30})',
        text.upper(),
    )
    if match:
        clean = match.group(1).replace(" ", "")
        if 15 <= len(clean) <= 34:
            return clean
    return None


def _find_bic(text: str) -> str | None:
    """Cherche un code BIC/SWIFT dans le texte."""
    match = re.search(r'\b([A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?)\b', text.upper())
    if match:
        bic = match.group(1)
        if 8 <= len(bic) <= 11:
            return bic
    return None


# ── Extracteurs par type de document ─────────────────────────────────────────

def extract_fields_bulletin_paie(ocr_result: dict) -> dict:
    """Extrait les champs clés d'un bulletin de paie français."""
    text = ocr_result.get("full_text", "")
    text_upper = text.upper()
    fields = {}

    # SIRET
    siret = _find_siret(text)
    if siret:
        fields["siret"] = siret

    # Salaire net
    net = _find_amount(text_upper, [
        r'NET\s*[AÀ]\s*PAYER\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'NET\s*IMPOSABLE\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'NET\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
    ])
    if net:
        fields["net_a_payer"] = net

    # Salaire brut
    brut = _find_amount(text_upper, [
        r'SALAIRE\s*BRUT\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'BRUT\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
    ])
    if brut:
        fields["salaire_brut"] = brut

    # Période
    mois_match = re.search(
        r'(JANVIER|F[EÉ]VRIER|MARS|AVRIL|MAI|JUIN|JUILLET|'
        r'AO[UÛ]T|SEPTEMBRE|OCTOBRE|NOVEMBRE|D[EÉ]CEMBRE)\s*(\d{4})',
        text_upper,
    )
    if mois_match:
        fields["periode"] = f"{mois_match.group(1)} {mois_match.group(2)}"

    return fields


def extract_fields_avis_imposition(ocr_result: dict) -> dict:
    """Extrait les champs clés d'un avis d'imposition français."""
    text = ocr_result.get("full_text", "")
    text_upper = text.upper()
    fields = {}

    # Revenu fiscal de référence
    rfr = _find_amount(text_upper, [
        r'REVENU\s*FISCAL\s*DE\s*R[EÉ]F[EÉ]RENCE\s*[:\s]*(\d[\d\s]*)',
    ])
    if rfr:
        fields["revenu_fiscal_reference"] = int(rfr)

    # Nombre de parts
    parts_match = re.search(r'NOMBRE\s*DE\s*PARTS?\s*[:\s]*([\d,\.]+)', text_upper)
    if parts_match:
        try:
            fields["nombre_parts"] = float(parts_match.group(1).replace(",", "."))
        except ValueError:
            pass

    # Impôt
    impot = _find_amount(text_upper, [
        r'IMP[ÔO]T\s*SUR\s*LE\s*REVENU\s*NET\s*[:\s]*(\d[\d\s]*)',
        r'IMP[ÔO]T\s*NET\s*[:\s]*(\d[\d\s]*)',
    ])
    if impot:
        fields["impot_revenu"] = int(impot)

    # Année
    annee_match = re.search(r'REVENUS?\s*(\d{4})', text_upper)
    if annee_match:
        fields["annee_revenus"] = int(annee_match.group(1))

    return fields


def extract_fields_facture(ocr_result: dict) -> dict:
    """Extrait les champs clés d'une facture française."""
    text = ocr_result.get("full_text", "")
    text_upper = text.upper()
    fields = {}

    # SIRET émetteur
    siret = _find_siret(text)
    if siret:
        fields["siret"] = siret

    # Numéro de facture
    num_match = re.search(
        r'FACTURE\s*N[°O\.]?\s*[:\s]*([A-Z0-9\-/]+)',
        text_upper,
    )
    if num_match:
        fields["numero_facture"] = num_match.group(1).strip()

    # Montant HT
    ht = _find_amount(text_upper, [
        r'TOTAL\s*HT\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'MONTANT\s*HT\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'HT\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
    ])
    if ht:
        fields["montant_ht"] = ht

    # TVA
    tva = _find_amount(text_upper, [
        r'TVA\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'MONTANT\s*TVA\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
    ])
    if tva:
        fields["montant_tva"] = tva

    # Taux TVA
    taux_match = re.search(r'TVA\s*[:\s]*(\d{1,2}[.,]?\d{0,2})\s*%', text_upper)
    if taux_match:
        try:
            fields["taux_tva"] = float(taux_match.group(1).replace(",", "."))
        except ValueError:
            pass

    # Montant TTC
    ttc = _find_amount(text_upper, [
        r'TOTAL\s*TTC\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'MONTANT\s*TTC\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'NET\s*[AÀ]\s*PAYER\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'TTC\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
    ])
    if ttc:
        fields["montant_ttc"] = ttc

    # Date facture
    date_match = re.search(
        r'DATE\s*[:\s]*(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})',
        text_upper,
    )
    if date_match:
        fields["date_facture"] = date_match.group(1)

    # IBAN
    iban = _find_iban(text)
    if iban:
        fields["iban"] = iban

    # TVA intracommunautaire
    tva_intra_match = re.search(r'(?:TVA|N[°O]?\s*TVA)\s*(?:INTRA\s*(?:COMMUNAUTAIRE)?|INTRACOMMUNAUTAIRE)\s*[:\s]*(FR\s?\d{2}\s?\d{9})', text_upper)
    if tva_intra_match:
        fields["tva_intra"] = tva_intra_match.group(1).replace(" ", "")
    else:
        # Fallback : chercher FR + 11 chiffres
        tva_match = re.search(r'\b(FR\s?\d{2}\s?\d{9})\b', text_upper)
        if tva_match:
            fields["tva_intra"] = tva_match.group(1).replace(" ", "")

    return fields


def extract_fields_rib(ocr_result: dict) -> dict:
    """Extrait les champs clés d'un RIB."""
    text = ocr_result.get("full_text", "")
    text_upper = text.upper()
    fields = {}

    # IBAN
    iban = _find_iban(text)
    if iban:
        fields["iban"] = iban

    # BIC
    bic = _find_bic(text)
    if bic:
        fields["bic"] = bic

    # Code banque (5 chiffres)
    banque_match = re.search(r'CODE\s*BANQUE\s*[:\s]*(\d{5})', text_upper)
    if banque_match:
        fields["code_banque"] = banque_match.group(1)

    # Code guichet (5 chiffres)
    guichet_match = re.search(r'CODE\s*GUICHET\s*[:\s]*(\d{5})', text_upper)
    if guichet_match:
        fields["code_guichet"] = guichet_match.group(1)

    # Numéro de compte
    compte_match = re.search(r'N[°O\.]?\s*COMPTE\s*[:\s]*(\d{11})', text_upper)
    if compte_match:
        fields["numero_compte"] = compte_match.group(1)

    # Clé RIB (2 chiffres)
    cle_match = re.search(r'CL[EÉ]\s*(?:RIB)?\s*[:\s]*(\d{2})', text_upper)
    if cle_match:
        fields["cle_rib"] = cle_match.group(1)

    # Titulaire
    tit_match = re.search(r'TITULAIRE\s*[:\s]*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if tit_match:
        fields["titulaire"] = tit_match.group(1).strip()

    return fields


def extract_fields_releve_bancaire(ocr_result: dict) -> dict:
    """Extrait les champs clés d'un relevé bancaire."""
    text = ocr_result.get("full_text", "")
    text_upper = text.upper()
    fields = {}

    # IBAN
    iban = _find_iban(text)
    if iban:
        fields["iban"] = iban

    # Solde ancien / initial
    solde_ancien = _find_amount(text_upper, [
        r'ANCIEN\s*SOLDE\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'SOLDE\s*(?:AU|PR[EÉ]C[EÉ]DENT|INITIAL|D[EÉ]BITEUR|CR[EÉ]DITEUR)\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
    ])
    if solde_ancien:
        fields["solde_ancien"] = solde_ancien

    # Nouveau solde
    solde_nouveau = _find_amount(text_upper, [
        r'NOUVEAU\s*SOLDE\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'SOLDE\s*(?:AU|FINAL|ACTUEL)\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
    ])
    if solde_nouveau:
        fields["solde_nouveau"] = solde_nouveau

    # Total débits
    debits = _find_amount(text_upper, [
        r'TOTAL\s*(?:DES\s*)?D[EÉ]BITS?\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
    ])
    if debits:
        fields["total_debits"] = debits

    # Total crédits
    credits_ = _find_amount(text_upper, [
        r'TOTAL\s*(?:DES\s*)?CR[EÉ]DITS?\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
    ])
    if credits_:
        fields["total_credits"] = credits_

    # Période
    periode_match = re.search(
        r'(?:DU|P[EÉ]RIODE)\s*[:\s]*(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})\s*(?:AU)\s*(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})',
        text_upper,
    )
    if periode_match:
        fields["date_debut"] = periode_match.group(1)
        fields["date_fin"] = periode_match.group(2)

    return fields


def extract_fields_quittance_loyer(ocr_result: dict) -> dict:
    """Extrait les champs clés d'une quittance de loyer."""
    text = ocr_result.get("full_text", "")
    text_upper = text.upper()
    fields = {}

    # Montant loyer
    loyer = _find_amount(text_upper, [
        r'LOYER\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'LOYER\s*(?:MENSUEL|PRINCIPAL)\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
    ])
    if loyer:
        fields["loyer"] = loyer

    # Charges
    charges = _find_amount(text_upper, [
        r'CHARGES?\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'PROVISION\s*(?:SUR|POUR)\s*CHARGES?\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
    ])
    if charges:
        fields["charges"] = charges

    # Total
    total = _find_amount(text_upper, [
        r'TOTAL\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
        r'SOMME\s*(?:DE|TOTALE)?\s*[:\s]*(\d[\d\s]*[.,]\d{2})',
    ])
    if total:
        fields["total"] = total

    # Période
    periode_match = re.search(
        r'(?:MOIS|P[EÉ]RIODE)\s*(?:DE|DU|D\')?\s*'
        r'(JANVIER|F[EÉ]VRIER|MARS|AVRIL|MAI|JUIN|JUILLET|'
        r'AO[UÛ]T|SEPTEMBRE|OCTOBRE|NOVEMBRE|D[EÉ]CEMBRE)\s*(\d{4})',
        text_upper,
    )
    if periode_match:
        fields["periode"] = f"{periode_match.group(1)} {periode_match.group(2)}"

    # Adresse du bien
    adresse_match = re.search(
        r'(?:SITU[EÉ]|ADRESSE|BIEN|LOGEMENT)\s*[:\s]*(.+?)(?:\n|$)',
        text, re.IGNORECASE,
    )
    if adresse_match:
        fields["adresse_bien"] = adresse_match.group(1).strip()[:100]

    return fields
