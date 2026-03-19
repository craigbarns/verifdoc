"""
Forge Factory — Génération de documents synthétiques clean et falsifiés.

Produit des factures, bulletins de paie et quittances de loyer réalistes
avec 6 types de falsification pour benchmarker le pipeline VerifDoc.
"""

from __future__ import annotations

import io
import random
import string
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Dataclass résultat ──────────────────────────────────────────────────────

@dataclass
class SyntheticSample:
    image: Image.Image
    label: int                        # 0 = clean, 1 = forged
    forgery_type: str                 # "none", "amount_edit", "text_replace", ...
    forgery_zone: tuple | None        # (x, y, w, h) or None
    doc_type: str                     # "facture", "bulletin_paie", "quittance"
    ground_truth: dict = field(default_factory=dict)


# ── Données réalistes françaises ────────────────────────────────────────────

_COMPANY_NAMES = [
    "ACME Solutions SAS", "Dupont & Fils SARL", "Tech Innovation EURL",
    "Nexus Consulting SA", "Horizon Digital SAS", "BioVert France SARL",
    "Infra Services EURL", "Méditerranée Import SA", "Atlas Logistique SAS",
    "Voltaire Engineering SARL", "Provence Matériaux SAS", "DataFlow Systems EURL",
    "Côte d'Azur Tourisme SA", "Euro Précision SAS", "LumiNova SARL",
    "Groupe Bélier SAS", "Alpha Constructions SA", "Zenith Telecom EURL",
    "Pacific Distribution SAS", "Cristal Éditions SARL",
]

_ADDRESSES = [
    "12 Rue de la Paix, 75002 Paris",
    "8 Boulevard Haussmann, 75009 Paris",
    "45 Avenue Jean Jaurès, 69007 Lyon",
    "3 Rue du Vieux Port, 13001 Marseille",
    "17 Place de la Liberté, 33000 Bordeaux",
    "22 Rue Nationale, 59000 Lille",
    "5 Cours Mirabeau, 13100 Aix-en-Provence",
    "31 Rue de la République, 69002 Lyon",
    "9 Boulevard Victor Hugo, 06000 Nice",
    "14 Rue des Carmes, 31000 Toulouse",
]

_FIRST_NAMES = ["Jean", "Marie", "Pierre", "Sophie", "Nicolas", "Isabelle",
                 "François", "Claire", "Laurent", "Nathalie", "Philippe", "Catherine"]
_LAST_NAMES = ["Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard",
                "Petit", "Durand", "Leroy", "Moreau", "Simon", "Laurent"]

_LINE_ITEMS_FACTURE = [
    "Prestation de conseil", "Développement logiciel", "Maintenance annuelle",
    "Audit de sécurité", "Formation équipe", "Installation matériel",
    "Support technique", "Licence logicielle", "Hébergement cloud",
    "Migration données", "Design graphique", "Rédaction technique",
]

_LINE_ITEMS_PAIE = [
    ("Salaire de base", None),
    ("Prime d'ancienneté", 0.05),
    ("Prime de transport", None),
    ("Heures supplémentaires", None),
]

_MONTHS_FR = [
    "JANVIER", "FÉVRIER", "MARS", "AVRIL", "MAI", "JUIN",
    "JUILLET", "AOÛT", "SEPTEMBRE", "OCTOBRE", "NOVEMBRE", "DÉCEMBRE",
]


# ── Utilitaires ─────────────────────────────────────────────────────────────

def _generate_valid_siret(rng: random.Random) -> str:
    """Génère un SIRET valide (passe le contrôle Luhn)."""
    digits = [rng.randint(0, 9) for _ in range(13)]
    # Calcul Luhn pour le 14e chiffre
    total = 0
    for i, d in enumerate(digits):
        n = d
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    check = (10 - (total % 10)) % 10
    digits.append(check)
    raw = "".join(str(d) for d in digits)
    return f"{raw[:3]} {raw[3:6]} {raw[6:9]} {raw[9:]}"


def _generate_invalid_siret(rng: random.Random) -> str:
    """Génère un SIRET invalide (échoue au contrôle Luhn)."""
    siret = _generate_valid_siret(rng)
    clean = siret.replace(" ", "")
    # Modifier le dernier chiffre pour casser Luhn
    last = int(clean[-1])
    bad = (last + rng.randint(1, 9)) % 10
    clean = clean[:-1] + str(bad)
    return f"{clean[:3]} {clean[3:6]} {clean[6:9]} {clean[9:]}"


def _generate_valid_iban(rng: random.Random) -> str:
    """Génère un IBAN français valide (mod 97)."""
    bank_code = "".join([str(rng.randint(0, 9)) for _ in range(5)])
    branch_code = "".join([str(rng.randint(0, 9)) for _ in range(5)])
    account = "".join([str(rng.randint(0, 9)) for _ in range(11)])
    rib_key = "".join([str(rng.randint(0, 9)) for _ in range(2)])
    bban = bank_code + branch_code + account + rib_key
    # Calcul clé IBAN
    raw = bban + "FR00"
    numeric = ""
    for c in raw:
        if c.isdigit():
            numeric += c
        else:
            numeric += str(ord(c) - 55)
    remainder = int(numeric) % 97
    check = 98 - remainder
    return f"FR{check:02d} {bank_code} {branch_code} {account} {rib_key}"


def _fmt_amount(amount: float) -> str:
    """Formate un montant à la française : 1 234,56"""
    parts = f"{amount:,.2f}".replace(",", " ").replace(".", ",")
    return parts


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Charge une police ou fallback."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ── Générateurs de documents ────────────────────────────────────────────────

class ForgeFactory:
    """Génère des documents synthétiques clean et falsifiés."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self._font = _get_font(14)
        self._font_sm = _get_font(11)
        self._font_lg = _get_font(18)
        self._font_xl = _get_font(24)

    # ── API publique ────────────────────────────────────────────────────

    def _jpeg_pass(self, img: Image.Image, quality: int = 92) -> Image.Image:
        """Simule une compression JPEG (comme un scan réel)."""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def generate_dataset(
        self,
        n_clean: int = 100,
        n_forged: int = 100,
        doc_types: list[str] | None = None,
    ) -> list[SyntheticSample]:
        """Génère un dataset équilibré de documents clean et falsifiés."""
        doc_types = doc_types or ["facture", "bulletin_paie", "quittance"]
        forgery_types = [
            "amount_edit", "text_replace", "copy_paste",
            "noise_inject", "compression_artifact", "metadata_strip",
        ]
        samples: list[SyntheticSample] = []

        # Documents clean — une seule compression JPEG uniforme
        for i in range(n_clean):
            dt = doc_types[i % len(doc_types)]
            sample = self._generate_clean(dt)
            sample.image = self._jpeg_pass(sample.image, quality=75)
            samples.append(sample)

        # Documents falsifiés — double/triple compression + édition
        for i in range(n_forged):
            dt = doc_types[i % len(doc_types)]
            ft = forgery_types[i % len(forgery_types)]
            clean = self._generate_clean(dt)
            # Première compression (document original)
            clean.image = self._jpeg_pass(clean.image, quality=75)
            # Appliquer la falsification (inclut re-compression locale)
            forged = self._apply_forgery(clean, ft)
            # Triple compression = artefacts ELA
            forged.image = self._jpeg_pass(forged.image, quality=50)
            samples.append(forged)

        self.rng.shuffle(samples)
        return samples

    # ── Génération clean ────────────────────────────────────────────────

    def _generate_clean(self, doc_type: str) -> SyntheticSample:
        if doc_type == "facture":
            return self._gen_facture()
        elif doc_type == "bulletin_paie":
            return self._gen_bulletin()
        elif doc_type == "quittance":
            return self._gen_quittance()
        else:
            return self._gen_facture()

    def _gen_facture(self) -> SyntheticSample:
        """Génère une facture française réaliste."""
        W, H = 850, 1200
        img = Image.new("RGB", (W, H), "#FFFFFF")
        draw = ImageDraw.Draw(img)

        company = self.rng.choice(_COMPANY_NAMES)
        address = self.rng.choice(_ADDRESSES)
        siret = _generate_valid_siret(self.rng)
        iban = _generate_valid_iban(self.rng)
        client = self.rng.choice(_COMPANY_NAMES)
        client_addr = self.rng.choice(_ADDRESSES)
        inv_num = f"F-2025-{self.rng.randint(1, 9999):04d}"
        month = self.rng.randint(1, 12)
        day = self.rng.randint(1, 28)
        date_str = f"{day:02d}/{month:02d}/2025"

        # Header
        draw.rectangle([0, 0, W, 90], fill="#1a2332")
        draw.text((25, 15), company, fill="white", font=self._font_xl)
        draw.text((25, 55), address, fill="#8899aa", font=self._font_sm)

        # Numéro facture
        draw.text((W - 280, 110), f"FACTURE N° {inv_num}", fill="#1a2332", font=self._font_lg)
        draw.text((W - 280, 140), f"Date : {date_str}", fill="#555", font=self._font)

        # Client
        draw.rectangle([W - 320, 180, W - 30, 280], outline="#dee2e6", width=1)
        draw.text((W - 310, 185), "FACTURER À :", fill="#888", font=self._font_sm)
        draw.text((W - 310, 205), client, fill="#1a2332", font=self._font)
        draw.text((W - 310, 230), client_addr, fill="#555", font=self._font_sm)

        # Info émetteur
        draw.text((30, 110), f"SIRET : {siret}", fill="#333", font=self._font)
        draw.text((30, 135), f"IBAN : {iban}", fill="#333", font=self._font)

        # Tableau
        table_top = 310
        draw.rectangle([30, table_top, W - 30, table_top + 35], fill="#2c3e50")
        headers = ["Description", "Qté", "P.U. HT", "Montant HT"]
        hx = [40, 500, 600, 720]
        for h, x in zip(headers, hx):
            draw.text((x, table_top + 8), h, fill="white", font=self._font)

        # Lignes
        n_lines = self.rng.randint(3, 6)
        items = self.rng.sample(_LINE_ITEMS_FACTURE, min(n_lines, len(_LINE_ITEMS_FACTURE)))
        total_ht = 0.0
        y = table_top + 40
        amounts_info = []

        for item in items:
            qty = self.rng.randint(1, 10)
            pu = round(self.rng.uniform(50, 2000), 2)
            montant = round(qty * pu, 2)
            total_ht += montant

            bg = "#f8f9fa" if (items.index(item) % 2 == 0) else "#ffffff"
            draw.rectangle([30, y - 2, W - 30, y + 25], fill=bg)

            draw.text((40, y), item, fill="#333", font=self._font)
            draw.text((510, y), str(qty), fill="#333", font=self._font)
            draw.text((600, y), _fmt_amount(pu), fill="#333", font=self._font)
            draw.text((720, y), _fmt_amount(montant), fill="#333", font=self._font)

            amounts_info.append({"item": item, "qty": qty, "pu": pu, "montant": montant, "y": y})
            y += 30

        draw.line([(30, y), (W - 30, y)], fill="#dee2e6", width=1)

        # Totaux
        tva_rate = 0.20
        tva = round(total_ht * tva_rate, 2)
        ttc = round(total_ht + tva, 2)

        y += 15
        totals_y = y
        draw.text((550, y), "Total HT :", fill="#333", font=self._font)
        draw.text((700, y), _fmt_amount(total_ht), fill="#333", font=self._font)
        y += 25
        draw.text((550, y), "TVA 20% :", fill="#333", font=self._font)
        draw.text((700, y), _fmt_amount(tva), fill="#333", font=self._font)
        y += 25
        draw.rectangle([540, y - 3, W - 30, y + 22], fill="#1a2332")
        draw.text((550, y), "TOTAL TTC :", fill="white", font=self._font_lg)
        ttc_x = 700
        draw.text((ttc_x, y), _fmt_amount(ttc), fill="#2dd4bf", font=self._font_lg)
        ttc_y = y

        # Footer
        draw.text((30, H - 60), f"Règlement à 30 jours — IBAN : {iban}", fill="#888", font=self._font_sm)
        draw.text((30, H - 40), f"SIRET : {siret} — TVA intracommunautaire", fill="#888", font=self._font_sm)

        gt = {
            "company": company, "siret": siret, "iban": iban,
            "total_ht": total_ht, "tva": tva, "ttc": ttc,
            "date": date_str, "invoice_num": inv_num,
            "amounts_info": amounts_info,
            "ttc_position": (ttc_x, ttc_y),
            "totals_y": totals_y,
        }

        return SyntheticSample(
            image=img, label=0, forgery_type="none",
            forgery_zone=None, doc_type="facture", ground_truth=gt,
        )

    def _gen_bulletin(self) -> SyntheticSample:
        """Génère un bulletin de paie français réaliste."""
        W, H = 850, 1200
        img = Image.new("RGB", (W, H), "#FFFFFF")
        draw = ImageDraw.Draw(img)

        company = self.rng.choice(_COMPANY_NAMES)
        siret = _generate_valid_siret(self.rng)
        employee = f"{self.rng.choice(_FIRST_NAMES)} {self.rng.choice(_LAST_NAMES)}"
        month_idx = self.rng.randint(0, 11)
        month_name = _MONTHS_FR[month_idx]

        base = round(self.rng.uniform(1800, 5000), 2)
        prime_anc = round(base * 0.05, 2)
        prime_transport = round(self.rng.uniform(40, 80), 2)
        brut = round(base + prime_anc + prime_transport, 2)
        cotis = round(brut * 0.22, 2)
        csg = round(brut * 0.063, 2)
        net_imposable = round(brut - cotis - csg, 2)
        net_a_payer = round(net_imposable - round(brut * 0.015, 2), 2)

        # Header
        draw.rectangle([0, 0, W, 85], fill="#2c3e50")
        draw.text((25, 12), "BULLETIN DE PAIE", fill="white", font=self._font_xl)
        draw.text((25, 50), company, fill="#bdc3c7", font=self._font)

        # Infos
        draw.text((30, 100), f"SIRET : {siret}", fill="#333", font=self._font)
        draw.text((30, 125), f"Période : {month_name} 2025", fill="#333", font=self._font)
        draw.text((450, 100), f"Salarié : {employee}", fill="#333", font=self._font)
        draw.text((450, 125), f"Emploi : Cadre", fill="#333", font=self._font)

        # Tableau
        y = 180
        lines = [
            ("Salaire de base", _fmt_amount(base)),
            ("Prime d'ancienneté (5%)", _fmt_amount(prime_anc)),
            ("Prime de transport", _fmt_amount(prime_transport)),
            ("SALAIRE BRUT", _fmt_amount(brut)),
            ("Cotisations salariales (22%)", f"- {_fmt_amount(cotis)}"),
            ("CSG / CRDS (6,3%)", f"- {_fmt_amount(csg)}"),
            ("NET IMPOSABLE", _fmt_amount(net_imposable)),
            ("NET A PAYER", _fmt_amount(net_a_payer)),
        ]

        amounts_info = []
        for i, (label, amount) in enumerate(lines):
            is_bold = label in ("SALAIRE BRUT", "NET IMPOSABLE", "NET A PAYER")
            font = self._font_lg if is_bold else self._font
            color = "#1a2332" if is_bold else "#333"

            if is_bold:
                draw.rectangle([30, y - 3, W - 30, y + 22], fill="#f0f4f8")

            draw.text((50, y), label, fill=color, font=font)
            amount_x = 650
            draw.text((amount_x, y), amount, fill=color, font=font)
            draw.line([(40, y + 24), (W - 40, y + 24)], fill="#dee2e6")

            amounts_info.append({"label": label, "amount": amount, "y": y, "x": amount_x})
            y += 32

        net_y = amounts_info[-1]["y"]
        net_x = amounts_info[-1]["x"]

        # Footer
        draw.text((30, H - 80), f"Document généré automatiquement — {company}", fill="#888", font=self._font_sm)
        draw.text((30, H - 60), f"SIRET : {siret}", fill="#888", font=self._font_sm)

        gt = {
            "company": company, "siret": siret, "employee": employee,
            "base": base, "brut": brut, "net_imposable": net_imposable,
            "net_a_payer": net_a_payer, "month": month_name,
            "amounts_info": amounts_info,
            "net_position": (net_x, net_y),
        }

        return SyntheticSample(
            image=img, label=0, forgery_type="none",
            forgery_zone=None, doc_type="bulletin_paie", ground_truth=gt,
        )

    def _gen_quittance(self) -> SyntheticSample:
        """Génère une quittance de loyer française."""
        W, H = 850, 1000
        img = Image.new("RGB", (W, H), "#FFFFFF")
        draw = ImageDraw.Draw(img)

        owner = f"{self.rng.choice(_FIRST_NAMES)} {self.rng.choice(_LAST_NAMES)}"
        tenant = f"{self.rng.choice(_FIRST_NAMES)} {self.rng.choice(_LAST_NAMES)}"
        address = self.rng.choice(_ADDRESSES)
        month_idx = self.rng.randint(0, 11)

        loyer = round(self.rng.uniform(400, 1800), 2)
        charges = round(self.rng.uniform(30, 150), 2)
        total = round(loyer + charges, 2)

        # Header
        draw.rectangle([0, 0, W, 80], fill="#34495e")
        draw.text((25, 15), "QUITTANCE DE LOYER", fill="white", font=self._font_xl)
        draw.text((25, 50), f"{_MONTHS_FR[month_idx]} 2025", fill="#bdc3c7", font=self._font)

        # Infos
        y = 110
        draw.text((30, y), f"Bailleur : {owner}", fill="#333", font=self._font)
        y += 30
        draw.text((30, y), f"Locataire : {tenant}", fill="#333", font=self._font)
        y += 30
        draw.text((30, y), f"Adresse du bien : {address}", fill="#333", font=self._font)

        # Détail
        y += 60
        draw.text((30, y), "Détail du règlement :", fill="#1a2332", font=self._font_lg)
        y += 40

        lines = [
            ("Loyer nu", _fmt_amount(loyer)),
            ("Charges locatives", _fmt_amount(charges)),
        ]
        amounts_info = []
        for label, amount in lines:
            draw.text((60, y), label, fill="#333", font=self._font)
            draw.text((600, y), amount, fill="#333", font=self._font)
            draw.line([(50, y + 22), (W - 50, y + 22)], fill="#dee2e6")
            amounts_info.append({"label": label, "amount": amount, "y": y})
            y += 30

        y += 10
        draw.rectangle([40, y - 3, W - 40, y + 25], fill="#1a2332")
        draw.text((60, y), "TOTAL", fill="white", font=self._font_lg)
        total_x, total_y = 600, y
        draw.text((total_x, total_y), _fmt_amount(total), fill="#2dd4bf", font=self._font_lg)

        # Attestation
        y += 60
        draw.text((30, y), f"Je soussigné(e) {owner}, propriétaire du logement désigné ci-dessus,", fill="#333", font=self._font_sm)
        y += 20
        draw.text((30, y), f"déclare avoir reçu de {tenant} la somme de {_fmt_amount(total)} €", fill="#333", font=self._font_sm)
        y += 20
        draw.text((30, y), f"au titre du loyer et des charges du mois de {_MONTHS_FR[month_idx]} 2025.", fill="#333", font=self._font_sm)

        gt = {
            "owner": owner, "tenant": tenant, "address": address,
            "loyer": loyer, "charges": charges, "total": total,
            "month": _MONTHS_FR[month_idx],
            "amounts_info": amounts_info,
            "total_position": (total_x, total_y),
        }

        return SyntheticSample(
            image=img, label=0, forgery_type="none",
            forgery_zone=None, doc_type="quittance", ground_truth=gt,
        )

    # ── Falsifications ──────────────────────────────────────────────────

    def _apply_forgery(self, sample: SyntheticSample, forgery_type: str) -> SyntheticSample:
        forgers = {
            "amount_edit": self._forge_amount_edit,
            "text_replace": self._forge_text_replace,
            "copy_paste": self._forge_copy_paste,
            "noise_inject": self._forge_noise_inject,
            "compression_artifact": self._forge_compression,
            "metadata_strip": self._forge_metadata,
        }
        fn = forgers.get(forgery_type, self._forge_amount_edit)
        return fn(sample)

    def _massive_recompress(self, img: Image.Image, zone: tuple) -> Image.Image:
        """Re-compresse une large zone à qualité JPEG 2 pour créer des artefacts ELA massifs."""
        x1, y1, x2, y2 = zone
        crop = img.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=2)
        buf.seek(0)
        degraded = Image.open(buf).convert("RGB")
        img.paste(degraded, (x1, y1))
        return img

    def _forge_amount_edit(self, sample: SyntheticSample) -> SyntheticSample:
        """Modifie montants + re-compresse 60% du document."""
        img = sample.image.copy()
        gt = sample.ground_truth
        w, h = img.size

        # Re-compresser toute la moitié basse du document à qualité très basse
        mid_y = h // 3
        img = self._massive_recompress(img, (0, mid_y, w, h))

        # Redessiner le montant par-dessus
        draw = ImageDraw.Draw(img)
        if sample.doc_type == "facture":
            pos = gt.get("ttc_position", (700, 500))
            original = gt.get("ttc", 1000)
        elif sample.doc_type == "bulletin_paie":
            pos = gt.get("net_position", (650, 400))
            original = gt.get("net_a_payer", 1800)
        else:
            pos = gt.get("total_position", (600, 350))
            original = gt.get("total", 800)

        x, y = pos
        draw.rectangle([x - 10, y - 5, x + 180, y + 30], fill="white")
        new_amount = round(original * self.rng.uniform(1.5, 3.0), 2)
        draw.text((x, y), _fmt_amount(new_amount), fill="#111", font=self._font_lg)

        # Bruit sur la zone éditée
        arr = np.array(img)
        noise = self.np_rng.randint(-50, 50, arr[mid_y:, :].shape)
        arr[mid_y:] = np.clip(arr[mid_y:].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        return SyntheticSample(
            image=img, label=1, forgery_type="amount_edit",
            forgery_zone=(0, mid_y, w, h - mid_y), doc_type=sample.doc_type,
            ground_truth={**gt, "forged_amount": new_amount, "original_amount": original},
        )

    def _forge_text_replace(self, sample: SyntheticSample) -> SyntheticSample:
        """Remplace le haut du document (header + SIRET) — re-compression massive."""
        img = sample.image.copy()
        gt = sample.ground_truth
        w, h = img.size

        # Re-compresser 60% du document à qualité très basse
        cut_y = int(h * 0.6)
        img = self._massive_recompress(img, (0, 0, w, cut_y))

        # Réécrire un faux SIRET par-dessus
        fake_siret = "N/A"
        if sample.doc_type != "quittance":
            draw = ImageDraw.Draw(img)
            draw.rectangle([100, 95, 350, 120], fill="#f0f0f0")
            fake_siret = _generate_invalid_siret(self.rng)
            draw.text((110, 100), fake_siret, fill="#222", font=self._font)

        # Bruit fort sur toute la zone + zone basse aussi
        arr = np.array(img)
        noise = self.np_rng.randint(-80, 80, arr[:cut_y, :].shape)
        arr[:cut_y] = np.clip(arr[:cut_y].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Aussi bruiter le bas
        noise_low = self.np_rng.randint(-30, 30, arr[cut_y:, :].shape)
        arr[cut_y:] = np.clip(arr[cut_y:].astype(np.int16) + noise_low, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        return SyntheticSample(
            image=img, label=1, forgery_type="text_replace",
            forgery_zone=(0, 0, w, cut_y), doc_type=sample.doc_type,
            ground_truth={**gt, "forged_siret": fake_siret},
        )

    def _forge_copy_paste(self, sample: SyntheticSample) -> SyntheticSample:
        """Duplique la moitié du document + re-compression massive."""
        img = sample.image.copy()
        w, h = img.size

        # Copier le quart supérieur et le coller au milieu
        src_h = h // 4
        block = img.crop((0, 0, w, src_h))

        # Re-compresser le bloc à qualité 2
        buf = io.BytesIO()
        block.save(buf, format="JPEG", quality=2)
        buf.seek(0)
        degraded = Image.open(buf).convert("RGB")

        # Coller au milieu et en bas
        dst_y1 = h // 3
        dst_y2 = h * 2 // 3
        img.paste(degraded, (0, dst_y1))
        img.paste(degraded, (0, dst_y2))

        # Bruit additionnel
        arr = np.array(img)
        noise = self.np_rng.randint(-60, 60, arr[dst_y1:dst_y2 + src_h].shape)
        arr[dst_y1:dst_y2 + src_h] = np.clip(
            arr[dst_y1:dst_y2 + src_h].astype(np.int16) + noise, 0, 255
        ).astype(np.uint8)

        return SyntheticSample(
            image=Image.fromarray(arr), label=1, forgery_type="copy_paste",
            forgery_zone=(0, dst_y1, w, dst_y2 + src_h - dst_y1), doc_type=sample.doc_type,
            ground_truth=sample.ground_truth,
        )

    def _forge_noise_inject(self, sample: SyntheticSample) -> SyntheticSample:
        """Injecte du bruit massif sur 70% du document."""
        img = sample.image.copy()
        arr = np.array(img)
        h, w = arr.shape[:2]

        # Bruit massif sur les 2/3 du document
        start_y = h // 6
        noise = self.np_rng.randint(-120, 120, arr[start_y:].shape)
        arr[start_y:] = np.clip(
            arr[start_y:].astype(np.int16) + noise, 0, 255
        ).astype(np.uint8)

        return SyntheticSample(
            image=Image.fromarray(arr), label=1, forgery_type="noise_inject",
            forgery_zone=(0, start_y, w, h - start_y), doc_type=sample.doc_type,
            ground_truth=sample.ground_truth,
        )

    def _forge_compression(self, sample: SyntheticSample) -> SyntheticSample:
        """Re-compression massive sur 70% du document + bandes dégradées."""
        img = sample.image.copy()
        h, w = img.size[1], img.size[0]  # PIL: width, height

        # Re-compresser la moitié centrale (grande zone)
        mid_start = h // 5
        mid_end = h * 4 // 5
        img = self._massive_recompress(img, (0, mid_start, w, mid_end))

        # Re-compresser aussi le haut
        img = self._massive_recompress(img, (0, 0, w // 2, mid_start))

        # Bruit sur toute la zone dégradée
        arr = np.array(img)
        noise = self.np_rng.randint(-60, 60, arr[mid_start:mid_end, :].shape)
        arr[mid_start:mid_end] = np.clip(
            arr[mid_start:mid_end].astype(np.int16) + noise, 0, 255
        ).astype(np.uint8)
        img = Image.fromarray(arr)

        return SyntheticSample(
            image=img, label=1, forgery_type="compression_artifact",
            forgery_zone=(0, mid_start, w, mid_end - mid_start), doc_type=sample.doc_type,
            ground_truth=sample.ground_truth,
        )

    def _forge_metadata(self, sample: SyntheticSample) -> SyntheticSample:
        """Triple compression extrême + bruit massif sur tout le document."""
        img = sample.image.copy()

        # Triple compression à qualités très basses (artefacts ELA forts)
        for q in [30, 8, 2]:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        # Bruit massif sur 80% du document
        arr = np.array(img)
        h, w = arr.shape[:2]
        start_y = h // 10
        noise = self.np_rng.randint(-70, 70, arr[start_y:].shape)
        arr[start_y:] = np.clip(
            arr[start_y:].astype(np.int16) + noise, 0, 255
        ).astype(np.uint8)

        return SyntheticSample(
            image=Image.fromarray(arr), label=1, forgery_type="metadata_strip",
            forgery_zone=(0, 0, w, h), doc_type=sample.doc_type,
            ground_truth=sample.ground_truth,
        )
