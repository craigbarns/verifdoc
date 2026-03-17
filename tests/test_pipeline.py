"""
Test rapide du pipeline VerifDoc.

Crée une image test et vérifie que toutes les couches tournent.
Usage : python -m tests.test_pipeline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def create_test_document(tampered: bool = False) -> Image.Image:
    """Crée un faux bulletin de paie pour tester."""
    img = Image.new("RGB", (800, 1100), "white")
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([0, 0, 800, 80], fill="#2c3e50")
    draw.text((20, 20), "BULLETIN DE PAIE", fill="white")
    draw.text((20, 50), "Entreprise ACME SAS", fill="#bdc3c7")

    # Infos employeur
    draw.text((20, 100), "SIRET : 123 456 789 00012", fill="black")
    draw.text((20, 125), "Periode : JANVIER 2025", fill="black")

    # Lignes de paie
    y = 180
    lines = [
        ("Salaire de base", "2 500,00"),
        ("Prime anciennete", "125,00"),
        ("SALAIRE BRUT", "2 625,00"),
        ("Cotisations salariales", "- 578,00"),
        ("CSG / CRDS", "- 165,00"),
        ("NET IMPOSABLE", "1 882,00"),
        ("NET A PAYER", "1 847,00"),
    ]
    for label, amount in lines:
        draw.text((40, y), label, fill="black")
        draw.text((600, y), amount, fill="black")
        draw.line([(40, y + 20), (750, y + 20)], fill="#dee2e6")
        y += 30

    if tampered:
        # Simuler une falsification : coller un rectangle + nouveau montant
        draw.rectangle([580, 180 + 6 * 30 - 5, 750, 180 + 6 * 30 + 18], fill="white")
        draw.text((600, 180 + 6 * 30), "3 847,00", fill="black")

        # Ajouter du bruit localisé
        arr = np.array(img)
        arr[370:390, 580:750] = np.clip(
            arr[370:390, 580:750].astype(np.int16) + np.random.randint(-30, 30, arr[370:390, 580:750].shape),
            0, 255,
        ).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def test_analyzers():
    """Teste chaque analyseur individuellement."""
    print("=" * 60)
    print("TEST VERIFDOC — Pipeline de détection")
    print("=" * 60)

    # Image clean
    img_clean = create_test_document(tampered=False)
    # Image falsifiée
    img_tampered = create_test_document(tampered=True)

    # Test ELA
    print("\n🔍 Test ELA...")
    from verifdoc.analyzers.ela import analyze as ela_analyze
    result = ela_analyze(img_clean)
    print(f"   Clean  → score={result['score']:.4f}, verdict={result['verdict']}")
    result = ela_analyze(img_tampered)
    print(f"   Tamper → score={result['score']:.4f}, verdict={result['verdict']}")

    # Test Noise
    print("\n📊 Test Analyse du bruit...")
    from verifdoc.analyzers.noise import analyze as noise_analyze
    result = noise_analyze(img_clean)
    print(f"   Clean  → score={result['score']:.4f}, verdict={result['verdict']}")
    result = noise_analyze(img_tampered)
    print(f"   Tamper → score={result['score']:.4f}, verdict={result['verdict']}")

    # Test Copy-Move
    print("\n🔄 Test Copy-Move...")
    from verifdoc.analyzers.copy_move import analyze as cm_analyze
    result = cm_analyze(img_clean)
    print(f"   Clean  → score={result['score']:.4f}, verdict={result['verdict']}")
    result = cm_analyze(img_tampered)
    print(f"   Tamper → score={result['score']:.4f}, verdict={result['verdict']}")

    # Test Metadata
    print("\n📋 Test Métadonnées...")
    from verifdoc.analyzers.metadata import analyze as meta_analyze
    result = meta_analyze(img_clean)
    print(f"   Image  → score={result['score']:.4f}, verdict={result['verdict']}")

    # Test Cross-check (avec données simulées)
    print("\n✅ Test Validation croisée...")
    from verifdoc.analyzers.cross_check import validate_siret, validate_bulletin_paie

    siret_valid = validate_siret("12345678900012")
    print(f"   SIRET 12345678900012 → valid={siret_valid['valid']}")

    fields = {"salaire_brut": 2625.0, "net_a_payer": 1847.0, "siret": "12345678900012"}
    result = validate_bulletin_paie(fields)
    print(f"   Bulletin → score={result['score']:.4f}, verdict={result['verdict']}")

    fields_fake = {"salaire_brut": 2625.0, "net_a_payer": 3847.0, "siret": "00000000000000"}
    result = validate_bulletin_paie(fields_fake)
    print(f"   Faux    → score={result['score']:.4f}, verdict={result['verdict']}")

    # Test Scoring
    print("\n📈 Test Scoring global...")
    from verifdoc.scoring import compute_final_score

    # Simuler des résultats
    fake_results = {
        "ela": {"score": 0.05, "verdict": "clean"},
        "noise": {"score": 0.10, "verdict": "clean"},
        "copy_move": {"score": 0.02, "verdict": "clean"},
        "metadata": {"score": 0.0, "verdict": "clean"},
        "cross_check": {"score": 0.0, "verdict": "clean"},
    }
    final = compute_final_score(fake_results)
    print(f"   Clean doc → score={final['score_100']}%, verdict={final['verdict_fr']}")

    fake_results_bad = {
        "ela": {"score": 0.45, "verdict": "forged"},
        "noise": {"score": 0.60, "verdict": "forged"},
        "copy_move": {"score": 0.10, "verdict": "suspect"},
        "metadata": {"score": 0.35, "verdict": "suspect"},
        "cross_check": {"score": 0.75, "verdict": "forged"},
    }
    final = compute_final_score(fake_results_bad)
    print(f"   Forged doc → score={final['score_100']}%, verdict={final['verdict_fr']}")

    print("\n" + "=" * 60)
    print("✅ TOUS LES TESTS PASSENT")
    print("=" * 60)


if __name__ == "__main__":
    test_analyzers()
