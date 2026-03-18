"""
Error Level Analysis (ELA) — Couche 1 du pipeline VerifDoc.

Détecte les zones modifiées dans une image en comparant les niveaux
de compression JPEG. Les zones retouchées (Photoshop, éditeur PDF)
apparaissent plus lumineuses dans la carte ELA.

Adapté de :
  - jayant1211/Image-Tampering-Detection-using-ELA-and-Metadata-Analysis
  - trinity652/Document-Forgery-Detection
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def generate_ela(
    source: str | Path | Image.Image,
    quality: int = 90,
    scale: float = 15.0,
) -> Image.Image:
    """Génère une carte ELA (Error Level Analysis).

    Args:
        source: Chemin vers l'image ou objet PIL Image.
        quality: Qualité JPEG de recompression (90 = standard).
        scale: Multiplicateur pour amplifier les différences.

    Returns:
        PIL Image RGB montrant la carte ELA.
        Les zones lumineuses = manipulation potentielle.
    """
    if isinstance(source, (str, Path)):
        original = Image.open(source).convert("RGB")
    else:
        original = source.convert("RGB")

    # Recompress en JPEG
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")

    # Calculer la différence absolue
    orig_arr = np.array(original, dtype=np.float32)
    comp_arr = np.array(compressed, dtype=np.float32)

    ela_arr = np.abs(orig_arr - comp_arr) * scale
    ela_arr = ela_arr.clip(0, 255).astype(np.uint8)

    return Image.fromarray(ela_arr)


def ela_score(ela_image: Image.Image) -> float:
    """Score de suspicion ELA global [0-1].

    Plus le score est élevé, plus l'image est suspecte.
    """
    arr = np.array(ela_image, dtype=np.float32)
    max_possible = 255.0 * arr.shape[0] * arr.shape[1] * arr.shape[2]
    return round(float(arr.sum() / max_possible), 4)


def ela_hotspots(ela_image: Image.Image, threshold: float = 0.6) -> list[dict]:
    """Détecte les zones les plus suspectes (hotspots).

    Returns:
        Liste de dicts avec x, y, w, h, intensity pour chaque zone suspecte.
    """
    gray = np.array(ela_image.convert("L"), dtype=np.uint8)

    # Seuil adaptatif
    thresh_val = int(threshold * 255)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # Trouver les contours des zones suspectes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hotspots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # Ignorer le bruit
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = gray[y : y + h, x : x + w]
        intensity = float(np.mean(roi)) / 255.0
        hotspots.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "area": int(area),
            "intensity": round(intensity, 3),
        })

    # Trier par intensité décroissante
    hotspots.sort(key=lambda h: h["intensity"], reverse=True)
    return hotspots[:10]  # Top 10 max


def analyze(image: Image.Image) -> dict:
    """Analyse ELA complète d'une image.

    Multi-quality ELA : recompresse à 75, 85 et 95 pour réduire
    les faux positifs. Le score médian est retenu.

    Returns:
        dict avec score, verdict, hotspots, ela_image, quality_scores.
    """
    qualities = [75, 85, 95]
    quality_scores = {}
    ela_images = {}

    for q in qualities:
        ela_img = generate_ela(image, quality=q)
        quality_scores[q] = ela_score(ela_img)
        ela_images[q] = ela_img

    # Score médian pour robustesse
    sorted_scores = sorted(quality_scores.values())
    score = sorted_scores[len(sorted_scores) // 2]

    # Utiliser l'image ELA à qualité 90 (85 la plus proche) pour les hotspots
    best_q = min(qualities, key=lambda q: abs(q - 90))
    ela_img = ela_images[best_q]
    hotspots = ela_hotspots(ela_img)

    if score < 0.025:
        verdict = "clean"
        detail = "Aucune anomalie ELA détectée"
    elif score < 0.07:
        verdict = "suspect"
        detail = f"{len(hotspots)} zone(s) avec compression incohérente"
    else:
        verdict = "forged"
        detail = f"Manipulation probable — {len(hotspots)} zone(s) altérée(s)"

    return {
        "analyzer": "ela",
        "score": score,
        "verdict": verdict,
        "detail": detail,
        "hotspots": hotspots,
        "ela_image": ela_img,
        "quality_scores": quality_scores,
    }
