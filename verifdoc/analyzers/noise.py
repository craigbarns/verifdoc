"""
Analyse du bruit par décomposition wavelet — Couche 2 du pipeline VerifDoc.

Les documents authentiques ont un profil de bruit uniforme.
Les zones modifiées introduisent des incohérences dans les
sous-bandes de détail wavelet (hautes fréquences).

Adapté de : trinity652/Document-Forgery-Detection (wavelet.py)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import pywt
except ImportError:
    pywt = None


def _to_gray(source: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    """Convertit n'importe quelle source en image grayscale numpy."""
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    elif isinstance(source, Image.Image):
        img = np.array(source.convert("L"))
    elif isinstance(source, np.ndarray):
        img = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source
    else:
        raise TypeError(f"Type non supporté: {type(source)}")
    return img


def wavelet_decompose(
    source: str | Path | Image.Image | np.ndarray,
    wavelet: str = "db1",
    level: int = 3,
) -> dict:
    """Décomposition wavelet multi-niveaux.

    Zéro les coefficients d'approximation pour ne garder que
    les détails (bords, bruit, artefacts de manipulation).

    Returns:
        dict avec reconstructed (uint8), heatmap (RGB), detail_bands.
    """
    if pywt is None:
        return {
            "reconstructed": None,
            "heatmap": None,
            "detail_bands": [],
            "available": False,
        }

    gray = _to_gray(source)
    img_float = gray.astype(np.float32) / 255.0

    coeffs = pywt.wavedec2(img_float, wavelet=wavelet, level=level)

    # Zéro approximation, garder détails
    coeffs_detail = list(coeffs)
    coeffs_detail[0] = np.zeros_like(coeffs[0])

    reconstructed = pywt.waverec2(coeffs_detail, wavelet=wavelet)
    reconstructed = reconstructed[: gray.shape[0], : gray.shape[1]]

    recon_norm = cv2.normalize(
        np.abs(reconstructed), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(recon_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return {
        "reconstructed": recon_norm,
        "heatmap": heatmap_rgb,
        "detail_bands": [coeffs[i] for i in range(1, len(coeffs))],
        "available": True,
    }


def _noise_score_single(gray: np.ndarray, block_size: int) -> float:
    """Score d'incohérence de bruit pour un block_size donné."""
    noise_map = cv2.Laplacian(gray, cv2.CV_32F)

    h, w = noise_map.shape
    variances = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = noise_map[y : y + block_size, x : x + block_size]
            variances.append(np.var(block))

    if len(variances) < 4:
        return 0.0

    variances = np.array(variances)
    median_var = np.median(variances)
    if median_var == 0:
        return 0.0

    q1, q3 = np.percentile(variances, [10, 90])
    filtered = variances[(variances >= q1) & (variances <= q3)]

    if len(filtered) < 4:
        filtered = variances

    mean_var = np.mean(filtered)
    if mean_var == 0:
        return 0.0

    cv = np.std(filtered) / mean_var
    score = min(1.0, cv / 3.0)
    return round(score, 4)


def noise_score(source: str | Path | Image.Image | np.ndarray) -> float:
    """Calcule un score d'incohérence de bruit [0-1].

    Multi-scale : analyse les blocs 16×16 et 32×32 puis retient
    le pire score (le plus suspect) pour ne rien rater.
    """
    gray = _to_gray(source).astype(np.float32)

    block_sizes = [16, 32]
    scores = [_noise_score_single(gray, bs) for bs in block_sizes]

    # Worst-case : retenir le pire score
    return max(scores) if scores else 0.0


def analyze(image: Image.Image) -> dict:
    """Analyse de bruit complète.

    Returns:
        dict avec score, verdict, detail, heatmap.
    """
    score = noise_score(image)
    wavelet_data = wavelet_decompose(image)

    if score < 0.25:
        verdict = "clean"
        detail = "Profil de bruit uniforme"
    elif score < 0.50:
        verdict = "suspect"
        detail = "Légères incohérences dans le profil de bruit"
    else:
        verdict = "forged"
        detail = "Incohérences de bruit significatives — zones modifiées probables"

    return {
        "analyzer": "noise",
        "score": score,
        "verdict": verdict,
        "detail": detail,
        "heatmap": wavelet_data.get("heatmap"),
    }
