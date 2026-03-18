"""
Détection Copy-Move — Couche 3 du pipeline VerifDoc.

Détecte quand une zone du document a été copiée et collée
ailleurs (ex: dupliquer un montant, masquer du texte).

Méthode : ORB keypoints + RANSAC homography.
Adapté de : trinity652/Document-Forgery-Detection (copy_move/detector.py)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _orb_ransac(
    gray: np.ndarray,
    nfeatures: int = 10000,
    min_match_count: int = 10,
    min_spatial_dist: float = 20.0,
) -> dict:
    """Détecte les zones copy-move par ORB + RANSAC adaptatif.

    Améliorations v2 :
      - 10 000 features au lieu de 5 000 → meilleure couverture
      - Seuil RANSAC adaptatif selon la résolution de l'image
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(gray, None)

    if des is None or len(kp) < 2:
        return {
            "score": 0.0,
            "mask": np.zeros_like(gray),
            "match_count": 0,
        }

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des, des, k=3)

    good_matches = []
    for m_list in raw_matches:
        for m in m_list[1:]:  # Skip self-match
            pt1 = np.array(kp[m.queryIdx].pt)
            pt2 = np.array(kp[m.trainIdx].pt)
            if np.linalg.norm(pt1 - pt2) > min_spatial_dist:
                good_matches.append(m)
                break

    mask_out = np.zeros(gray.shape, dtype=np.uint8)
    score = 0.0

    if len(good_matches) >= min_match_count:
        src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Seuil RANSAC adaptatif : images haute résolution → seuil plus large
        h, w = gray.shape
        ransac_thresh = max(3.0, min(8.0, (h + w) / 500))

        _, ransac_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

        if ransac_mask is not None:
            inliers = ransac_mask.ravel().tolist()
            inlier_count = sum(inliers)
            score = min(1.0, inlier_count / max(1, len(good_matches)))

            for m, inlier in zip(good_matches, inliers):
                if inlier:
                    pt1 = tuple(map(int, kp[m.queryIdx].pt))
                    pt2 = tuple(map(int, kp[m.trainIdx].pt))
                    cv2.circle(mask_out, pt1, 5, 255, -1)
                    cv2.circle(mask_out, pt2, 5, 255, -1)
                    cv2.line(mask_out, pt1, pt2, 128, 1)

    return {
        "score": round(score, 4),
        "mask": mask_out,
        "match_count": len(good_matches),
    }


def analyze(image: Image.Image) -> dict:
    """Analyse copy-move complète.

    Returns:
        dict avec score, verdict, detail, mask.
    """
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    result = _orb_ransac(gray)
    score = result["score"]

    if score < 0.10:
        verdict = "clean"
        detail = "Aucune zone dupliquée détectée"
    elif score < 0.40:
        verdict = "suspect"
        detail = f"{result['match_count']} correspondances suspectes détectées"
    else:
        verdict = "forged"
        detail = f"Zone(s) copy-move détectée(s) — {result['match_count']} correspondances"

    return {
        "analyzer": "copy_move",
        "score": score,
        "verdict": verdict,
        "detail": detail,
        "mask": result["mask"],
    }
