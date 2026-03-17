"""
PDF Handler — Conversion PDF → Images pour analyse forensique.

Utilise PyMuPDF (fitz) pour extraire les pages en images haute résolution.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image


def pdf_to_images(
    pdf_path: str | Path,
    dpi: int = 200,
    max_pages: int = 10,
) -> list[Image.Image]:
    """Convertit un PDF en liste d'images PIL.

    Args:
        pdf_path: Chemin vers le fichier PDF.
        dpi: Résolution de rendu (200 = bon compromis qualité/vitesse).
        max_pages: Nombre max de pages à convertir.

    Returns:
        Liste d'images PIL (une par page).
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF requis: pip install PyMuPDF")

    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    images = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(min(len(doc), max_pages)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        images.append(img)

    doc.close()
    return images


def pdf_page_count(pdf_path: str | Path) -> int:
    """Retourne le nombre de pages d'un PDF."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0


def is_pdf(file_path: str | Path) -> bool:
    """Vérifie si un fichier est un PDF (par magic bytes)."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(5)
        return header == b"%PDF-"
    except Exception:
        return False
