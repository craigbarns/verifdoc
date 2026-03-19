"""
Dataset Loader — Chargeur unifié pour benchmarks VerifDoc.

Supporte : datasets synthétiques, dossiers locaux, HuggingFace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image


@dataclass
class BenchmarkSample:
    """Échantillon standardisé pour le benchmark."""
    image: Image.Image
    label: int                  # 0 = clean, 1 = forged
    source: str                 # "synthetic", "folder", "huggingface"
    forgery_type: str           # "none", "amount_edit", etc.
    doc_type: str               # "facture", "bulletin_paie", etc.
    metadata: dict = field(default_factory=dict)


class DatasetLoader:
    """Chargeur unifié de datasets pour le benchmark."""

    @staticmethod
    def load_synthetic(
        n_clean: int = 100,
        n_forged: int = 100,
        seed: int = 42,
        doc_types: list[str] | None = None,
    ) -> list[BenchmarkSample]:
        """Génère un dataset synthétique via ForgeFactory."""
        from .forge_factory import ForgeFactory

        factory = ForgeFactory(seed=seed)
        samples = factory.generate_dataset(n_clean=n_clean, n_forged=n_forged, doc_types=doc_types)

        return [
            BenchmarkSample(
                image=s.image,
                label=s.label,
                source="synthetic",
                forgery_type=s.forgery_type,
                doc_type=s.doc_type,
                metadata=s.ground_truth,
            )
            for s in samples
        ]

    @staticmethod
    def load_from_folder(
        folder: str | Path,
        labels_csv: str | Path | None = None,
    ) -> list[BenchmarkSample]:
        """Charge un dataset depuis un dossier local.

        Structure attendue (sans CSV) :
            folder/
                clean/      (ou authentic/, original/)
                    img1.png
                forged/     (ou tampered/, manipulated/)
                    img2.png

        Avec CSV (colonnes : filename, label, forgery_type) :
            folder/
                img1.png
                img2.png
            labels.csv
        """
        folder = Path(folder)
        samples: list[BenchmarkSample] = []

        if labels_csv:
            # Mode CSV
            import csv
            csv_path = Path(labels_csv)
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get("filename", "")
                    img_path = folder / fname
                    if not img_path.exists():
                        continue
                    try:
                        img = Image.open(img_path).convert("RGB")
                        samples.append(BenchmarkSample(
                            image=img,
                            label=int(row.get("label", 0)),
                            source="folder",
                            forgery_type=row.get("forgery_type", "unknown"),
                            doc_type=row.get("doc_type", "unknown"),
                        ))
                    except Exception:
                        continue
        else:
            # Mode sous-dossiers
            clean_names = ["clean", "authentic", "original"]
            forged_names = ["forged", "tampered", "manipulated"]

            for name in clean_names:
                d = folder / name
                if d.is_dir():
                    for img_path in sorted(d.iterdir()):
                        if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                            try:
                                img = Image.open(img_path).convert("RGB")
                                samples.append(BenchmarkSample(
                                    image=img, label=0, source="folder",
                                    forgery_type="none", doc_type="unknown",
                                ))
                            except Exception:
                                continue

            for name in forged_names:
                d = folder / name
                if d.is_dir():
                    for img_path in sorted(d.iterdir()):
                        if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                            try:
                                img = Image.open(img_path).convert("RGB")
                                samples.append(BenchmarkSample(
                                    image=img, label=1, source="folder",
                                    forgery_type="unknown", doc_type="unknown",
                                ))
                            except Exception:
                                continue

        return samples

    @staticmethod
    def load_huggingface(
        dataset_name: str = "Capstone-S21/DocTamper",
        split: str = "test",
        max_samples: int = 200,
    ) -> list[BenchmarkSample]:
        """Charge un dataset depuis HuggingFace.

        Nécessite : pip install datasets lmdb
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError(
                "Le package 'datasets' n'est pas installé.\n"
                "Installez-le avec : pip install datasets lmdb"
            )

        ds = load_dataset(dataset_name, split=split, streaming=True)
        samples: list[BenchmarkSample] = []

        for i, example in enumerate(ds):
            if i >= max_samples:
                break
            try:
                img = example.get("image")
                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    img = Image.open(img).convert("RGB")
                else:
                    img = img.convert("RGB")

                label = int(example.get("label", 0))
                samples.append(BenchmarkSample(
                    image=img,
                    label=label,
                    source="huggingface",
                    forgery_type="unknown",
                    doc_type="document",
                    metadata={"dataset": dataset_name, "split": split},
                ))
            except Exception:
                continue

        return samples
