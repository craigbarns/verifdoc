"""
Benchmark Runner — Exécution du pipeline VerifDoc sur un dataset.

Collecte les scores par couche, timing et verdicts pour chaque sample.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .dataset_loader import BenchmarkSample


@dataclass
class SampleResult:
    """Résultat d'analyse d'un seul échantillon."""
    sample_id: int
    label: int                               # ground truth
    source: str
    forgery_type: str
    doc_type: str

    # Pipeline output
    final_score: float = 0.0                 # 0.0 - 1.0
    score_100: float = 0.0
    verdict: str = ""
    predicted_label: int = 0                 # 0 = clean, 1 = forged/suspect

    # Per-layer scores
    layer_scores: dict = field(default_factory=dict)
    layer_verdicts: dict = field(default_factory=dict)

    # Timing
    total_time_ms: float = 0.0

    # Error
    error: str | None = None


@dataclass
class BenchmarkRun:
    """Résultat complet d'une exécution de benchmark."""
    run_id: str
    timestamp: str
    config: dict
    results: list[SampleResult]
    total_time_seconds: float
    dataset_info: dict


class BenchmarkRunner:
    """Exécute le pipeline VerifDoc sur un dataset de benchmark."""

    def __init__(
        self,
        run_ocr: bool = True,
        skip_ai: bool = True,
        doc_type: str = "auto",
        verbose: bool = True,
    ):
        self.run_ocr = run_ocr
        self.skip_ai = skip_ai
        self.doc_type = doc_type
        self.verbose = verbose

    def run(
        self,
        samples: list[BenchmarkSample],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BenchmarkRun:
        """Exécute le benchmark sur tous les échantillons."""
        run_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now(timezone.utc).isoformat()
        total = len(samples)
        results: list[SampleResult] = []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  VerifDoc Benchmark — {total} samples")
            print(f"  OCR: {'ON' if self.run_ocr else 'OFF'} | AI: {'ON' if not self.skip_ai else 'OFF'}")
            print(f"{'='*60}\n")

        t_start = time.time()

        for i, sample in enumerate(samples):
            result = self._analyze_single(i, sample)
            results.append(result)

            if progress_callback:
                try:
                    progress_callback(i + 1, total)
                except Exception:
                    pass

            if self.verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                speed = (i + 1) / elapsed
                eta = (total - i - 1) / speed if speed > 0 else 0
                print(f"  [{i+1}/{total}] {speed:.1f} samples/s — ETA {eta:.0f}s")

        total_time = time.time() - t_start

        # Dataset info
        n_clean = sum(1 for s in samples if s.label == 0)
        n_forged = sum(1 for s in samples if s.label == 1)
        sources = list(set(s.source for s in samples))
        forgery_types = list(set(s.forgery_type for s in samples if s.label == 1))

        dataset_info = {
            "total": total,
            "clean": n_clean,
            "forged": n_forged,
            "sources": sources,
            "forgery_types": forgery_types,
        }

        config = {
            "run_ocr": self.run_ocr,
            "skip_ai": self.skip_ai,
            "doc_type": self.doc_type,
        }

        run = BenchmarkRun(
            run_id=run_id,
            timestamp=timestamp,
            config=config,
            results=results,
            total_time_seconds=round(total_time, 2),
            dataset_info=dataset_info,
        )

        if self.verbose:
            correct = sum(1 for r in results if r.predicted_label == r.label)
            acc = correct / total if total > 0 else 0
            print(f"\n{'='*60}")
            print(f"  Terminé en {total_time:.1f}s — Accuracy brute : {acc:.1%}")
            print(f"{'='*60}\n")

        return run

    def _analyze_single(self, idx: int, sample: BenchmarkSample) -> SampleResult:
        """Analyse un seul échantillon."""
        try:
            from verifdoc.pipeline import analyze_image

            t0 = time.perf_counter()

            # Forcer la désactivation de l'IA si demandé
            import os
            original_key = os.environ.get("ANTHROPIC_API_KEY")
            if self.skip_ai:
                os.environ.pop("ANTHROPIC_API_KEY", None)
                # Recharger le module pour mettre à jour le flag
                from verifdoc.analyzers import ai_analysis
                ai_analysis._API_KEY = None

            result = analyze_image(
                image=sample.image,
                doc_type=self.doc_type if self.doc_type != "auto" else sample.doc_type,
                run_ocr=self.run_ocr,
            )

            if self.skip_ai and original_key:
                os.environ["ANTHROPIC_API_KEY"] = original_key

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Extraire les scores par couche
            analysis = result.get("analysis", {})
            layer_scores = {}
            layer_verdicts = {}
            for layer_name in ["ela", "noise", "copy_move", "metadata", "cross_check", "ai_analysis"]:
                layer_data = analysis.get(layer_name, {})
                score = layer_data.get("score")
                layer_scores[layer_name] = float(score) if score is not None else -1
                layer_verdicts[layer_name] = layer_data.get("verdict", "skipped")

            final_score = result.get("final_score", 0.0)
            verdict = result.get("verdict", "clean")

            # Mapping verdict -> label binaire
            predicted = 0 if verdict == "clean" else 1

            return SampleResult(
                sample_id=idx,
                label=sample.label,
                source=sample.source,
                forgery_type=sample.forgery_type,
                doc_type=sample.doc_type,
                final_score=round(final_score, 4),
                score_100=round(result.get("score_100", 0.0), 1),
                verdict=verdict,
                predicted_label=predicted,
                layer_scores=layer_scores,
                layer_verdicts=layer_verdicts,
                total_time_ms=round(elapsed_ms, 1),
            )

        except Exception as e:
            return SampleResult(
                sample_id=idx,
                label=sample.label,
                source=sample.source,
                forgery_type=sample.forgery_type,
                doc_type=sample.doc_type,
                error=str(e),
            )

    # ── Persistance ─────────────────────────────────────────────────────

    @staticmethod
    def save_results(run: BenchmarkRun, output_dir: str | Path) -> Path:
        """Sauvegarde les résultats en JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "run_id": run.run_id,
            "timestamp": run.timestamp,
            "config": run.config,
            "dataset_info": run.dataset_info,
            "total_time_seconds": run.total_time_seconds,
            "results": [asdict(r) for r in run.results],
        }

        class _NpEncoder(json.JSONEncoder):
            def default(self, obj):
                try:
                    import numpy as np
                    if isinstance(obj, (np.floating,)):
                        return float(obj)
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                except ImportError:
                    pass
                return super().default(obj)

        path = output_dir / "results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, cls=_NpEncoder)

        return path

    @staticmethod
    def load_results(json_path: str | Path) -> BenchmarkRun:
        """Charge les résultats depuis un fichier JSON."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = [SampleResult(**r) for r in data["results"]]

        return BenchmarkRun(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            config=data["config"],
            results=results,
            total_time_seconds=data["total_time_seconds"],
            dataset_info=data["dataset_info"],
        )
