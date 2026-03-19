#!/usr/bin/env python3
"""
VerifDoc Benchmark CLI — Évaluation des performances de détection.

Usage:
    python benchmark_cli.py --synthetic                     # 200 samples synthétiques
    python benchmark_cli.py --synthetic --count 500         # 500 samples
    python benchmark_cli.py --folder /path/to/dataset       # Dataset local
    python benchmark_cli.py --folder /path --labels l.csv   # Avec CSV
    python benchmark_cli.py --report results.json           # Regénérer rapport
    python benchmark_cli.py --synthetic --no-ocr            # Sans OCR
    python benchmark_cli.py --synthetic --with-ai           # Avec IA Claude
"""

import argparse
import sys
import time
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="VerifDoc Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true",
                        help="Générer et tester un dataset synthétique")
    source.add_argument("--folder", type=str,
                        help="Chemin vers un dossier d'images (clean/ + forged/)")
    source.add_argument("--dataset", type=str,
                        help="Dataset HuggingFace (ex: Capstone-S21/DocTamper)")
    source.add_argument("--report", type=str,
                        help="Regénérer le rapport depuis un results.json")

    parser.add_argument("--count", type=int, default=200,
                        help="Nombre total de samples synthétiques (default: 200)")
    parser.add_argument("--labels", type=str,
                        help="Fichier CSV de labels pour --folder")
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine aléatoire (default: 42)")
    parser.add_argument("--no-ocr", action="store_true",
                        help="Désactiver la couche OCR")
    parser.add_argument("--with-ai", action="store_true",
                        help="Activer la couche IA (nécessite ANTHROPIC_API_KEY)")
    parser.add_argument("--doc-type", type=str, default="auto",
                        help="Type de document (default: auto)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Dossier de sortie (default: benchmark_results)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Mode silencieux")

    args = parser.parse_args()

    from verifdoc.benchmark.dataset_loader import DatasetLoader
    from verifdoc.benchmark.runner import BenchmarkRunner
    from verifdoc.benchmark.metrics import compute_metrics
    from verifdoc.benchmark.report import generate_report

    # ── Charger les données ─────────────────────────────────────────

    if args.report:
        # Mode regénération de rapport
        print(f"Chargement des résultats depuis {args.report}...")
        run = BenchmarkRunner.load_results(args.report)
        metrics = compute_metrics(run)
        output_dir = Path(args.output_dir)
        report_path = generate_report(metrics, run, output_dir / "report.html")
        print(f"\n✅ Rapport regénéré : {report_path}")
        return

    print("\n" + "=" * 60)
    print("  🔬 VerifDoc Benchmark Suite v1.0")
    print("=" * 60)

    if args.synthetic:
        n_clean = args.count // 2
        n_forged = args.count - n_clean
        print(f"\n📦 Génération du dataset synthétique ({n_clean} clean + {n_forged} forged)...")
        t0 = time.time()
        samples = DatasetLoader.load_synthetic(
            n_clean=n_clean, n_forged=n_forged, seed=args.seed
        )
        print(f"   ✅ {len(samples)} samples générés en {time.time() - t0:.1f}s")

    elif args.folder:
        print(f"\n📦 Chargement depuis {args.folder}...")
        samples = DatasetLoader.load_from_folder(args.folder, args.labels)
        print(f"   ✅ {len(samples)} samples chargés")

    elif args.dataset:
        print(f"\n📦 Chargement depuis HuggingFace ({args.dataset})...")
        samples = DatasetLoader.load_huggingface(args.dataset, max_samples=args.count)
        print(f"   ✅ {len(samples)} samples chargés")

    if not samples:
        print("❌ Aucun sample chargé. Vérifiez vos paramètres.")
        sys.exit(1)

    # ── Exécuter le benchmark ───────────────────────────────────────

    runner = BenchmarkRunner(
        run_ocr=not args.no_ocr,
        skip_ai=not args.with_ai,
        doc_type=args.doc_type,
        verbose=not args.quiet,
    )

    run = runner.run(samples)

    # ── Calculer les métriques ──────────────────────────────────────

    print("\n📊 Calcul des métriques...")
    metrics = compute_metrics(run)

    # ── Sauvegarder et générer le rapport ───────────────────────────

    output_dir = Path(args.output_dir)
    results_path = BenchmarkRunner.save_results(run, output_dir)
    print(f"   💾 Résultats : {results_path}")

    report_path = generate_report(metrics, run, output_dir / "report.html")
    print(f"   📄 Rapport HTML : {report_path}")

    # ── Résumé ──────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("  📊 RÉSULTATS")
    print(f"{'='*60}")
    print(f"  Seuil opt.: {metrics.optimal_threshold:.4f} (Youden's J = {metrics.youden_j:.3f})")
    print(f"  Accuracy  : {metrics.accuracy:.1%}")
    print(f"  Precision : {metrics.precision:.1%}")
    print(f"  Recall    : {metrics.recall:.1%}")
    print(f"  F1-Score  : {metrics.f1:.1%}")
    print(f"  AUC-ROC   : {metrics.auc:.4f}")
    print(f"  Latence   : {metrics.latency_mean_ms:.0f}ms (moy) / {metrics.latency_p95_ms:.0f}ms (P95)")
    print(f"  Débit     : {metrics.throughput_per_second:.1f} docs/s")

    # Score separation
    cs = metrics.score_stats.get("clean", {})
    fs = metrics.score_stats.get("forged", {})
    print(f"\n  Scores clean  : {cs.get('mean', 0):.4f} ± {cs.get('std', 0):.4f}  [{cs.get('min', 0):.4f} - {cs.get('max', 0):.4f}]")
    print(f"  Scores forged : {fs.get('mean', 0):.4f} ± {fs.get('std', 0):.4f}  [{fs.get('min', 0):.4f} - {fs.get('max', 0):.4f}]")
    print(f"  Ratio séparation : {metrics.score_stats.get('separation_ratio', 0)}x")
    print()

    if metrics.metrics_by_forgery:
        print("  Détection par type :")
        ft_labels = {
            "amount_edit": "Montant modifié",
            "text_replace": "Texte remplacé",
            "copy_paste": "Copier-coller",
            "noise_inject": "Bruit injecté",
            "compression_artifact": "Artefact JPEG",
            "metadata_strip": "Métadonnées",
        }
        for ft, data in sorted(metrics.metrics_by_forgery.items()):
            label = ft_labels.get(ft, ft)
            print(f"    {label:25s} : {data['detection_rate']:.0%} ({data['detected']}/{data['count']})")

    print(f"\n{'='*60}")
    print(f"  ✅ Rapport investor-ready : {report_path}")
    print(f"  ⏱  Temps total : {run.total_time_seconds:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
