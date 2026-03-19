"""
Metrics Engine — Calcul de toutes les métriques de performance.

Precision, Recall, F1, AUC-ROC, confusion matrix, latency stats,
analyse par couche et par type de falsification.

Seuil optimal via Youden's J-statistic (maximise TPR - FPR).
Aucune dépendance sklearn — tout en pure Python + numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .runner import BenchmarkRun, SampleResult


@dataclass
class BenchmarkMetrics:
    """Métriques complètes d'un benchmark."""

    # Classification (au seuil optimal)
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # Seuil optimal
    optimal_threshold: float = 0.0
    youden_j: float = 0.0

    # Confusion matrix (au seuil optimal)
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # ROC
    roc_thresholds: list = field(default_factory=list)
    roc_tpr: list = field(default_factory=list)
    roc_fpr: list = field(default_factory=list)
    auc: float = 0.0

    # Score distribution
    clean_scores: list = field(default_factory=list)
    forged_scores: list = field(default_factory=list)
    score_stats: dict = field(default_factory=dict)

    # Per-layer
    layer_metrics: dict = field(default_factory=dict)

    # Timing
    latency_mean_ms: float = 0.0
    latency_median_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    throughput_per_second: float = 0.0

    # By forgery type
    metrics_by_forgery: dict = field(default_factory=dict)

    # Total
    total_samples: int = 0
    total_clean: int = 0
    total_forged: int = 0


def compute_metrics(run: BenchmarkRun) -> BenchmarkMetrics:
    """Calcule toutes les métriques depuis un BenchmarkRun."""
    results = [r for r in run.results if r.error is None]
    if not results:
        return BenchmarkMetrics()

    y_true = [r.label for r in results]
    scores = [r.final_score for r in results]

    m = BenchmarkMetrics()
    m.total_samples = len(results)
    m.total_clean = sum(1 for y in y_true if y == 0)
    m.total_forged = sum(1 for y in y_true if y == 1)

    # ── ROC Curve (haute résolution + data-driven) ───────────────
    # Utiliser les scores réels comme seuils + grille fine dans la plage utile
    unique_scores = sorted(set(scores))
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0

    # Grille fine dans la plage des données + au-delà
    fine_range = max_score - min_score
    fine_step = fine_range / 500 if fine_range > 0 else 0.001
    fine_thresholds = [min_score + i * fine_step for i in range(501)]

    # Ajouter les scores exacts et quelques marges
    all_thresholds = set(fine_thresholds)
    all_thresholds.update(unique_scores)
    all_thresholds.add(0.0)
    all_thresholds.add(max_score * 1.5)
    all_thresholds.add(1.0)
    # Grille supplémentaire très fine entre clean max et forged min
    clean_scores_raw = [s for s, t in zip(scores, y_true) if t == 0]
    forged_scores_raw = [s for s, t in zip(scores, y_true) if t == 1]

    if clean_scores_raw and forged_scores_raw:
        gap_low = max(clean_scores_raw)
        gap_high = min(forged_scores_raw)
        if gap_low < gap_high:
            # Zone de séparation : grille ultra-fine
            gap_step = (gap_high - gap_low) / 200
            for i in range(201):
                all_thresholds.add(gap_low + i * gap_step)

    thresholds = sorted(all_thresholds)
    tpr_list = []
    fpr_list = []
    j_values = []  # Youden's J = TPR - FPR

    for thresh in thresholds:
        tp = sum(1 for t, s in zip(y_true, scores) if t == 1 and s >= thresh)
        fp = sum(1 for t, s in zip(y_true, scores) if t == 0 and s >= thresh)
        fn = sum(1 for t, s in zip(y_true, scores) if t == 1 and s < thresh)
        tn = sum(1 for t, s in zip(y_true, scores) if t == 0 and s < thresh)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        j_values.append(tpr - fpr)

    m.roc_thresholds = [round(float(t), 6) for t in thresholds]
    m.roc_tpr = [round(float(t), 6) for t in tpr_list]
    m.roc_fpr = [round(float(t), 6) for t in fpr_list]

    # AUC (trapezoidal)
    m.auc = _auc_trapezoidal(fpr_list, tpr_list)

    # ── Seuil optimal (Youden's J-statistic) ─────────────────────
    best_j_idx = max(range(len(j_values)), key=lambda i: j_values[i])
    m.optimal_threshold = round(float(thresholds[best_j_idx]), 6)
    m.youden_j = round(float(j_values[best_j_idx]), 4)

    # ── Classification au seuil optimal ──────────────────────────
    y_pred_optimal = [1 if s >= m.optimal_threshold else 0 for s in scores]

    m.true_positives = sum(1 for t, p in zip(y_true, y_pred_optimal) if t == 1 and p == 1)
    m.true_negatives = sum(1 for t, p in zip(y_true, y_pred_optimal) if t == 0 and p == 0)
    m.false_positives = sum(1 for t, p in zip(y_true, y_pred_optimal) if t == 0 and p == 1)
    m.false_negatives = sum(1 for t, p in zip(y_true, y_pred_optimal) if t == 1 and p == 0)

    total = m.true_positives + m.true_negatives + m.false_positives + m.false_negatives
    m.accuracy = (m.true_positives + m.true_negatives) / total if total > 0 else 0

    denom_p = m.true_positives + m.false_positives
    m.precision = m.true_positives / denom_p if denom_p > 0 else 0

    denom_r = m.true_positives + m.false_negatives
    m.recall = m.true_positives / denom_r if denom_r > 0 else 0

    if m.precision + m.recall > 0:
        m.f1 = 2 * m.precision * m.recall / (m.precision + m.recall)
    else:
        m.f1 = 0.0

    # ── Score Distribution ──────────────────────────────────────────
    m.clean_scores = [r.final_score for r in results if r.label == 0]
    m.forged_scores = [r.final_score for r in results if r.label == 1]

    m.score_stats = {
        "clean": _score_stats(m.clean_scores),
        "forged": _score_stats(m.forged_scores),
        "separation_ratio": round(
            float(np.mean(m.forged_scores) / max(np.mean(m.clean_scores), 0.0001)), 1
        ) if m.forged_scores and m.clean_scores else 0,
    }

    # ── Per-Layer Analysis ────────────────────────────────────────────
    layer_names = ["ela", "noise", "copy_move", "metadata", "cross_check"]
    for layer in layer_names:
        clean_layer = [r.layer_scores.get(layer, -1) for r in results if r.label == 0]
        forged_layer = [r.layer_scores.get(layer, -1) for r in results if r.label == 1]

        # Filtrer les -1 (skipped)
        clean_layer = [s for s in clean_layer if s >= 0]
        forged_layer = [s for s in forged_layer if s >= 0]

        if clean_layer and forged_layer:
            mean_c = np.mean(clean_layer)
            mean_f = np.mean(forged_layer)
            std_c = np.std(clean_layer) or 0.001
            std_f = np.std(forged_layer) or 0.001
            pooled_std = math.sqrt((std_c**2 + std_f**2) / 2) or 0.001
            cohens_d = abs(mean_f - mean_c) / pooled_std

            # Calculer aussi l'AUC par couche
            layer_scores_all = clean_layer + forged_layer
            layer_labels_all = [0] * len(clean_layer) + [1] * len(forged_layer)
            layer_auc = _compute_simple_auc(layer_labels_all, layer_scores_all)

            m.layer_metrics[layer] = {
                "mean_clean": round(float(mean_c), 4),
                "mean_forged": round(float(mean_f), 4),
                "std_clean": round(float(std_c), 4),
                "std_forged": round(float(std_f), 4),
                "cohens_d": round(float(cohens_d), 3),
                "auc": round(float(layer_auc), 4),
                "separation": "excellent" if cohens_d > 1.5 else "bon" if cohens_d > 0.8 else "faible",
            }

    # ── Timing ────────────────────────────────────────────────────────
    times = [r.total_time_ms for r in results if r.total_time_ms > 0]
    if times:
        times_arr = np.array(times)
        m.latency_mean_ms = round(float(np.mean(times_arr)), 1)
        m.latency_median_ms = round(float(np.median(times_arr)), 1)
        m.latency_p95_ms = round(float(np.percentile(times_arr, 95)), 1)
        m.latency_p99_ms = round(float(np.percentile(times_arr, 99)), 1)
        m.latency_min_ms = round(float(np.min(times_arr)), 1)
        m.latency_max_ms = round(float(np.max(times_arr)), 1)
        total_s = sum(times) / 1000
        m.throughput_per_second = round(len(times) / total_s, 2) if total_s > 0 else 0

    # ── By Forgery Type (au seuil optimal) ────────────────────────────
    forgery_types = set(r.forgery_type for r in results if r.label == 1)
    for ft in forgery_types:
        ft_results = [r for r in results if r.forgery_type == ft]
        ft_scores = [r.final_score for r in ft_results if r.label == 1]
        detected = sum(1 for s in ft_scores if s >= m.optimal_threshold)
        total_ft = len(ft_scores)
        detection_rate = detected / total_ft if total_ft > 0 else 0

        m.metrics_by_forgery[ft] = {
            "count": total_ft,
            "detected": detected,
            "detection_rate": round(detection_rate, 3),
            "avg_score": round(float(np.mean(ft_scores)), 4) if ft_scores else 0,
        }

    return m


def _auc_trapezoidal(fpr: list[float], tpr: list[float]) -> float:
    """AUC par la règle trapézoïdale."""
    # Trier par FPR croissant
    pairs = sorted(zip(fpr, tpr))
    # Dédupliquer (prendre le max TPR pour chaque FPR)
    deduped = {}
    for f, t in pairs:
        f_r = round(f, 8)
        if f_r not in deduped or t > deduped[f_r]:
            deduped[f_r] = t
    sorted_pairs = sorted(deduped.items())
    fpr_s = [p[0] for p in sorted_pairs]
    tpr_s = [p[1] for p in sorted_pairs]

    auc = 0.0
    for i in range(1, len(fpr_s)):
        dx = fpr_s[i] - fpr_s[i - 1]
        dy = (tpr_s[i] + tpr_s[i - 1]) / 2
        auc += dx * dy

    return round(min(max(auc, 0.0), 1.0), 4)


def _compute_simple_auc(labels: list[int], scores: list[float]) -> float:
    """AUC simplifiée pour une couche individuelle."""
    if not labels or len(set(labels)) < 2:
        return 0.5
    n_thresholds = 50
    score_min = min(scores)
    score_max = max(scores)
    step = (score_max - score_min) / n_thresholds if score_max > score_min else 0.001
    thresholds = [score_min + i * step for i in range(n_thresholds + 1)]

    fpr_list = []
    tpr_list = []
    for thresh in thresholds:
        tp = sum(1 for t, s in zip(labels, scores) if t == 1 and s >= thresh)
        fp = sum(1 for t, s in zip(labels, scores) if t == 0 and s >= thresh)
        fn = sum(1 for t, s in zip(labels, scores) if t == 1 and s < thresh)
        tn = sum(1 for t, s in zip(labels, scores) if t == 0 and s < thresh)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return _auc_trapezoidal(fpr_list, tpr_list)


def _score_stats(scores: list[float]) -> dict:
    """Statistiques descriptives d'une liste de scores."""
    if not scores:
        return {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0}
    arr = np.array(scores)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }
