"""
metrics.py — Tính các chỉ số đánh giá chất lượng Keyframe Extraction.

Gồm hai nhóm metric:
  1. Unsupervised (luôn tính được, không cần Ground Truth):
       - Diversity Score: mức độ đa dạng giữa các keyframe trong cùng shot.
       - Redundancy Score: ngược lại với Diversity.
       - Coverage Score: % thời lượng shot được đại diện bởi keyframe.

  2. Supervised (chỉ dùng khi có Ground Truth ảnh keyframe):
       - One-to-one greedy matching theo cosine similarity.
       - TP, FP, FN → Precision, Recall, F1, Average Matched Similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.core.models import ShotKeyframes, ShotRecord


# ---------------------------------------------------------------------------
# Unsupervised metrics
# ---------------------------------------------------------------------------

def compute_diversity_score(embeddings: np.ndarray) -> float:
    """
    Tính Diversity Score = 1 - avg(cosine_similarity) giữa tất cả cặp keyframe.

    Cao (gần 1.0) = các keyframe rất khác nhau về ngữ nghĩa (tốt).
    Thấp (gần 0.0) = các keyframe giống nhau (redundant).

    Args:
        embeddings : numpy array shape (N, D), đã L2-normalized.

    Returns:
        Diversity score trong khoảng [0.0, 1.0].
        Trả về 1.0 nếu chỉ có 1 keyframe (không có cặp nào để so sánh).
    """
    n = len(embeddings)
    if n <= 1:
        return 1.0

    # Ma trận cosine similarity (đã normalize → chỉ cần dot product)
    sim_matrix = embeddings @ embeddings.T  # shape (N, N)

    # Lấy upper triangle (không kể đường chéo chính)
    upper_idx = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_idx]

    avg_sim = float(np.mean(pairwise_sims))
    return max(0.0, 1.0 - avg_sim)


def compute_redundancy_score(embeddings: np.ndarray) -> float:
    """
    Redundancy Score = avg(cosine_similarity) giữa tất cả cặp keyframe.
    Là nghịch đảo của Diversity Score.

    Args:
        embeddings : numpy array shape (N, D), đã L2-normalized.

    Returns:
        Redundancy score trong khoảng [0.0, 1.0].
    """
    return 1.0 - compute_diversity_score(embeddings)


def compute_coverage_score(
    shot: ShotRecord,
    keyframe_indices: List[int],
    window_sec: float = 1.0,
) -> float:
    """
    Coverage Score = % thời gian của shot được "cover" bởi ít nhất một keyframe.

    Mỗi keyframe cover một cửa sổ [frame_idx - w/2, frame_idx + w/2]
    trong đó w = window_sec * fps.

    Args:
        shot             : ShotRecord chứa start_frame, end_frame, fps.
        keyframe_indices : Danh sách frame_idx của các keyframe đã chọn.
        window_sec       : Bán kính coverage mỗi keyframe (giây).

    Returns:
        Coverage ratio trong khoảng [0.0, 1.0].
    """
    total = shot.total_frames
    if total == 0 or not keyframe_indices:
        return 0.0

    window_frames = int(window_sec * shot.fps)
    covered = np.zeros(total, dtype=bool)

    for kf_idx in keyframe_indices:
        # Chuyển frame_idx tuyệt đối → chỉ số tương đối trong shot
        rel = kf_idx - shot.start_frame
        lo = max(0, rel - window_frames // 2)
        hi = min(total - 1, rel + window_frames // 2)
        covered[lo : hi + 1] = True

    return float(covered.sum()) / total


# ---------------------------------------------------------------------------
# Supervised metrics — One-to-one Greedy Matching
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """
    Kết quả đánh giá một video theo phương pháp one-to-one matching.

    Fields:
        tp                   : True Positives (số cặp matched thành công)
        fp                   : False Positives (predicted không match được GT nào)
        fn                   : False Negatives (GT không được match bởi predicted nào)
        precision            : TP / (TP + FP)
        recall               : TP / (TP + FN)
        f1                   : Harmonic mean of Precision and Recall
        avg_matched_sim      : Trung bình cosine similarity của các cặp matched (0.0 nếu TP=0)
        gt_best_similarities : similarity tốt nhất của từng GT (-1.0 nếu unmatched)
    """
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    avg_matched_sim: float = 0.0
    gt_best_similarities: List[float] = field(default_factory=list)


def evaluate_with_ground_truth(
    pred_embeddings: np.ndarray,
    gt_embeddings: np.ndarray,
    threshold: float = 0.85,
) -> EvalResult:
    """
    Đánh giá Keyframe Extraction bằng one-to-one greedy matching.

    Theo protocol trong Layer_2_Testing.txt:
      Step 1 — Tính cosine similarity matrix sim[P, G].
      Step 2 — Lọc theo threshold τ: chỉ giữ cặp có sim >= τ.
      Step 3 — Greedy one-to-one matching:
                  Sắp xếp tất cả cặp (p, g) có sim >= τ theo thứ tự giảm dần.
                  Lần lượt gán cặp (p, g) nếu cả p và g chưa được gán.
                  → Đảm bảo mỗi GT chỉ match với tối đa 1 Prediction và ngược lại.
      Step 4 — Đếm TP (cặp matched), FP (predicted chưa match), FN (GT chưa match).
      Step 5 — Tính Precision, Recall, F1, Avg Matched Similarity.

    Args:
        pred_embeddings : shape (P, D) — embedding Prediction, L2-normalized.
        gt_embeddings   : shape (G, D) — embedding Ground Truth, L2-normalized.
        threshold       : Ngưỡng cosine similarity τ (default 0.85).

    Returns:
        EvalResult với đầy đủ TP, FP, FN, Precision, Recall, F1, AvgMatchedSim.
    """
    P = len(pred_embeddings)
    G = len(gt_embeddings)

    if P == 0 or G == 0:
        return EvalResult(
            tp=0, fp=P, fn=G,
            precision=0.0, recall=0.0, f1=0.0,
            avg_matched_sim=0.0,
            gt_best_similarities=[-1.0] * G,
        )

    # Step 1 — Cosine similarity matrix (P × G)
    # Both arrays should be L2-normalized; use matmul for efficiency
    pred_norm = pred_embeddings / (np.linalg.norm(pred_embeddings, axis=1, keepdims=True) + 1e-8)
    gt_norm   = gt_embeddings   / (np.linalg.norm(gt_embeddings,   axis=1, keepdims=True) + 1e-8)
    sim_matrix = pred_norm @ gt_norm.T   # (P, G)

    # Step 2 — Collect all valid pairs (sim >= threshold), sorted descending
    rows, cols = np.where(sim_matrix >= threshold)
    sims       = sim_matrix[rows, cols]
    order      = np.argsort(-sims)          # descending by similarity
    rows, cols, sims = rows[order], cols[order], sims[order]

    # Step 3 — Greedy one-to-one assignment
    matched_pred = [False] * P
    matched_gt   = [False] * G
    gt_best_sim  = [-1.0]  * G             # -1 = unmatched
    matched_sims: List[float] = []

    for p_idx, g_idx, sim in zip(rows, cols, sims):
        if matched_pred[p_idx] or matched_gt[g_idx]:
            continue
        matched_pred[p_idx] = True
        matched_gt[g_idx]   = True
        gt_best_sim[g_idx]  = float(sim)
        matched_sims.append(float(sim))

    # Step 4 — Count TP, FP, FN
    tp = len(matched_sims)
    fp = P - tp
    fn = G - tp

    # Step 5 — Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    avg_sim = float(np.mean(matched_sims)) if matched_sims else 0.0

    return EvalResult(
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        avg_matched_sim=avg_sim,
        gt_best_similarities=gt_best_sim,
    )
