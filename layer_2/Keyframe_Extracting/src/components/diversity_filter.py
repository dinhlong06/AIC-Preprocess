"""
diversity_filter.py — Diversity & Redundancy Filter.

Thực hiện lọc Redundant keyframe dựa trên ma trận Cosine Similarity pairwise.
Loại bỏ các frame có độ tương đồng vượt quá redundancy_threshold và tối đa hóa
Diversity Score (1 - avg_similarity) của tập keyframe đầu ra.

Upgrade v2:
  - Score-aware NMS: thay vì giữ frame đầu tiên (greedy theo thứ tự thời gian),
    ta giữ frame có DAKE steepness score cao nhất khi 2 frame cạnh tranh.
    Tránh tình trạng 1 frame mờ/blurry đầu shot đè bỏ 1 keyframe sắc nét hơn.
  - Temporal min-distance: 2 keyframe cách nhau ít hơn min_temporal_gap_frames
    (tính theo frame index) được coi là "temporal duplicate" và chỉ giữ 1.
  - Soft floor: đảm bảo shot ngắn luôn có ít nhất min_keyframes_per_shot kf
    (tránh shot hợp lệ bị filter hết).
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np


class DiversityRedundancyFilter:
    """
    Lọc các keyframe trùng lặp (redundant) dựa trên pairwise cosine similarity matrix.

    Args:
        redundancy_threshold    : Ngưỡng tương đồng tối đa giữa bất kỳ cặp keyframe nào.
                                  Nếu sim(i, j) >= redundancy_threshold, giữ frame có score cao hơn.
                                  Mặc định 0.88.
        min_temporal_gap_frames : Khoảng cách frame tối thiểu giữa 2 keyframe.
                                  2 frame ở frame_idx gần hơn ngưỡng này => loại frame score thấp hơn.
                                  Mặc định 3.
        min_keyframes_per_shot  : Số keyframe tối thiểu giữ lại cho mỗi shot (soft floor).
                                  Nếu sau filter còn ít hơn, add lại frame top-score bị loại.
                                  Mặc định 1.
    """

    def __init__(
        self,
        redundancy_threshold: float = 0.88,
        min_temporal_gap_frames: int = 5,
        min_keyframes_per_shot: int = 1,
    ):
        self.redundancy_threshold = redundancy_threshold
        self.min_temporal_gap_frames = min_temporal_gap_frames
        self.min_keyframes_per_shot = min_keyframes_per_shot

    def filter(
        self,
        frame_indices: List[int],
        embeddings: np.ndarray,
        scores: Optional[List[float]] = None,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Lọc danh sách candidate frame bằng ma trận pairwise cosine similarity NMS.

        Upgrade v2:
          - Score-aware tiebreak: giữ frame có score cao nhất (DAKE steepness hoặc
            mặc định theo vị trí ngược — frame giữa shot được ưu tiên hơn frame đầu).
          - Temporal duplicate removal: loại frame quá gần nhau về frame_idx.

        Args:
            frame_indices : List frame_idx (đã sort tăng dần).
            embeddings    : numpy array shape (N, D), L2-normalized.
            scores        : Optional list steepness score tương ứng với từng frame.
                            Nếu None, dùng uniform score = 1.0.

        Returns:
            Tuple (selected_indices, selected_embeddings).
        """
        N = len(frame_indices)
        if N <= 1:
            return frame_indices, embeddings

        # Use provided scores or default uniform
        if scores is None:
            frame_scores = np.ones(N, dtype=np.float32)
        else:
            frame_scores = np.array(scores, dtype=np.float32)

        # Ma trận cosine similarity N x N (embeddings đã L2-normalized)
        sim_matrix = embeddings @ embeddings.T  # (N, N)

        # Greedy NMS với score-aware tiebreak
        # Xử lý theo thứ tự score giảm dần (frame nào "quan trọng" nhất được xem xét trước)
        priority_order = np.argsort(frame_scores)[::-1]  # index sort by score desc

        keep_mask = np.zeros(N, dtype=bool)
        suppressed = np.zeros(N, dtype=bool)

        for i in priority_order:
            if suppressed[i]:
                continue

            keep_mask[i] = True

            # Suppress all frames too similar or too close temporally
            for j in range(N):
                if j == i or suppressed[j]:
                    continue

                # Temporal proximity check
                temporal_dist = abs(frame_indices[j] - frame_indices[i])
                if temporal_dist < self.min_temporal_gap_frames:
                    suppressed[j] = True
                    continue

                # Semantic similarity check
                sim = float(sim_matrix[i, j])
                if sim >= self.redundancy_threshold:
                    suppressed[j] = True

        # Soft floor: ensure minimum keyframes per shot
        selected_count = int(np.sum(keep_mask))
        if selected_count < self.min_keyframes_per_shot and N > 0:
            # Re-add suppressed frames by score until we hit the floor
            suppressed_by_score = sorted(
                [i for i in range(N) if not keep_mask[i]],
                key=lambda i: -frame_scores[i],
            )
            needed = self.min_keyframes_per_shot - selected_count
            for i in suppressed_by_score[:needed]:
                keep_mask[i] = True

        # Restore temporal order
        selected_indices = [frame_indices[i] for i in range(N) if keep_mask[i]]
        selected_embeddings = embeddings[keep_mask]

        return selected_indices, selected_embeddings
