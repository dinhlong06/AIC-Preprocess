"""
transition_selector.py — Transition-Aware Keyframe Selection.

Phát hiện các điểm chuyển cảnh (Transition Peaks / Shot Cuts / Motion Spikes)
và bảo vệ các frame này khỏi bị xóa bởi Semantic / Diversity Filtering.
Đảm bảo không bỏ sót các frame quan trọng ở ranh giới phân cảnh.

Upgrade v2:
  - Prominence-based peak detection (thay vì chỉ so sánh 1 lân cận ngay kề).
    Một peak phải "nổi" hơn tất cả frame xung quanh trong cửa sổ prominence_window,
    ngăn các transition gradual / fade có nhiều frame cao gần bằng nhau tạo ra nhiều đỉnh.
  - Min-inter-peak distance: 2 transition peaks phải cách nhau ít nhất
    min_peak_distance frame để ngăn cluster đỉnh quá sát nhau.
  - Score-weighted tiebreak: khi 2 peaks cạnh tranh trong cùng cluster,
    chỉ giữ frame có steepness cao nhất.
"""

from __future__ import annotations

from typing import List, Tuple, Set
import numpy as np


class TransitionAwareSelector:
    """
    Phát hiện và bảo vệ các frame ranh giới chuyển cảnh.

    Args:
        peak_percentile    : Ngưỡng percentile để xác định steepness peak.
        prominence_window  : Bán kính cửa sổ tìm local prominence.
                             Peak i phải là max trong [i-w, i+w]. Mặc định 3.
        min_peak_distance  : Khoảng cách tối thiểu (frame count) giữa 2 peaks.
                             Loại bỏ cluster peaks do transition gradual. Mặc định 5.
        protect_shot_boundaries : Tự động bảo vệ frame đầu của mỗi shot.
    """

    def __init__(
        self,
        peak_percentile: float = 90.0,
        prominence_window: int = 3,
        min_peak_distance: int = 5,
        protect_shot_boundaries: bool = True,
    ):
        self.peak_percentile = peak_percentile
        self.prominence_window = prominence_window
        self.min_peak_distance = min_peak_distance
        self.protect_shot_boundaries = protect_shot_boundaries

    def find_transition_peaks(
        self,
        frame_indices: List[int],
        steepness_scores: List[float],
    ) -> Set[int]:
        """
        Tìm các prominent peaks (cực đại địa phương nổi bật) trong steepness scores.

        Upgrade v2:
          1. Lọc qua percentile threshold.
          2. Áp dụng prominence window: peak phải là max trong [i-w, i+w].
          3. NMS theo min_peak_distance: loại peak yếu hơn nếu quá gần.
          4. Kết quả: 1 peak duy nhất đại diện cho mỗi vùng chuyển cảnh.

        Args:
            frame_indices   : Danh sách frame_idx.
            steepness_scores: Điểm thay đổi thị giác (JPEG steepness).

        Returns:
            Tập hợp frame_idx được đánh dấu là Transition Keyframes.
        """
        if not frame_indices or not steepness_scores:
            return set()

        protected: Set[int] = set()
        N = len(steepness_scores)

        if self.protect_shot_boundaries:
            protected.add(frame_indices[0])

        if N < 3:
            return protected

        scores_arr = np.array(steepness_scores, dtype=np.float32)
        perc_thresh = float(np.percentile(scores_arr, self.peak_percentile))
        w = self.prominence_window

        # Step 1: Find prominent local maxima using prominence window
        raw_peaks: List[Tuple[int, float]] = []  # (frame_idx, score)
        for i in range(1, N - 1):
            if scores_arr[i] < perc_thresh:
                continue
            lo = max(0, i - w)
            hi = min(N - 1, i + w)
            # Must be the maximum within the prominence window
            if scores_arr[i] >= np.max(scores_arr[lo : hi + 1]):
                raw_peaks.append((frame_indices[i], float(scores_arr[i])))

        if not raw_peaks:
            return protected

        # Step 2: NMS by min_peak_distance — suppress weaker peaks too close together
        # Sort by score descending so stronger peaks win tiebreaks
        raw_peaks.sort(key=lambda x: x[1], reverse=True)

        suppressed: Set[int] = set()
        accepted_frame_indices: List[int] = []

        for frame_idx, score in raw_peaks:
            # Check if this peak is too close to any already-accepted peak
            too_close = any(
                abs(frame_idx - acc) < self.min_peak_distance
                for acc in accepted_frame_indices
            )
            if not too_close:
                accepted_frame_indices.append(frame_idx)
            else:
                suppressed.add(frame_idx)

        protected.update(accepted_frame_indices)
        return protected

    def merge_protected_keyframes(
        self,
        candidate_indices: List[int],
        candidate_embeddings: np.ndarray,
        filtered_indices: List[int],
        filtered_embeddings: np.ndarray,
        protected_indices: Set[int],
    ) -> Tuple[List[int], np.ndarray]:
        """
        Kết hợp các frame đã được lọc với các transition frame bắt buộc phải bảo vệ.

        Args:
            candidate_indices    : Tất cả candidate frame_idx ban đầu.
            candidate_embeddings : Embedding của tất cả candidate.
            filtered_indices     : Frame_idx sống sót sau Semantic / Diversity filter.
            filtered_embeddings  : Embedding tương ứng của filtered_indices.
            protected_indices    : Tập hợp frame_idx chuyển cảnh bắt giữ lại.

        Returns:
            Tuple (final_indices, final_embeddings) đã sort theo thứ tự thời gian.
        """
        if not protected_indices:
            return filtered_indices, filtered_embeddings

        final_map = {idx: emb for idx, emb in zip(filtered_indices, filtered_embeddings)}

        # Thêm các protected indices chưa có
        for idx, emb in zip(candidate_indices, candidate_embeddings):
            if idx in protected_indices and idx not in final_map:
                final_map[idx] = emb

        sorted_indices = sorted(final_map.keys())
        sorted_embeddings = np.stack([final_map[idx] for idx in sorted_indices])

        return sorted_indices, sorted_embeddings
