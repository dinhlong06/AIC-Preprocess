"""
sharpness_selector.py — Local Sharpness Reselection.

Với mỗi keyframe đã chọn, xét cửa sổ ±window_radius frame lân cận (giới hạn
trong shot) và tráo ảnh sang frame nét hơn (Laplacian variance cao hơn) nếu
nội dung vẫn là cùng 1 cảnh (không phải frame ở shot/cảnh khác).

frame_idx/timestamp của keyframe KHÔNG đổi — chỉ đổi ảnh pixel dùng để
encode JPEG, để không phá vỡ min_frame_distance đã được enforce theo frame_idx
gốc ở các bước trước.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, List


class SharpnessReselector:
    """
    Tráo keyframe sang frame lân cận nét hơn, cùng nội dung.

    Args:
        window_radius  : Bán kính cửa sổ frame lân cận để xét. Mặc định 4.
        max_pixel_diff : Ngưỡng mean-abs pixel diff (grayscale, 0-255) để coi
                         2 frame là "cùng cảnh". Vượt ngưỡng này coi là khác
                         cảnh (vd cắt cảnh), không tráo. Mặc định 25.0.
    """

    def __init__(self, window_radius: int = 4, max_pixel_diff: float = 25.0):
        self.window_radius = window_radius
        self.max_pixel_diff = max_pixel_diff

    def reselect(
        self,
        indices: List[int],
        candidate_map: Dict[int, np.ndarray],
        shot_start: int,
        shot_end: int,
    ) -> List[np.ndarray]:
        """
        Trả về ảnh (có thể đã tráo) tương ứng với từng frame_idx trong `indices`,
        cùng thứ tự.
        """
        return [
            self._reselect_one(idx, candidate_map, shot_start, shot_end)
            for idx in indices
        ]

    def _reselect_one(
        self,
        idx: int,
        candidate_map: Dict[int, np.ndarray],
        shot_start: int,
        shot_end: int,
    ) -> np.ndarray:
        original = candidate_map[idx]
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        best_img = original
        best_var = float(cv2.Laplacian(gray_original, cv2.CV_64F).var())
        gray_original_f32 = gray_original.astype(np.float32)

        lo = max(shot_start, idx - self.window_radius)
        hi = min(shot_end, idx + self.window_radius)
        for cand_idx in range(lo, hi + 1):
            if cand_idx == idx:
                continue
            cand_img = candidate_map.get(cand_idx)
            if cand_img is None or cand_img.shape != original.shape:
                continue
            gray_cand = cv2.cvtColor(cand_img, cv2.COLOR_BGR2GRAY)
            if not self._same_scene(gray_original_f32, gray_cand.astype(np.float32)):
                continue
            var = float(cv2.Laplacian(gray_cand, cv2.CV_64F).var())
            if var > best_var:
                best_var = var
                best_img = cand_img

        return best_img

    def _sharpness(self, img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _same_scene(self, gray_a: np.ndarray, gray_b: np.ndarray) -> bool:
        return float(np.mean(np.abs(gray_a - gray_b))) < self.max_pixel_diff
