"""
blank_veto.py — Blank/Black/White Transition Cut Hard-Veto Filter.

Loại các frame "blank" (đen phẳng hoặc trắng/sáng phẳng, không có nội dung —
thường là artefact của transition cut) trước khi đưa vào BEiT-3 encode, để
chúng không lọt qua Semantic/Diversity Filter chỉ vì "đủ khác biệt" so với
các keyframe khác trong khi thực ra không mang thông tin gì.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple


class BlankTransitionVetoFilter:
    """
    Loại frame blank (đen hoặc trắng, không chi tiết) khỏi tập candidate.

    Args:
        min_brightness : Dưới ngưỡng này (mean pixel) coi là tối. Mặc định 15.0.
        max_brightness : Trên ngưỡng này (mean pixel) coi là sáng/trắng. Mặc định 240.0.
        min_variance    : Dưới ngưỡng này (Laplacian variance) coi là phẳng,
                          không có cạnh/chi tiết. Mặc định 15.0.
    """

    def __init__(
        self,
        min_brightness: float = 15.0,
        max_brightness: float = 240.0,
        min_variance: float = 15.0,
    ):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_variance = min_variance

    def filter(
        self,
        frames: List[Tuple[int, np.ndarray]],
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Loại frame blank khỏi danh sách candidate.

        Args:
            frames : List (frame_idx, BGR image numpy array).

        Returns:
            List (frame_idx, image) đã loại bỏ các frame blank.
        """
        return [(idx, img) for idx, img in frames if not self._is_blank(img)]

    def _is_blank(self, img: np.ndarray) -> bool:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance >= self.min_variance:
            return False
        brightness = float(gray.mean())
        return brightness < self.min_brightness or brightness > self.max_brightness
