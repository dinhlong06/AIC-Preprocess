"""
text_prescan.py — Đánh dấu frame mà VÙNG TEXT thay đổi.

DAKE chấm điểm theo steepness của kích thước frame nén — proxy cho "chuyển động",
mù hoàn toàn với "có chữ". Frame chỉ khác frame trước ở chỗ có thêm banner chữ thì
BEiT-3 whole-frame cosine ~0.98, bị Semantic Filter loại trước cả khi tới
Diversity Filter. Set trả về ở đây được pipeline_h dùng để miễn cho chúng hai
filter đó.

Dò dải text bằng Sobel row-energy chứ không dùng OCR detector: layer_2 không có
paddlepaddle trong image, và thứ cần ở đây chỉ là "vùng text có đổi không", không
phải box thật. Cùng ý tưởng với layer_3/OCR/ocr/frame_skip.py:crop_news_bands().
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional, Set, Tuple

_HASH_W, _HASH_H = 9, 8  # dHash 8x8 = 64 bit


class TextRegionScanner:
    """
    Quét một cửa sổ frame và trả về các frame_idx mà dải text vừa xuất hiện hoặc
    vừa đổi nội dung.

    Args:
        stride            : Lấy mẫu mỗi N frame. Text bản tin sống vài giây nên
                            ~1 giây là đủ; caller quy đổi từ giây bằng fps.
        band_ratio        : Dải top/bottom được coi là có text khi edge energy
                            của nó >= band_ratio lần energy giữa khung.
        change_threshold  : Hamming distance tối thiểu giữa 2 dHash liên tiếp để
                            coi là "chữ đã đổi" (0-64).
        min_gap_frames    : Khoảng cách tối thiểu giữa 2 frame được đánh dấu.
                            Chân chạy (ticker) đổi liên tục từng frame — không có
                            chặn này thì mỗi mẫu đều được đánh dấu.
        top_pct           : Tỉ lệ chiều cao tính là dải trên.
        bottom_pct        : Tỉ lệ chiều cao tính là dải dưới.
    """

    def __init__(
        self,
        stride: int,
        band_ratio: float = 2.0,
        change_threshold: int = 8,
        min_gap_frames: int = 50,
        top_pct: float = 0.12,
        bottom_pct: float = 0.28,
    ):
        self.stride = max(1, int(stride))
        self.band_ratio = band_ratio
        self.change_threshold = change_threshold
        self.min_gap_frames = min_gap_frames
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct

    def scan(self, frames: List[Tuple[int, np.ndarray]]) -> Set[int]:
        """
        Args:
            frames : List (frame_idx, BGR image) của cả cửa sổ, đã nằm sẵn trong RAM.

        Returns:
            Set frame_idx có vùng text vừa xuất hiện/đổi nội dung.
        """
        protected: Set[int] = set()
        prev_sig: Optional[int] = None
        # Mẫu đầu cửa sổ chỉ làm mốc so sánh: prev_sig=None ở đó nghĩa là "chưa có
        # mẫu trước", không phải "trước đó không có chữ" — đánh dấu nó sẽ biến mọi
        # banner tĩnh thành keyframe ở mỗi biên cửa sổ 1200 frame.
        is_first = True
        last_marked: Optional[int] = None

        for idx, img in frames[:: self.stride]:
            sig = self._band_signature(img)
            if (
                not is_first
                and sig is not None
                and (prev_sig is None or _hamming(prev_sig, sig) >= self.change_threshold)
                and (last_marked is None or idx - last_marked >= self.min_gap_frames)
            ):
                protected.add(idx)
                last_marked = idx
            prev_sig, is_first = sig, False

        return protected

    def _band_signature(self, img: np.ndarray) -> Optional[int]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        row_energy = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)), axis=1)

        top_end = max(1, int(h * self.top_pct))
        bot_start = min(h - 1, int(h * (1.0 - self.bottom_pct)))
        mid_energy = float(np.mean(row_energy[top_end:bot_start])) + 1e-6

        bands = []
        if float(np.mean(row_energy[:top_end])) / mid_energy >= self.band_ratio:
            bands.append(gray[:top_end])
        if float(np.mean(row_energy[bot_start:])) / mid_energy >= self.band_ratio:
            bands.append(gray[bot_start:])

        if not bands:
            return None
        return _dhash(np.vstack(bands) if len(bands) > 1 else bands[0])


def _dhash(band: np.ndarray) -> int:
    small = cv2.resize(band, (_HASH_W, _HASH_H), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = small[:, 1:] > small[:, :-1]
    return int.from_bytes(np.packbits(bits.ravel()).tobytes(), "big")


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
