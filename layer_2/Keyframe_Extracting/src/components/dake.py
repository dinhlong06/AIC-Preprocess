"""
dake.py — Triển khai thuật toán DAKE (JPEG-size based Coarse Temporal Filter).

DAKE KHÔNG phải semantic model. Vai trò duy nhất của DAKE là:
  - Đọc frame từ shot.
  - Mã hóa JPEG từng frame và lấy kích thước file.
  - Tính Steepness (mức độ thay đổi giữa các frame liên tiếp).
  - Làm mượt bằng Sliding Window (Local Aggregation).
  - Xếp hạng và trả về Top-k frame làm "Candidate Frames".

Candidate Frames này sau đó sẽ được đưa vào BEiT-3 hoặc MobileNet để
semantic filtering — DAKE không quyết định Keyframe cuối cùng.

Tham khảo: KEYFRAME_EXTRACTING.md, Section 6.
"""

from __future__ import annotations

import io
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

# cv2.imencode nhả GIL nên thread thật sự chạy song song. Đo trên frame 1080p:
# tuần tự 10.4 ms/frame, 4 thread 3.1, 8 thread 1.7 — kết quả giống hệt từng byte.
# Máy dùng chung và sẽ chạy nhiều shard video song song, nên để chỉnh qua env.
_JPEG_THREADS = int(os.environ.get("DAKE_THREADS", "2"))


class DAKESelector:
    """
    Triển khai thuật toán DAKE để chọn Candidate Frames từ một shot.

    Không dùng CNN, không dùng GPU — chỉ dùng JPEG compression size.

    Args:
        candidate_ratio : Tỉ lệ frame giữ lại (ví dụ 0.02 = giữ 2%).
        window_size     : Kích thước cửa sổ sliding window (phải lẻ).
        jpeg_quality    : Chất lượng nén JPEG khi đo kích thước (1–100).
    """

    def __init__(
        self,
        candidate_ratio: float = 0.02,
        window_size: int = 3,
        jpeg_quality: int = 75,
    ):
        self.candidate_ratio = candidate_ratio
        self.window_size = window_size
        self.jpeg_quality = jpeg_quality

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_with_scores(
        self,
        frames: List[Tuple[int, np.ndarray]],
    ) -> Tuple[List[int], List[float], List[float]]:
        """
        Nhận vào list (frame_idx, BGR image) của một shot, trả về
        (candidate_indices, steepness, aggregated).

        Pipeline G cần steepness (chọn transition peak) và aggregated (score-aware
        diversity NMS); tính lại chúng ở ngoài là encode JPEG lần hai cho cùng một
        bộ frame — 6.4 ms/frame tuần tự, gấp đôi chi phí đắt nhất của DAKE.

        Returns:
            Tuple (candidate_indices, steepness, aggregated) — hai list sau cùng
            độ dài và cùng thứ tự với frames.
        """
        if not frames:
            return [], [], []

        # Bước 1: Tính JPEG size của từng frame
        indices = [idx for idx, _ in frames]
        with ThreadPoolExecutor(_JPEG_THREADS) as pool:
            sizes = list(pool.map(self._compute_jpeg_size, (img for _, img in frames)))

        # Bước 2: Tính Steepness (độ thay đổi giữa các frame liên tiếp)
        steepness = self._compute_steepness(sizes)

        # Bước 3: Local Aggregation — làm mượt bằng Sliding Window
        aggregated = self._aggregate_scores(steepness, self.window_size)

        # Bước 4: Xếp hạng và chọn Top-k
        k = max(1, int(len(frames) * self.candidate_ratio))
        candidate_indices = self._select_top_k(indices, aggregated, k)

        # Frame đầu shot = cảnh mới sau cut, nhưng steepness của nó bị gán 0.0
        # (không có frame trước để so trong list) nên gần như luôn out top-k.
        # Ép giữ làm candidate đại diện mở cảnh; DAKE lo phần motion bên trong.
        return sorted(set(candidate_indices) | {indices[0]}), steepness, aggregated

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _compute_jpeg_size(self, frame: np.ndarray) -> float:
        """
        Mã hóa frame thành JPEG và trả về kích thước (bytes).

        Frame phức tạp (nhiều chi tiết) → JPEG size lớn hơn.
        Frame đơn giản (ít thay đổi) → JPEG size nhỏ hơn.

        Args:
            frame : BGR image dạng numpy array.

        Returns:
            Kích thước JPEG (bytes) dưới dạng float.
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, buf = cv2.imencode(".jpg", frame, encode_param)
        return float(len(buf))

    @staticmethod
    def _compute_steepness(sizes: List[float]) -> List[float]:
        """
        Tính Steepness = |size[i] - size[i-1]| / max(size[i-1], 1).

        Steepness cao → thay đổi nhiều giữa 2 frame liên tiếp.
        Giá trị được normalize để giảm ảnh hưởng của resolution và bitrate.

        Args:
            sizes : Danh sách JPEG size của từng frame.

        Returns:
            Danh sách steepness, cùng độ dài với sizes.
            (Frame đầu tiên có steepness = 0).
        """
        steepness = [0.0]
        for i in range(1, len(sizes)):
            delta = abs(sizes[i] - sizes[i - 1])
            normalized = delta / max(sizes[i - 1], 1.0)
            steepness.append(normalized)
        return steepness

    @staticmethod
    def _aggregate_scores(
        scores: List[float],
        window_size: int,
    ) -> List[float]:
        """
        Làm mượt danh sách scores bằng Sliding Window (mean pooling).

        Args:
            scores      : Danh sách giá trị cần làm mượt.
            window_size : Kích thước cửa sổ (khuyến nghị số lẻ).

        Returns:
            Danh sách scores đã được làm mượt, cùng độ dài với input.
        """
        n = len(scores)
        half = window_size // 2
        aggregated = []
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n - 1, i + half)
            window = scores[lo : hi + 1]
            aggregated.append(sum(window) / len(window))
        return aggregated

    @staticmethod
    def _select_top_k(
        indices: List[int],
        scores: List[float],
        k: int,
    ) -> List[int]:
        """
        Xếp hạng frame theo aggregated score và trả về Top-k frame_idx.

        Args:
            indices : Danh sách frame_idx tương ứng với scores.
            scores  : Aggregated steepness scores.
            k       : Số frame cần giữ lại.

        Returns:
            List frame_idx của Top-k frame (chưa sort theo thời gian).
        """
        paired = sorted(zip(scores, indices), reverse=True)
        top_k = [idx for _, idx in paired[:k]]
        return top_k
