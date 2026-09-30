"""
frame_loader.py — Đọc frame từ video hoặc thư mục ảnh.

Hỗ trợ hai chế độ theo spec KEYFRAME_EXTRACTING.md:
  Mode 1 — Standard: đọc frame từ [start_frame, end_frame] của video .mp4
  Mode 2 — Experimental: đọc keyframe BTC cắt sẵn trong thư mục webp.

Trả về list (frame_idx, numpy_array BGR).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


# Type alias: (frame_index_absolute, BGR_image)
FrameTuple = Tuple[int, np.ndarray]


class VideoFrameLoader:
    """
    Đọc frame từ file video .mp4 trong khoảng [start_frame, end_frame].

    Sử dụng OpenCV VideoCapture. Hiệu quả nhất khi đọc tuần tự,
    nhưng cũng hỗ trợ random access qua cv2.CAP_PROP_POS_FRAMES.
    """

    frame_step = 1

    def __init__(self, video_path: Path):
        """
        Args:
            video_path : Đường dẫn đến file .mp4.
        """
        self.video_path = Path(video_path)
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Mở VideoCapture. Phải gọi trước khi read."""
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

    def close(self) -> None:
        """Giải phóng VideoCapture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        """FPS của video. Cần gọi open() trước."""
        if self._cap is None:
            raise RuntimeError("VideoCapture chưa được mở. Gọi open() trước.")
        return self._cap.get(cv2.CAP_PROP_FPS) or 25.0

    def read_range(
        self,
        start_frame: int,
        end_frame: int,
        stride: int = 1,
    ) -> List[FrameTuple]:
        """
        Đọc tuần tự các frame trong [start_frame, end_frame] với bước nhảy stride.

        Args:
            start_frame : Frame bắt đầu (inclusive).
            end_frame   : Frame kết thúc (inclusive).
            stride      : Chỉ lấy mỗi stride frame (1 = lấy tất cả).

        Returns:
            List các (frame_idx, BGR image). Frame không đọc được sẽ bị bỏ qua.
        """
        if self._cap is None:
            raise RuntimeError("Gọi open() trước khi đọc frame.")

        # Shot liền kề nhau nên sau khi đọc hết shot trước con trỏ đã đúng chỗ.
        # Mỗi seek tốn 118 ms (nhảy về I-frame rồi decode xuôi lại) nên chỉ seek khi lệch.
        if int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) != start_frame:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames: List[FrameTuple] = []

        for idx in range(start_frame, end_frame + 1):
            ret, frame = self._cap.read()
            if not ret:
                break
            if (idx - start_frame) % stride == 0:
                frames.append((idx, frame))

        return frames

    def __enter__(self) -> "VideoFrameLoader":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()


class KeyframeDirLoader:
    """
    Đọc keyframe BTC cắt sẵn (kf_batch2/<video_id>/frame_XXX.webp + metadata.json) thay
    cho video: cùng interface VideoFrameLoader nhưng frame_idx lấy từ trường "id" nên
    chỉ trả về các frame BTC đã cắt (cách nhau >= 5 frame).
    """

    # BTC cắt tối thiểu mỗi 5 frame: knob tính theo "mỗi N frame lấy mẫu" phải chia cho số này.
    frame_step = 5

    def __init__(self, video_dir: Path):
        self.video_dir = Path(video_dir)
        # Không phải thư mục video nào BTC cũng kèm metadata.json riêng; file gộp ở gốc
        # (~150 MB) thì luôn đủ nhưng nạp chậm nên chỉ dùng khi thiếu.
        own = self.video_dir / "metadata.json"
        meta = json.loads((own if own.exists() else self.video_dir.parent / "metadata.json").read_text(encoding="utf-8"))
        frames = meta[self.video_dir.name]
        self._files = {f["id"]: name for name, f in frames.items()}
        self.fps = next(iter(frames.values()))["fps"]

    def read_range(self, start_frame: int, end_frame: int) -> List[FrameTuple]:
        indices = sorted(i for i in self._files if start_frame <= i <= end_frame)
        return [
            (idx, img)
            for idx, img in zip(indices, self._decode.map(lambda i: cv2.imdecode(
                np.frombuffer(self._blobs[i].result(), np.uint8), cv2.IMREAD_COLOR), indices))
            if img is not None
        ]

    def __enter__(self) -> "KeyframeDirLoader":
        # NAS nghẽn ở độ trễ (~6 file/s tuần tự, 128 luồng ~60 file/s): đặt đọc CẢ video
        # ngay từ đầu để I/O chạy chồng lên phần tính của các cửa sổ trước. Chỉ giữ bytes
        # nén (~450 MB/video), decode lúc cần.
        self._io = ThreadPoolExecutor(128)
        self._decode = ThreadPoolExecutor(4)
        self._blobs = {
            i: self._io.submit((self.video_dir / f"{name}.webp").read_bytes)
            for i, name in sorted(self._files.items())
        }
        return self

    def __exit__(self, *args) -> None:
        self._io.shutdown(cancel_futures=True)
        self._decode.shutdown()
