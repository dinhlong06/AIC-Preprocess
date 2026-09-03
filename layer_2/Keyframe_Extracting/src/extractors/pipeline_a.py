"""
pipeline_a.py — Pipeline A: Semantic Only với BEiT-3.

Flow:
  Frames (toàn bộ shot) → BEiT-3 encode → Semantic Filter → Keyframes

Đặc điểm:
  - Chất lượng semantic cao nhất (BEiT-3 Large, 1024-dim).
  - Không có bước pre-filter (encode toàn bộ frame).
  - Tốn VRAM và thời gian nhất trong 4 pipeline.
  - Dùng làm baseline chất lượng cao.

Cấu hình:
  pipeline: beit3
  dake.enabled: false
  semantic.threshold: 0.90 (configurable)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Inject beit3 source path trước khi import encoder
_beit3_src = Path(__file__).resolve().parents[2] / "unilm" / "beit3"
if str(_beit3_src) not in sys.path:
    sys.path.insert(0, str(_beit3_src))

from src.core.interfaces import BaseKeyframeExtractor
from src.core.models import ShotRecord, ShotKeyframes, Keyframe
from src.components.frame_loader import VideoFrameLoader
from src.components.beit3_encoder import BEiT3Encoder
from src.components.semantic_filter import SemanticFilter


class PipelineA(BaseKeyframeExtractor):
    """
    Pipeline A: BEiT-3 Semantic Only.

    Args:
        checkpoint_path      : Path đến beit3_large_patch16_224.pth.
        spm_path             : Path đến beit3.spm.
        similarity_threshold : Ngưỡng cosine similarity để lọc frame (default 0.90).
        device               : "cuda" hoặc "cpu".
        batch_size           : Số frame encode mỗi batch.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        spm_path: str | Path,
        similarity_threshold: float = 0.90,
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.spm_path = Path(spm_path)
        self.similarity_threshold = similarity_threshold
        self.device = device
        self.batch_size = batch_size

        self._encoder: BEiT3Encoder | None = None
        self._filter = SemanticFilter(similarity_threshold)

    @property
    def name(self) -> str:
        """Tên định danh pipeline — dùng làm tên thư mục output."""
        return "pipeline_a"

    def setup(self) -> None:
        """Load BEiT-3 model vào GPU/CPU. Gọi một lần trước vòng lặp video."""
        self._encoder = BEiT3Encoder(
            checkpoint_path=self.checkpoint_path,
            spm_path=self.spm_path,
            device=self.device,
            batch_size=self.batch_size,
        )
        self._encoder.load()

    def teardown(self) -> None:
        """Giải phóng BEiT-3 khỏi VRAM sau khi hoàn thành."""
        if self._encoder:
            self._encoder.unload()
            self._encoder = None

    def extract(
        self,
        video_path: Path,
        shots: List[ShotRecord],
    ) -> List[ShotKeyframes]:
        """
        Trích xuất keyframe cho tất cả shots trong video bằng BEiT-3.

        Với mỗi shot:
          1. Đọc toàn bộ frame trong [start_frame, end_frame].
          2. Encode bằng BEiT-3 → CLS embedding (1024-dim).
          3. Semantic Filter: loại frame có sim >= threshold.
          4. Lưu kết quả vào ShotKeyframes.

        Args:
            video_path : Đường dẫn video .mp4.
            shots      : Danh sách ShotRecord.

        Returns:
            List ShotKeyframes — mỗi shot có danh sách keyframe đã chọn.
        """
        results: List[ShotKeyframes] = []

        with VideoFrameLoader(video_path) as loader:
            fps = loader.fps

            for shot in shots:
                # Đọc toàn bộ frame trong shot
                frames = loader.read_range(shot.start_frame, shot.end_frame)
                if not frames:
                    results.append(ShotKeyframes(shot.video_id, shot.shot_id, []))
                    continue

                # Encode bằng BEiT-3
                indices, embeddings = self._encoder.encode_batch(frames)

                # Semantic Filter
                selected, selected_embeddings, _ = self._filter.filter(indices, embeddings)

                # Tạo Keyframe objects
                keyframes = _make_keyframes(selected, selected_embeddings, fps, shot.shot_id)
                results.append(ShotKeyframes(shot.video_id, shot.shot_id, keyframes))

        return results


def _make_keyframes(
    frame_indices: List[int],
    embeddings,
    fps: float,
    shot_id: str,
) -> List[Keyframe]:
    """
    Tạo list Keyframe dataclass từ danh sách frame_idx đã chọn.

    Args:
        frame_indices : List frame_idx được chọn.
        embeddings    : numpy array (N, D) — embeddings[i] ứng với frame_indices[i].
        fps           : FPS của video để tính timestamp.
        shot_id       : ID của shot hiện tại.

    Returns:
        List Keyframe với ID, timestamp, embedding, và path placeholder.
    """
    keyframes = []
    for i, (idx, emb) in enumerate(zip(frame_indices, embeddings)):
        ts_ms = int(idx / max(fps, 1.0) * 1000)
        kf = Keyframe(
            keyframe_id=f"{shot_id}_kf{i + 1:04d}",
            frame_idx=idx,
            timestamp_ms=ts_ms,
            image_path="",   # Runner sẽ điền path sau khi lưu ảnh
            embedding=emb,
        )
        keyframes.append(kf)
    return keyframes
