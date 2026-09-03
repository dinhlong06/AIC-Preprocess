"""
pipeline_b.py — Pipeline B: Semantic Only với MobileNetV3-Large.

Flow:
  Frames (toàn bộ shot) → MobileNet encode → Semantic Filter → Keyframes

Đặc điểm:
  - Nhanh hơn Pipeline A (~8-10x), ít VRAM hơn (~5x).
  - Không cần checkpoint thủ công (dùng ImageNet pretrained từ torchvision).
  - Chất lượng semantic thấp hơn BEiT-3 nhưng đủ dùng cho benchmark.
  - Dùng để đánh giá Accuracy–Latency trade-off.

Cấu hình:
  pipeline: mobilenet
  dake.enabled: false
  semantic.threshold: 0.90 (configurable)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from src.core.interfaces import BaseKeyframeExtractor
from src.core.models import ShotRecord, ShotKeyframes, Keyframe
from src.components.frame_loader import VideoFrameLoader
from src.components.mobilenet_encoder import MobileNetEncoder
from src.components.semantic_filter import SemanticFilter


class PipelineB(BaseKeyframeExtractor):
    """
    Pipeline B: MobileNetV3-Large Semantic Only.

    Args:
        similarity_threshold : Ngưỡng cosine similarity để lọc frame (default 0.90).
        device               : "cuda" hoặc "cpu".
        batch_size           : Số frame encode mỗi batch.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.90,
        device: str | None = None,
        batch_size: int = 64,
    ):
        self.similarity_threshold = similarity_threshold
        self.device = device
        self.batch_size = batch_size

        self._encoder: MobileNetEncoder | None = None
        self._filter = SemanticFilter(similarity_threshold)

    @property
    def name(self) -> str:
        """Tên định danh pipeline — dùng làm tên thư mục output."""
        return "pipeline_b"

    def setup(self) -> None:
        """Load MobileNetV3-Large từ torchvision. Gọi một lần trước vòng lặp."""
        self._encoder = MobileNetEncoder(
            device=self.device,
            batch_size=self.batch_size,
        )
        self._encoder.load()

    def teardown(self) -> None:
        """Giải phóng MobileNet khỏi VRAM sau khi hoàn thành."""
        if self._encoder:
            self._encoder.unload()
            self._encoder = None

    def extract(
        self,
        video_path: Path,
        shots: List[ShotRecord],
    ) -> List[ShotKeyframes]:
        """
        Trích xuất keyframe bằng MobileNet cho tất cả shots trong video.

        Với mỗi shot:
          1. Đọc toàn bộ frame trong [start_frame, end_frame].
          2. Encode bằng MobileNetV3-Large → 960-dim embedding.
          3. Semantic Filter: loại frame có sim >= threshold.

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
                frames = loader.read_range(shot.start_frame, shot.end_frame)
                if not frames:
                    results.append(ShotKeyframes(shot.video_id, shot.shot_id, []))
                    continue

                indices, embeddings = self._encoder.encode_batch(frames)
                selected, selected_embeddings, _ = self._filter.filter(indices, embeddings)
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
            image_path="",
            embedding=emb,
        )
        keyframes.append(kf)
    return keyframes
