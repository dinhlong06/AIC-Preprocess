"""
pipeline_d.py — Pipeline D: DAKE + MobileNetV3-Large.

Flow:
  Frames (toàn bộ shot)
    → DAKE (coarse temporal filter)
    → MobileNet (encode candidates)
    → Semantic Filter
    → Keyframes

Mục tiêu:
  Đánh giá Accuracy–Latency trade-off giữa BEiT-3 và MobileNet
  khi kết hợp với DAKE. So với Pipeline C:
    - Nhanh hơn đáng kể (MobileNet inference nhẹ hơn BEiT-3).
    - Ít VRAM hơn (~4-5x).
    - Chất lượng semantic thấp hơn một chút.

Cấu hình:
  pipeline: dake_mobilenet
  dake.enabled: true
  dake.candidate_ratio: 0.02
  dake.window_size: 3
  semantic.threshold: 0.90
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from src.core.interfaces import BaseKeyframeExtractor
from src.core.models import ShotRecord, ShotKeyframes, Keyframe
from src.components.frame_loader import VideoFrameLoader
from src.components.dake import DAKESelector
from src.components.mobilenet_encoder import MobileNetEncoder
from src.components.semantic_filter import SemanticFilter


class PipelineD(BaseKeyframeExtractor):
    """
    Pipeline D: DAKE → MobileNetV3-Large → Semantic Filter.

    Args:
        candidate_ratio      : Tỉ lệ frame DAKE giữ lại (default 0.02).
        window_size          : Sliding window size của DAKE (default 3).
        similarity_threshold : Cosine similarity threshold cho SemanticFilter.
        device               : "cuda" hoặc "cpu".
        batch_size           : Số frame encode mỗi batch MobileNet.
    """

    def __init__(
        self,
        candidate_ratio: float = 0.02,
        window_size: int = 3,
        similarity_threshold: float = 0.90,
        device: str | None = None,
        batch_size: int = 64,
    ):
        self.candidate_ratio = candidate_ratio
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.device = device
        self.batch_size = batch_size

        # DAKE không cần GPU
        self._dake = DAKESelector(
            candidate_ratio=candidate_ratio,
            window_size=window_size,
        )
        self._encoder: MobileNetEncoder | None = None
        self._filter = SemanticFilter(similarity_threshold)

    @property
    def name(self) -> str:
        """Tên định danh pipeline — dùng làm tên thư mục output."""
        return "pipeline_d"

    def setup(self) -> None:
        """Load MobileNetV3-Large. Gọi một lần trước vòng lặp video."""
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
        Trích xuất keyframe theo flow DAKE → MobileNet → Filter.

        Với mỗi shot:
          1. Đọc toàn bộ frame trong [start_frame, end_frame].
          2. DAKE: chọn Top-k Candidate Frames.
          3. MobileNet encode Candidates → 960-dim embedding.
          4. Semantic Filter → Keyframes cuối cùng.

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
                # Bước 1: Đọc toàn bộ frame
                all_frames = loader.read_range(shot.start_frame, shot.end_frame)
                if not all_frames:
                    results.append(ShotKeyframes(shot.video_id, shot.shot_id, []))
                    continue

                # Bước 2: DAKE chọn Candidate Frames
                candidate_indices = self._dake.select_candidates(all_frames)
                candidate_map = {idx: img for idx, img in all_frames}
                candidate_frames = [
                    (idx, candidate_map[idx])
                    for idx in candidate_indices
                    if idx in candidate_map
                ]
                if not candidate_frames:
                    candidate_frames = [all_frames[0]]

                # Bước 3: Encode bằng MobileNet
                enc_indices, embeddings = self._encoder.encode_batch(candidate_frames)

                # Bước 4: Semantic Filter
                selected, selected_embeddings, _ = self._filter.filter(enc_indices, embeddings)
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
