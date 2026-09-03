"""
pipeline_c.py — Pipeline C: DAKE + BEiT-3 (Pipeline đề xuất chính).

Flow:
  Frames (toàn bộ shot)
    → DAKE (coarse temporal filter, chọn Top-k candidates)
    → BEiT-3 (encode candidates)
    → Semantic Filter
    → Keyframes

Ưu điểm so với Pipeline A:
  - DAKE giảm số frame cần encode bằng BEiT-3 (candidate_ratio × total_frames).
  - Giảm thời gian inference và VRAM so với encode toàn bộ frame.
  - Chất lượng keyframe cao hơn vì DAKE ưu tiên frame "interesting" (nhiều thay đổi).
  - Đây là pipeline được khuyến nghị cho AI Challenge 2026.

Cấu hình:
  pipeline: dake_beit3
  dake.enabled: true
  dake.candidate_ratio: 0.02
  dake.window_size: 3
  semantic.threshold: 0.90
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Inject beit3 source path
_beit3_src = Path(__file__).resolve().parents[2] / "unilm" / "beit3"
if str(_beit3_src) not in sys.path:
    sys.path.insert(0, str(_beit3_src))

from src.core.interfaces import BaseKeyframeExtractor
from src.core.models import ShotRecord, ShotKeyframes, Keyframe
from src.components.frame_loader import VideoFrameLoader
from src.components.dake import DAKESelector
from src.components.beit3_encoder import BEiT3Encoder
from src.components.semantic_filter import SemanticFilter


class PipelineC(BaseKeyframeExtractor):
    """
    Pipeline C: DAKE → BEiT-3 → Semantic Filter (pipeline đề xuất chính).

    Args:
        checkpoint_path      : Path đến beit3_large_patch16_224.pth.
        spm_path             : Path đến beit3.spm.
        candidate_ratio      : Tỉ lệ frame DAKE giữ lại (default 0.02 = 2%).
        window_size          : Sliding window size của DAKE (default 3).
        similarity_threshold : Cosine similarity threshold cho SemanticFilter.
        device               : "cuda" hoặc "cpu".
        batch_size           : Số frame encode mỗi batch BEiT-3.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        spm_path: str | Path,
        candidate_ratio: float = 0.02,
        window_size: int = 3,
        similarity_threshold: float = 0.90,
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.spm_path = Path(spm_path)
        self.candidate_ratio = candidate_ratio
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.device = device
        self.batch_size = batch_size

        # DAKE không cần GPU — khởi tạo luôn
        self._dake = DAKESelector(
            candidate_ratio=candidate_ratio,
            window_size=window_size,
        )
        self._encoder: BEiT3Encoder | None = None
        self._filter = SemanticFilter(similarity_threshold)

    @property
    def name(self) -> str:
        """Tên định danh pipeline — dùng làm tên thư mục output."""
        return "pipeline_c"

    def setup(self) -> None:
        """Load BEiT-3 model. Gọi một lần trước vòng lặp video."""
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
        Trích xuất keyframe cho tất cả shots theo flow DAKE → BEiT-3 → Filter.

        Với mỗi shot:
          1. Đọc toàn bộ frame trong [start_frame, end_frame].
          2. DAKE: tính JPEG steepness, chọn Top-k Candidate Frames.
          3. Đọc lại chỉ Candidate Frames (random access).
          4. BEiT-3 encode Candidates → 1024-dim embedding.
          5. Semantic Filter → Keyframes cuối cùng.

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
                # Bước 1: Đọc toàn bộ frame trong shot
                all_frames = loader.read_range(shot.start_frame, shot.end_frame)
                if not all_frames:
                    results.append(ShotKeyframes(shot.video_id, shot.shot_id, []))
                    continue

                # Bước 2: DAKE chọn Candidate Frames
                candidate_indices = self._dake.select_candidates(all_frames)

                # Bước 3: Lấy chỉ Candidate Frames từ all_frames
                candidate_map = {idx: img for idx, img in all_frames}
                candidate_frames = [
                    (idx, candidate_map[idx])
                    for idx in candidate_indices
                    if idx in candidate_map
                ]

                if not candidate_frames:
                    # Fallback: giữ frame đầu shot nếu DAKE không chọn được gì
                    candidate_frames = [all_frames[0]]

                # Bước 4: Encode bằng BEiT-3
                enc_indices, embeddings = self._encoder.encode_batch(candidate_frames)

                # Bước 5: Semantic Filter
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
