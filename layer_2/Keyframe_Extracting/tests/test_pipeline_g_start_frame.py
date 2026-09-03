"""
Test tích hợp: shot.start_frame chỉ được ép làm *candidate encode*, không còn
bypass Semantic/Diversity Filter — nó có thể bị lọc như mọi candidate khác.
Nhưng mỗi shot vẫn phải có >=1 keyframe: nếu toàn bộ candidate trong shot (kể
cả start_frame) đều bị lọc vì trùng embedding với keyframe trước, pipeline phải
fallback về chính start_frame để shot không bao giờ trống trơn.

Dùng encoder giả (không load BEiT-3 thật, quá nặng cho unit test) để kiểm soát
chính xác embedding trả về theo frame_idx.
"""

import numpy as np
import cv2
import pytest

from src.core.models import ShotRecord
from src.extractors.pipeline_g import PipelineG

VEC_A = np.array([1.0, 0.0], dtype=np.float32)  # embedding "keyframe cuối shot trước"
VEC_B = np.array([0.0, 1.0], dtype=np.float32)  # embedding khác biệt hoàn toàn


class _FakeEncoder:
    """Thay BEiT3Encoder thật — trả embedding theo frame_idx do test kiểm soát."""

    def __init__(self, embed_fn):
        self._embed_fn = embed_fn

    def load(self):
        pass

    def unload(self):
        pass

    def encode_batch(self, frames):
        indices = [idx for idx, _ in frames]
        if not indices:
            return [], np.empty((0, 2), dtype=np.float32)
        embeddings = np.stack([self._embed_fn(idx) for idx in indices]).astype(np.float32)
        return indices, embeddings


def _make_synthetic_video(path, num_frames=20, size=32, fps=25.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    rng = np.random.default_rng(42)
    for _ in range(num_frames):
        frame = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def synthetic_video(tmp_path):
    video_path = tmp_path / "synthetic.mp4"
    _make_synthetic_video(video_path)
    return video_path


def _embed_fn(idx: int) -> np.ndarray:
    # Toàn bộ shot 1 (0-9) VÀ frame đầu shot 2 (10) dùng chung embedding VEC_A
    # -> mô phỏng đúng tình huống "frame đầu shot mới giống hệt keyframe trước".
    if idx < 10 or idx == 10:
        return VEC_A
    return VEC_B


def _embed_fn_all_duplicate(idx: int) -> np.ndarray:
    # Toàn bộ shot 1 VÀ shot 2 dùng chung VEC_A -> mọi candidate của shot 2,
    # kể cả start_frame, đều bị Semantic Filter loại vì trùng lịch sử. Chỉ cơ
    # chế fallback-khi-shot-trống mới cứu được shot 2 khỏi 0 keyframe.
    return VEC_A


def test_shot_falls_back_to_start_frame_when_everything_filtered(synthetic_video):
    pipeline = PipelineG(
        checkpoint_path="unused",
        spm_path="unused",
        min_frame_distance=2,
        similarity_threshold=0.90,
        enable_sharpness=False,  # không liên quan tới test này
    )
    pipeline._encoder = _FakeEncoder(_embed_fn_all_duplicate)  # bỏ qua setup()/BEiT-3 thật

    shots = [
        ShotRecord(video_id="T", shot_id="S0", start_frame=0, end_frame=9, fps=25.0),
        ShotRecord(video_id="T", shot_id="S1", start_frame=10, end_frame=19, fps=25.0),
    ]

    results = pipeline.extract(synthetic_video, shots)

    assert len(results) == 2
    shot1_indices = [kf.frame_idx for kf in results[0].keyframes]
    shot2_indices = [kf.frame_idx for kf in results[1].keyframes]

    # Cả 2 shot đều phải có ít nhất 1 keyframe — shot 2 không được rớt về 0
    # dù toàn bộ candidate của nó trùng embedding với lịch sử.
    assert 0 in shot1_indices
    assert shot2_indices == [10], (
        "shot 2 có embedding trùng hệt shot 1 (VEC_A) trên toàn bộ candidate — "
        f"fallback phải trả về đúng [start_frame]=[10]. Thực tế: {shot2_indices}"
    )


def test_other_shot2_candidates_still_go_through_normal_filtering(synthetic_video):
    """Các candidate khác trong shot 2 (embedding VEC_B, khác biệt) vẫn phải đi
    qua Semantic + Diversity Filter như cũ — không bị vô hiệu hoá bởi cơ chế
    ép chọn frame đầu shot."""
    pipeline = PipelineG(
        checkpoint_path="unused",
        spm_path="unused",
        min_frame_distance=2,
        similarity_threshold=0.90,
        candidate_ratio=0.5,
        enable_sharpness=False,
    )
    pipeline._encoder = _FakeEncoder(_embed_fn)

    shots = [
        ShotRecord(video_id="T", shot_id="S0", start_frame=0, end_frame=9, fps=25.0),
        ShotRecord(video_id="T", shot_id="S1", start_frame=10, end_frame=19, fps=25.0),
    ]
    results = pipeline.extract(synthetic_video, shots)
    shot2_indices = [kf.frame_idx for kf in results[1].keyframes]

    # VEC_B đủ khác VEC_A nên nếu DAKE/transition chọn được candidate nào khác
    # 10 trong shot 2, nó phải được giữ lại (không bị coi là trùng lặp).
    others = [idx for idx in shot2_indices if idx != 10]
    assert all(idx > 10 for idx in others)
