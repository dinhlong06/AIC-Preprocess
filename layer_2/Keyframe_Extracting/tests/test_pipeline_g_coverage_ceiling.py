"""
Regression test: frame force-pick bởi coverage ceiling (Stage 3) không được để
Diversity Filter (Stage 4) xoá mất. Bug thực tế đã gặp: mọi frame force-pick
trong 1 cảnh tĩnh đều rất giống nhau (đúng bản chất cảnh tĩnh) nên Diversity
Filter (redundancy_threshold=0.88) coi chúng là trùng lặp và xoá gần hết,
khiến coverage ceiling không còn tác dụng trên video thật (đã tái hiện và xác
nhận bằng debug script chạy trên L25_V036 shot dài 20726 frame).
"""

import numpy as np
import cv2
import pytest

from src.core.models import ShotRecord
from src.extractors.pipeline_g import PipelineG

# Toàn bộ frame dùng CHUNG 1 embedding -> mô phỏng cảnh tĩnh tuyệt đối:
# similarity check sẽ luôn loại, chỉ force-pick mới chọn được frame nào.
STATIC_VEC = np.array([1.0, 0.0], dtype=np.float32)


class _FakeEncoder:
    def load(self):
        pass

    def unload(self):
        pass

    def encode_batch(self, frames):
        indices = [idx for idx, _ in frames]
        if not indices:
            return [], np.empty((0, 2), dtype=np.float32)
        embeddings = np.stack([STATIC_VEC.copy() for _ in indices]).astype(np.float32)
        return indices, embeddings


def _make_synthetic_video(path, num_frames=700, size=32, fps=25.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    rng = np.random.default_rng(7)
    for _ in range(num_frames):
        frame = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def static_video(tmp_path):
    video_path = tmp_path / "static.mp4"
    _make_synthetic_video(video_path, num_frames=700)
    return video_path


def test_forced_frames_survive_diversity_filter(static_video):
    pipeline = PipelineG(
        checkpoint_path="unused",
        spm_path="unused",
        min_frame_distance=5,
        similarity_threshold=0.90,
        redundancy_threshold=0.88,  # threshold thật, đủ thấp để bug (nếu tái xuất hiện) lộ ra
        gap_decay_start_frames=50,
        max_gap_frames=100,
        enable_sharpness=False,
    )
    pipeline._encoder = _FakeEncoder()

    shot = ShotRecord(video_id="T", shot_id="S0", start_frame=0, end_frame=699, fps=25.0)
    results = pipeline.extract(static_video, [shot])

    kf_indices = sorted(kf.frame_idx for kf in results[0].keyframes)

    # Cảnh tĩnh 700 frame, max_gap_frames=100 -> phải có nhiều hơn 1 keyframe
    # (nếu Diversity Filter xoá mất các frame force-pick, chỉ còn đúng 1 —
    # chính là bug đã xảy ra trên video thật).
    assert len(kf_indices) >= 5, (
        f"coverage ceiling bị Diversity Filter xoá mất — chỉ còn {kf_indices}"
    )
    assert kf_indices[0] == 0  # shot.start_frame vẫn phải có mặt

    prev = 0
    for idx in kf_indices:
        assert idx - prev <= 105, f"gap {prev}->{idx} vượt max_gap_frames"
        prev = idx
