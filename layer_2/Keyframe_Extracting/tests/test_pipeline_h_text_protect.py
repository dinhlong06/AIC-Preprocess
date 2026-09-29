"""
Regression test: frame mà vùng text vừa xuất hiện phải ra tới output, kể cả khi
whole-frame embedding của nó trùng khớp keyframe trước.

Đây chính là điểm mù của pipeline_g: DAKE chấm theo steepness của kích thước
frame nén (proxy cho chuyển động, mù với "có chữ"), còn Semantic Filter so
BEiT-3 whole-frame — frame chỉ khác ở cái banner chữ có cosine ~0.98 nên bị loại
trước cả khi tới Diversity Filter. Test cố định cả hai đầu: pipeline_g bỏ sót,
pipeline_h giữ.
"""

import numpy as np
import cv2
import pytest

from src.core.models import ShotRecord
from src.extractors.pipeline_g import PipelineG
from src.extractors.pipeline_h import PipelineH

STATIC_VEC = np.array([1.0, 0.0], dtype=np.float32)

_W, _H = 160, 96
_TEXT_ONSET = 200
_NUM_FRAMES = 400
_FPS = 25.0


class _FakeEncoder:
    """Mọi frame cùng một embedding — mô phỏng đúng ca tệ nhất: thêm banner chữ
    gần như không đổi whole-frame embedding."""

    def load(self):
        pass

    def unload(self):
        pass

    def encode_batch(self, frames):
        indices = [idx for idx, _ in frames]
        if not indices:
            return [], np.empty((0, 2), dtype=np.float32)
        return indices, np.stack([STATIC_VEC.copy() for _ in indices]).astype(np.float32)


def _frame(i: int) -> np.ndarray:
    """Nền phẳng xám (edge energy thấp đều). Từ _TEXT_ONSET trở đi có thêm dải
    chữ ở đáy: các vạch dọc trắng, đúng thứ Sobel-x dò ra."""
    img = np.full((_H, _W, 3), 120, dtype=np.uint8)
    img[:, :, 0] = 110 + (i % 3)  # nhiễu nhẹ để DAKE không thấy video đứng hình tuyệt đối
    if i >= _TEXT_ONSET:
        for x in range(6, _W - 6, 4):
            img[_H - 20 : _H - 6, x : x + 2] = 255
    return img


@pytest.fixture
def text_onset_video(tmp_path):
    path = tmp_path / "text_onset.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), _FPS, (_W, _H))
    for i in range(_NUM_FRAMES):
        writer.write(_frame(i))
    writer.release()
    return path


def _kwargs():
    return dict(
        checkpoint_path="unused",
        spm_path="unused",
        min_frame_distance=5,
        similarity_threshold=0.90,
        redundancy_threshold=0.88,
        # Tắt coverage ceiling: test này phải chứng minh text_prescan cứu frame,
        # không phải force-pick tình cờ cứu hộ.
        gap_decay_start_frames=10_000,
        max_gap_frames=10_000,
        # Nền phẳng cần cho Sobel band detection lại đúng là thứ blank veto loại.
        enable_veto=False,
        enable_sharpness=False,
    )


def _run(pipeline, video):
    pipeline._encoder = _FakeEncoder()
    shot = ShotRecord(video_id="T", shot_id="S0", start_frame=0, end_frame=_NUM_FRAMES - 1, fps=_FPS)
    return sorted(kf.frame_idx for kf in pipeline.extract(video, [shot])[0].keyframes)


def test_pipeline_g_misses_the_text_onset(text_onset_video):
    kf = _run(PipelineG(**_kwargs()), text_onset_video)
    assert not any(idx >= _TEXT_ONSET for idx in kf), (
        f"pipeline_g bỗng nhiên bắt được text onset ({kf}) — test không còn cố định "
        "điểm mù mà nó sinh ra để bảo vệ"
    )


def test_pipeline_h_keeps_the_text_onset(text_onset_video):
    kf = _run(PipelineH(**_kwargs(), text_stride_seconds=1.0, text_min_gap_seconds=2.0), text_onset_video)
    assert any(idx >= _TEXT_ONSET for idx in kf), f"text onset vẫn bị lọc mất: {kf}"


def test_disabled_prescan_matches_pipeline_g(text_onset_video):
    """enable_text_prescan=False phải cho ra đúng kết quả pipeline_g — mọi chênh
    lệch khi A/B chỉ được đến từ text_prescan."""
    g = _run(PipelineG(**_kwargs()), text_onset_video)
    h = _run(PipelineH(**_kwargs(), enable_text_prescan=False), text_onset_video)
    assert g == h
