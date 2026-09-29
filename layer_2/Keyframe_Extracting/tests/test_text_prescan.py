import numpy as np

from src.components.text_prescan import TextRegionScanner

_H, _W = 100, 160
_BAND_TOP = int(_H * (1.0 - 0.28))  # dải dưới bắt đầu ở đây


def _plain() -> np.ndarray:
    """Nền có chi tiết nhưng phân bố đều — không dải nào nổi bật."""
    rng = np.random.default_rng(0)
    return rng.integers(60, 90, size=(_H, _W, 3), dtype=np.uint8)


def _with_text(seed: int) -> np.ndarray:
    """Nền như trên + dải dưới nhiều cạnh dọc (mô phỏng chữ); seed đổi = chữ đổi."""
    img = _plain()
    rng = np.random.default_rng(seed)
    for x in rng.choice(_W - 2, size=40, replace=False):
        img[_BAND_TOP + 4 : _H - 4, x : x + 2] = 255
    return img


def test_marks_frame_where_text_appears():
    scanner = TextRegionScanner(stride=10, min_gap_frames=1)
    frames = [(i, _plain()) for i in range(20)] + [(i, _with_text(1)) for i in range(20, 40)]
    assert scanner.scan(frames) == {20}


def test_static_text_marked_once_not_every_sample():
    """Banner đứng yên suốt shot: chỉ frame nó xuất hiện mới đáng làm keyframe."""
    scanner = TextRegionScanner(stride=10, min_gap_frames=1)
    frames = [(i, _plain()) for i in range(10)] + [(i, _with_text(1)) for i in range(10, 100)]
    assert scanner.scan(frames) == {10}


def test_changed_text_marked_again():
    scanner = TextRegionScanner(stride=10, min_gap_frames=1)
    frames = (
        [(i, _with_text(1)) for i in range(0, 30)]
        + [(i, _with_text(2)) for i in range(30, 60)]
    )
    assert scanner.scan(frames) == {30}


def test_min_gap_blocks_ticker_from_marking_every_sample():
    """
    Chân chạy đổi nội dung liên tục: không có min_gap_frames thì MỌI mẫu đều bị
    đánh dấu -> 1 keyframe mỗi stride trên toàn bộ footage tin tức.
    """
    frames = [(i, _with_text(i)) for i in range(200)]

    no_gap = TextRegionScanner(stride=10, min_gap_frames=1).scan(frames)
    gated = TextRegionScanner(stride=10, min_gap_frames=50).scan(frames)

    assert len(no_gap) >= 15
    assert len(gated) <= 4
    prev = None
    for idx in sorted(gated):
        assert prev is None or idx - prev >= 50
        prev = idx


def test_no_text_anywhere_marks_nothing():
    scanner = TextRegionScanner(stride=10, min_gap_frames=1)
    assert scanner.scan([(i, _plain()) for i in range(100)]) == set()


def test_window_boundary_does_not_mark_static_banner():
    """
    Cửa sổ 1200 frame bắt đầu giữa lúc banner đang hiển thị: mẫu đầu chỉ là mốc
    so sánh, không phải "chữ vừa xuất hiện".
    """
    scanner = TextRegionScanner(stride=10, min_gap_frames=1)
    assert scanner.scan([(i, _with_text(1)) for i in range(1200, 1300)]) == set()
