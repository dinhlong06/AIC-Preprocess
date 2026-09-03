import numpy as np

from src.components.blank_veto import BlankTransitionVetoFilter


def _solid(value: int) -> np.ndarray:
    return np.full((64, 64, 3), value, dtype=np.uint8)


def _textured() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


def test_drops_pure_black_frame():
    veto = BlankTransitionVetoFilter()
    frames = [(0, _solid(0))]
    assert veto.filter(frames) == []


def test_drops_pure_white_flash_frame():
    veto = BlankTransitionVetoFilter()
    frames = [(0, _solid(255))]
    assert veto.filter(frames) == []


def test_keeps_dark_frame_with_content():
    veto = BlankTransitionVetoFilter()
    img = _solid(10)
    img[10:54, 10:54] = 200  # cạnh sáng giữa nền tối -> variance cao
    frames = [(0, img)]
    assert veto.filter(frames) == frames


def test_keeps_normal_textured_frame():
    veto = BlankTransitionVetoFilter()
    frames = [(0, _textured())]
    assert veto.filter(frames) == frames
