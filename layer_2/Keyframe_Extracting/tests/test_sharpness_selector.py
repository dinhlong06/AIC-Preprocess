import cv2
import numpy as np

from src.components.sharpness_selector import SharpnessReselector


def _sharp_scene() -> np.ndarray:
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (54, 54), (255, 255, 255), thickness=2)
    cv2.rectangle(img, (20, 20), (44, 44), (0, 0, 0), thickness=2)
    return img


def _blurred(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)


def _different_scene() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


def test_swaps_to_sharper_neighbor_same_scene():
    sharp = _sharp_scene()
    blurry = _blurred(sharp)
    candidate_map = {
        8: blurry,   # keyframe gốc, mờ
        9: sharp,    # lân cận, nét hơn, cùng cảnh
    }
    reselector = SharpnessReselector(window_radius=4)
    images = reselector.reselect([8], candidate_map, shot_start=0, shot_end=20)
    assert images[0] is sharp


def test_does_not_swap_across_different_scene():
    sharp = _sharp_scene()
    blurry = _blurred(sharp)
    candidate_map = {
        8: blurry,
        9: _different_scene(),  # nét hơn nhưng khác cảnh hoàn toàn
    }
    reselector = SharpnessReselector(window_radius=4)
    images = reselector.reselect([8], candidate_map, shot_start=0, shot_end=20)
    assert images[0] is blurry


def test_does_not_cross_shot_boundary():
    sharp = _sharp_scene()
    blurry = _blurred(sharp)
    candidate_map = {
        10: blurry,
        9: sharp,  # nét hơn nhưng nằm trước shot_start=10
    }
    reselector = SharpnessReselector(window_radius=4)
    images = reselector.reselect([10], candidate_map, shot_start=10, shot_end=20)
    assert images[0] is blurry
