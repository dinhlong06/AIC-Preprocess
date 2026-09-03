"""CPU motion filters used by the additional Layer 2 pipelines."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

FrameTuple = Tuple[int, np.ndarray]


class BaseMotionFilter:
    def __init__(self, min_frame_distance: int = 5, motion_threshold: float = 0.0):
        self.min_frame_distance = max(1, int(min_frame_distance))
        self.motion_threshold = float(motion_threshold)

    def select_candidates(self, frames: List[FrameTuple]) -> List[int]:
        raise NotImplementedError

    def _non_max_suppression(self, scores: List[Tuple[int, float]]) -> List[int]:
        valid = [(index, score) for index, score in scores if score > self.motion_threshold]
        valid.sort(key=lambda item: item[1], reverse=True)
        selected: List[int] = []
        for index, _ in valid:
            if all(abs(index - kept) >= self.min_frame_distance for kept in selected):
                selected.append(index)
        return sorted(selected)


class OpticalFlowFilter(BaseMotionFilter):
    """Dense Farneback optical-flow candidate selector (accurate, slower)."""

    def __init__(self, min_frame_distance: int = 5, motion_threshold: float = 1.0):
        super().__init__(min_frame_distance, motion_threshold)

    def select_candidates(self, frames: List[FrameTuple]) -> List[int]:
        if len(frames) <= 1:
            return [frames[0][0]] if frames else []

        previous_index, previous_frame = frames[0]
        previous_gray = cv2.cvtColor(
            cv2.resize(previous_frame, (256, 256)), cv2.COLOR_BGR2GRAY
        )
        scores = [(previous_index, 0.0)]
        for index, frame in frames[1:]:
            current_gray = cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            scores.append((index, float(np.mean(magnitude))))
            previous_gray = current_gray
        return self._non_max_suppression(scores) or [frames[0][0]]


class FrameDiffFilter(BaseMotionFilter):
    """Fast pixel-difference candidate selector."""

    def __init__(self, min_frame_distance: int = 5, motion_threshold: float = 5.0):
        super().__init__(min_frame_distance, motion_threshold)

    @staticmethod
    def _prepare(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(cv2.resize(frame, (144, 144)), cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def select_candidates(self, frames: List[FrameTuple]) -> List[int]:
        if len(frames) <= 1:
            return [frames[0][0]] if frames else []

        previous_index, previous_frame = frames[0]
        previous_gray = self._prepare(previous_frame)
        scores = [(previous_index, 0.0)]
        for index, frame in frames[1:]:
            current_gray = self._prepare(frame)
            scores.append((index, float(np.mean(cv2.absdiff(current_gray, previous_gray)))))
            previous_gray = current_gray
        return self._non_max_suppression(scores) or [frames[0][0]]
