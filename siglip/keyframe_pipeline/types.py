from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Keyframe:
    video_id: str
    keyframe_id: str
    image_path: Path


@dataclass(frozen=True, slots=True)
class VideoKeyframes:
    video_id: str
    keyframes: tuple[Keyframe, ...]


@dataclass(frozen=True, slots=True)
class DatasetIndex:
    root: Path
    videos: tuple[VideoKeyframes, ...]


@dataclass(frozen=True, slots=True)
class SiglipResult:
    video_id: str
    embeddings: np.ndarray
    keyframe_ids: tuple[str, ...]
    embedding_path: Path | None = None
    ids_path: Path | None = None


@dataclass(frozen=True, slots=True)
class SiglipDatasetResult:
    root: Path
    results: tuple[SiglipResult, ...]
