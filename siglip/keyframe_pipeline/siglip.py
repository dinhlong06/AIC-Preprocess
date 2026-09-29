from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .backends import TransformersSiglipBackend
from .config import (
    DEFAULT_SIGLIP_MODEL_ID,
    DEFAULT_SIGLIP_NUM_WORKERS,
    SIGLIP_EMBEDDING_DIM,
)
from .discovery import discover_dataset, discover_video, validate_keyframes
from .exceptions import InvalidArgumentError, ModelInferenceError
from .io import (
    load_siglip_result,
    preflight_targets,
    save_siglip_result,
    siglip_target_paths,
    validate_siglip_result,
)
from .protocols import SiglipBackend
from .types import Keyframe, SiglipDatasetResult, SiglipResult


def _resolve_backend(
    backend: SiglipBackend | None,
    *,
    model_id: str,
    device: str | None,
    num_workers: int,
) -> SiglipBackend:
    if backend is not None:
        if (
            model_id != DEFAULT_SIGLIP_MODEL_ID
            or device is not None
            or num_workers != DEFAULT_SIGLIP_NUM_WORKERS
        ):
            raise InvalidArgumentError(
                "model_id/device/num_workers cannot override an injected SigLIP backend"
            )
        return backend
    return TransformersSiglipBackend(
        model_id=model_id, device=device, num_workers=num_workers
    )


def extract_siglip(
    keyframes: Sequence[Keyframe],
    *,
    backend: SiglipBackend | None = None,
    model_id: str = DEFAULT_SIGLIP_MODEL_ID,
    output_dir: Path | None = None,
    batch_size: int = 32,
    device: str | None = None,
    num_workers: int = DEFAULT_SIGLIP_NUM_WORKERS,
    overwrite: bool = False,
) -> SiglipResult:
    if batch_size <= 0:
        raise InvalidArgumentError("batch_size must be positive")
    items = validate_keyframes(keyframes)
    video_id = items[0].video_id
    if output_dir is not None:
        preflight_targets(
            siglip_target_paths(video_id, output_dir), overwrite=overwrite
        )
    resolved_backend = _resolve_backend(
        backend, model_id=model_id, device=device, num_workers=num_workers
    )
    if resolved_backend.embedding_dim != SIGLIP_EMBEDDING_DIM:
        raise ModelInferenceError(
            f"SigLIP backend dimension {resolved_backend.embedding_dim}; "
            f"expected {SIGLIP_EMBEDDING_DIM}"
        )

    raw = np.asarray(
        resolved_backend.encode_paths([item.image_path for item in items], batch_size)
    )
    expected_shape = (len(items), SIGLIP_EMBEDDING_DIM)
    if raw.ndim != 2 or raw.shape != expected_shape:
        raise ModelInferenceError(
            f"SigLIP returned shape {raw.shape} for {video_id}; "
            f"expected {expected_shape}"
        )
    array = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ModelInferenceError(f"SigLIP returned NaN/Inf for {video_id}")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0):
        raise ModelInferenceError(f"SigLIP returned a zero vector for {video_id}")

    result = SiglipResult(
        video_id=video_id,
        embeddings=np.ascontiguousarray(array / norms[:, None], dtype=np.float32),
        keyframe_ids=tuple(item.keyframe_id for item in items),
    )
    validate_siglip_result(result)
    if output_dir is not None:
        return save_siglip_result(result, output_dir, overwrite=overwrite)
    return result


def extract_siglip_from_dir(
    video_dir: Path,
    *,
    video_id: str | None = None,
    backend: SiglipBackend | None = None,
    model_id: str = DEFAULT_SIGLIP_MODEL_ID,
    output_dir: Path | None = None,
    batch_size: int = 32,
    device: str | None = None,
    num_workers: int = DEFAULT_SIGLIP_NUM_WORKERS,
    overwrite: bool = False,
) -> SiglipResult:
    video = discover_video(video_dir, video_id=video_id)
    return extract_siglip(
        video.keyframes,
        backend=backend,
        model_id=model_id,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        overwrite=overwrite,
    )


def extract_siglip_dataset(
    dataset_root: Path,
    *,
    output_dir: Path,
    backend: SiglipBackend | None = None,
    model_id: str = DEFAULT_SIGLIP_MODEL_ID,
    batch_size: int = 32,
    device: str | None = None,
    num_workers: int = DEFAULT_SIGLIP_NUM_WORKERS,
    overwrite: bool = False,
) -> SiglipDatasetResult:
    if batch_size <= 0:
        raise InvalidArgumentError("batch_size must be positive")
    index = discover_dataset(dataset_root)
    # Resume: video đã có .npy thì bỏ qua, để lần chạy đứt giữa chừng chạy tiếp được.
    videos = [
        video
        for video in index.videos
        if overwrite or not siglip_target_paths(video.video_id, output_dir)[0].exists()
    ]
    preflight_targets(
        [path for video in videos for path in siglip_target_paths(video.video_id, output_dir)],
        overwrite=overwrite,
    )
    resolved_backend = _resolve_backend(
        backend, model_id=model_id, device=device, num_workers=num_workers
    )
    results = tuple(
        extract_siglip(
            video.keyframes,
            backend=resolved_backend,
            output_dir=output_dir,
            batch_size=batch_size,
            overwrite=overwrite,
        )
        for video in videos
    )
    return SiglipDatasetResult(root=index.root, results=results)


__all__ = [
    "extract_siglip",
    "extract_siglip_from_dir",
    "extract_siglip_dataset",
    "load_siglip_result",
]
