from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .config import SIGLIP_EMBEDDING_DIM
from .exceptions import ArtifactValidationError, OutputAlreadyExistsError
from .types import SiglipResult


def siglip_target_paths(video_id: str, output_dir: Path) -> tuple[Path, Path]:
    root = Path(output_dir).expanduser().resolve()
    return root / f"{video_id}.npy", root / f"{video_id}_ids.json"


def preflight_targets(paths: Iterable[Path], *, overwrite: bool) -> None:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise OutputAlreadyExistsError(f"Output already exists: {joined}")


def validate_siglip_result(result: SiglipResult) -> None:
    embeddings = result.embeddings
    if not isinstance(embeddings, np.ndarray):
        raise ArtifactValidationError("embeddings must be a NumPy array")
    expected = (len(result.keyframe_ids), SIGLIP_EMBEDDING_DIM)
    if embeddings.ndim != 2 or embeddings.shape != expected:
        raise ArtifactValidationError(
            f"Invalid embedding shape {embeddings.shape}; expected {expected}"
        )
    if embeddings.dtype != np.float32:
        raise ArtifactValidationError(
            f"Invalid embedding dtype {embeddings.dtype}; expected float32"
        )
    if not np.isfinite(embeddings).all():
        raise ArtifactValidationError("Embeddings contain NaN or Inf")
    if len(set(result.keyframe_ids)) != len(result.keyframe_ids):
        raise ArtifactValidationError("Duplicate keyframe IDs in SigLIP result")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ArtifactValidationError("Embedding rows are not L2-normalized")


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        writer(temp_path)
        # mkstemp tạo file mode 600 và os.replace giữ nguyên mode đó, nên artifact
        # ra 600 -> layer sau chạy bằng user khác không đọc được. Đặt lại 644.
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def save_siglip_result(
    result: SiglipResult,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> SiglipResult:
    validate_siglip_result(result)
    embedding_path, ids_path = siglip_target_paths(result.video_id, output_dir)
    preflight_targets((embedding_path, ids_path), overwrite=overwrite)

    def write_array(temp_path: Path) -> None:
        with temp_path.open("wb") as handle:
            np.save(handle, result.embeddings, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())

    def write_ids(temp_path: Path) -> None:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                list(result.keyframe_ids), handle, ensure_ascii=False, separators=(",", ":")
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_write(embedding_path, write_array)
    _atomic_write(ids_path, write_ids)
    return SiglipResult(
        video_id=result.video_id,
        embeddings=result.embeddings,
        keyframe_ids=result.keyframe_ids,
        embedding_path=embedding_path,
        ids_path=ids_path,
    )


def load_siglip_result(
    embedding_path: Path,
    ids_path: Path,
    *,
    mmap_mode: str | None = None,
) -> SiglipResult:
    embedding_path = Path(embedding_path).expanduser().resolve()
    ids_path = Path(ids_path).expanduser().resolve()
    try:
        embeddings = np.load(
            embedding_path, mmap_mode=mmap_mode, allow_pickle=False
        )
    except Exception as exc:
        raise ArtifactValidationError(f"Cannot load {embedding_path}") from exc
    try:
        with ids_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise ArtifactValidationError(f"Cannot load {ids_path}") from exc

    if not isinstance(payload, list) or not all(
        isinstance(value, str) for value in payload
    ):
        raise ArtifactValidationError("Invalid SigLIP IDs JSON schema")
    video_id = embedding_path.stem
    if ids_path.stem != f"{video_id}_ids":
        raise ArtifactValidationError(
            f"Artifact filenames do not pair up: {embedding_path.name} / {ids_path.name}"
        )

    result = SiglipResult(
        video_id=video_id,
        embeddings=embeddings,
        keyframe_ids=tuple(payload),
        embedding_path=embedding_path,
        ids_path=ids_path,
    )
    validate_siglip_result(result)
    return result
