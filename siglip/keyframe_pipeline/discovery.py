from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

from .exceptions import DatasetLayoutError, InvalidKeyframeInputError
from .types import DatasetIndex, Keyframe, VideoKeyframes

VIDEO_PATTERN = re.compile(r"^[A-Z]+\d+[_-]V\d+$", re.IGNORECASE)
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def natural_key(value: str) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(\d+)", value.casefold())
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)


def _validate_identifier(value: str, field: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise InvalidKeyframeInputError(f"Invalid {field}: {value!r}")


def _verify_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise InvalidKeyframeInputError(f"Unreadable image: {path}") from exc


def validate_keyframes(keyframes: Sequence[Keyframe]) -> tuple[Keyframe, ...]:
    items = tuple(keyframes)
    if not items:
        raise InvalidKeyframeInputError("Keyframe input is empty")

    video_id = items[0].video_id
    _validate_identifier(video_id, "video_id")
    seen: set[str] = set()
    for item in items:
        if item.video_id != video_id:
            raise InvalidKeyframeInputError(
                f"Mixed video IDs: expected {video_id!r}, got {item.video_id!r}"
            )
        _validate_identifier(item.keyframe_id, "keyframe_id")
        if item.keyframe_id in seen:
            raise InvalidKeyframeInputError(
                f"Duplicate keyframe_id {item.keyframe_id!r} in {video_id}"
            )
        seen.add(item.keyframe_id)
        path = Path(item.image_path)
        if not path.is_file():
            raise InvalidKeyframeInputError(f"Image does not exist: {path}")
        _verify_image(path)
    return items


def discover_video(
    video_dir: Path,
    *,
    video_id: str | None = None,
) -> VideoKeyframes:
    directory = Path(video_dir).expanduser().resolve()
    if not directory.is_dir():
        raise DatasetLayoutError(f"Video directory does not exist: {directory}")
    resolved_video_id = video_id or directory.name
    _validate_identifier(resolved_video_id, "video_id")

    image_paths = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: natural_key(path.name),
    )
    keyframes = tuple(
        Keyframe(
            video_id=resolved_video_id,
            keyframe_id=path.stem,
            image_path=path.resolve(),
        )
        for path in image_paths
    )
    return VideoKeyframes(resolved_video_id, validate_keyframes(keyframes))


def resolve_keyframes_root(dataset_root: Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise DatasetLayoutError(f"Dataset root does not exist: {root}")
    nested = root / "keyframes"
    if nested.is_dir():
        return nested.resolve()
    if any(path.is_dir() and VIDEO_PATTERN.fullmatch(path.name) for path in root.iterdir()):
        return root
    raise DatasetLayoutError(
        f"Expected {root / 'keyframes'} or direct <PREFIX>nn_Vnnn video directories"
    )


def discover_dataset(dataset_root: Path) -> DatasetIndex:
    root = resolve_keyframes_root(dataset_root)
    video_dirs = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and VIDEO_PATTERN.fullmatch(path.name)
        ),
        key=lambda path: natural_key(path.name),
    )
    if not video_dirs:
        raise DatasetLayoutError(f"No video directories found under {root}")
    videos: list[VideoKeyframes] = []
    for path in video_dirs:
        try:
            videos.append(discover_video(path))
        except InvalidKeyframeInputError:
            print(f"warning: skipping {path.name}: no keyframes found", file=sys.stderr)
    if not videos:
        raise DatasetLayoutError(f"No videos with keyframes found under {root}")
    return DatasetIndex(root=root, videos=tuple(videos))

