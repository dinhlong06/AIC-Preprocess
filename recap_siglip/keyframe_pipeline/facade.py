from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .backends import TransformersSiglipBackend
from .config import DEFAULT_SIGLIP_MODEL_ID
from .io import preflight_targets, siglip_target_paths
from .protocols import SiglipBackend
from .siglip import extract_siglip
from .types import Keyframe, SiglipDatasetResult, SiglipResult


class KeyframePipeline:
    def __init__(
        self,
        *,
        siglip_backend: SiglipBackend | None = None,
        siglip_model_id: str = DEFAULT_SIGLIP_MODEL_ID,
        siglip_device: str | None = None,
    ) -> None:
        self.siglip_backend = siglip_backend or TransformersSiglipBackend(
            model_id=siglip_model_id, device=siglip_device
        )

    def run_siglip(
        self,
        keyframes: Sequence[Keyframe],
        *,
        output_dir: Path | None = None,
        batch_size: int = 32,
        overwrite: bool = False,
    ) -> SiglipResult:
        return extract_siglip(
            keyframes,
            backend=self.siglip_backend,
            output_dir=output_dir,
            batch_size=batch_size,
            overwrite=overwrite,
        )

    def run_siglip_dataset(
        self,
        dataset_root: Path,
        *,
        output_root: Path,
        siglip_batch_size: int = 32,
        overwrite: bool = False,
    ) -> SiglipDatasetResult:
        from .discovery import discover_dataset

        index = discover_dataset(dataset_root)
        siglip_dir = Path(output_root).expanduser().resolve() / "siglip"
        preflight_targets(
            [
                path
                for video in index.videos
                for path in siglip_target_paths(video.video_id, siglip_dir)
            ],
            overwrite=overwrite,
        )
        return SiglipDatasetResult(
            index.root,
            tuple(
                self.run_siglip(
                    video.keyframes,
                    output_dir=siglip_dir,
                    batch_size=siglip_batch_size,
                    overwrite=overwrite,
                )
                for video in index.videos
            ),
        )
