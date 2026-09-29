from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..config import DEFAULT_SIGLIP_MODEL_ID, DEFAULT_SIGLIP_NUM_WORKERS, SIGLIP_EMBEDDING_DIM
from ..exceptions import ModelConfigurationError


class _ImageDataset:
    """Decode + preprocess one image, so DataLoader workers do it in parallel."""

    def __init__(self, paths: Sequence[Path], processor: Any) -> None:
        self.paths = paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Any:
        with Image.open(self.paths[index]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


class TransformersSiglipBackend:
    """Lazy-loading Hugging Face SigLIP image encoder."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_SIGLIP_MODEL_ID,
        device: str | None = None,
        num_workers: int = DEFAULT_SIGLIP_NUM_WORKERS,
    ) -> None:
        self.model_id = model_id
        self.requested_device = device
        self.num_workers = num_workers
        self.device: str | None = None
        self._torch: Any = None
        self._processor: Any = None
        self._model: Any = None

    @property
    def embedding_dim(self) -> int:
        return SIGLIP_EMBEDDING_DIM

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except Exception as exc:
            raise ModelConfigurationError(
                "SigLIP requires torch and transformers"
            ) from exc

        device = self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelConfigurationError(
                f"CUDA was requested ({device}) but CUDA PyTorch is unavailable"
            )
        try:
            # This backend only consumes images. Loading AutoProcessor also
            # initializes SigLIP's tokenizer and unnecessarily requires
            # sentencepiece for an image-only inference path.
            processor = AutoImageProcessor.from_pretrained(self.model_id)
            load_dtype = torch.float16 if device.startswith("cuda") else torch.float32
            try:
                model = AutoModel.from_pretrained(self.model_id, dtype=load_dtype)
            except TypeError:
                model = AutoModel.from_pretrained(
                    self.model_id, torch_dtype=load_dtype
                )
            vision_size = int(model.config.vision_config.hidden_size)
            if vision_size != SIGLIP_EMBEDDING_DIM:
                raise ModelConfigurationError(
                    f"Model {self.model_id!r} has vision hidden size {vision_size}; "
                    f"expected {SIGLIP_EMBEDDING_DIM}"
                )
            model = model.to(device).eval()
        except ModelConfigurationError:
            raise
        except Exception as exc:
            raise ModelConfigurationError(
                f"Cannot load SigLIP model {self.model_id!r}"
            ) from exc

        self._torch = torch
        self._processor = processor
        self._model = model
        self.device = device

    def encode_paths(self, paths: Sequence[Path], batch_size: int) -> np.ndarray:
        self._ensure_loaded()
        from torch.utils.data import DataLoader

        loader = DataLoader(
            _ImageDataset(list(paths), self._processor),
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.device.startswith("cuda"),
        )
        chunks = []
        with self._torch.inference_mode():
            for pixel_values in loader:
                features = self._model.get_image_features(
                    pixel_values=pixel_values.to(self.device, self._model.dtype)
                )
                chunks.append(features.float().cpu().numpy())
        return np.concatenate(chunks).astype(np.float32)
