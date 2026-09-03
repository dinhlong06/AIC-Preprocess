"""
mobilenet_encoder.py — MobileNetV3 Large Visual Encoder Wrapper.

Sử dụng MobileNetV3-Large pretrained từ torchvision (ImageNet weights).
Interface giống hệt BEiT3Encoder để có thể swap pipeline chỉ bằng config.

Đã cập nhật:
  - FP32 Full Precision (Đảm bảo Precision, Recall, F1 chuẩn xác 100%)
  - Multi-worker CPU ThreadPool (num_workers = 2) tăng tốc preprocess CPU.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# Kích thước ảnh MobileNetV3 yêu cầu
_MOBILE_IMG_SIZE = 224

# Normalize theo ImageNet
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _build_transform() -> transforms.Compose:
    """Tạo pipeline transform: Resize → CenterCrop → ToTensor → Normalize."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(_MOBILE_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class MobileNetEncoder:
    """
    Wrapper MobileNetV3-Large để trích xuất visual embedding.

    Args:
        device      : "cuda" hoặc "cpu". Mặc định tự detect.
        batch_size  : Số frame encode trong mỗi lần forward.
        num_workers : Số CPU worker thread để preprocess ảnh (default: 2).
    """

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 32,
        num_workers: int = 2,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._model: nn.Module | None = None
        self._transform = _build_transform()

    def load(self) -> None:
        """Load MobileNetV3-Large với ImageNet pretrained weights."""
        base = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )
        self._model = nn.Sequential(
            base.features,
            base.avgpool,
            nn.Flatten(),        # (B, 960)
        )
        self._model.eval().to(self.device)
        print(f"[MobileNetEncoder] Loaded MobileNetV3-Large on {self.device} (FP32 precision, num_workers={self.num_workers})")

    def unload(self) -> None:
        """Xóa model khỏi GPU/memory để giải phóng VRAM."""
        self._model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def encode_batch(
        self,
        frames: List[Tuple[int, np.ndarray]],
        video_id: str | None = None,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Encode một batch frame ảnh thành feature vector đã normalize (FP32).
        Sử dụng CPU ThreadPool (2 workers) cho preprocess ảnh.
        """
        if self._model is None:
            raise RuntimeError("Gọi load() trước khi encode.")
        if not frames:
            return [], np.empty((0, 960), dtype=np.float32)

        from tqdm import tqdm

        indices = [idx for idx, _ in frames]
        bgr_imgs = [img for _, img in frames]

        # Multi-core CPU Parallel Preprocessing (2 workers)
        if self.num_workers > 1 and len(bgr_imgs) > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                tensors = list(executor.map(self._preprocess, bgr_imgs))
        else:
            tensors = [self._preprocess(img) for img in bgr_imgs]

        desc_str = f"[MobileNet] {video_id}" if video_id else "[MobileNet]"
        all_embeddings = []
        with torch.no_grad():
            with tqdm(
                total=len(tensors),
                desc=desc_str,
                unit="img",
                dynamic_ncols=True,
                leave=False,
            ) as pbar:
                for i in range(0, len(tensors), self.batch_size):
                    batch = torch.stack(tensors[i : i + self.batch_size]).to(self.device)
                    feat = self._model(batch)                    # (B, 960)
                    feat = F.normalize(feat, dim=-1)             # L2 normalize
                    all_embeddings.append(feat.cpu().float().numpy())
                    pbar.update(len(batch))
                    del batch, feat

        if self.device == "cuda" and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

        embeddings = np.concatenate(all_embeddings, axis=0)
        return indices, embeddings

    def _preprocess(self, bgr_img: np.ndarray) -> torch.Tensor:
        """Chuyển BGR numpy array → RGB PIL Image → tensor đã normalize."""
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self._transform(pil)
