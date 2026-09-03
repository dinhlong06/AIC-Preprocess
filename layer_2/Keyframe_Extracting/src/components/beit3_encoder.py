"""
beit3_encoder.py — BEiT-3 Visual Encoder Wrapper.

Load model BEiT-3 Large từ checkpoint và trích xuất CLS token embedding
(1024-dim) cho từng frame. Embedding được L2-normalized để dùng với
cosine similarity.

Checkpoint cần:
  - beit3_large_patch16_224.pth  : model weights
  - beit3.spm                    : SentencePiece tokenizer

Lưu ý về PYTHONPATH:
  Script phải được chạy từ thư mục Keyframe_Extractor/ với:
    PYTHONPATH=unilm/beit3 python cli.py ...
  hoặc runner sẽ tự inject sys.path.
"""

from __future__ import annotations

import sys
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import cv2
from PIL import Image
from torchvision import transforms


# Kích thước ảnh BEiT-3 Large yêu cầu
_BEIT3_IMG_SIZE = 224

# Giá trị normalize theo ImageNet (chuẩn BEiT-3 dùng)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _build_transform() -> transforms.Compose:
    """
    Tạo pipeline transform ảnh: Resize → CenterCrop → ToTensor → Normalize.
    Giống cấu hình finetune của BEiT-3.
    """
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(_BEIT3_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class BEiT3Encoder:
    """
    Wrapper cho BEiT-3 Large để trích xuất visual embedding từ frame ảnh.

    Usage:
        encoder = BEiT3Encoder(
            checkpoint_path="checkpoint/beit-3/beit3_large_patch16_224.pth",
            spm_path="checkpoint/beit-3/beit3.spm",
        )
        encoder.load()
        embeddings = encoder.encode_batch(frames)   # shape (N, 1024)
        encoder.unload()

    Args:
        checkpoint_path : Đường dẫn đến file .pth (model weights).
        spm_path        : Đường dẫn đến file .spm (SentencePiece tokenizer).
        device          : "cuda" hoặc "cpu". Mặc định tự detect.
        batch_size      : Số frame encode trong mỗi lần forward.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        spm_path: str | Path,
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.spm_path = Path(spm_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._model = None
        self._transform = _build_transform()

    def load(self) -> None:
        """
        Load BEiT-3 model từ checkpoint vào memory/GPU.
        Gọi một lần trước khi encode.
        """
        # Inject beit3 source vào sys.path để import được
        beit3_src = Path(__file__).resolve().parents[2] / "unilm" / "beit3"
        if str(beit3_src) not in sys.path:
            sys.path.insert(0, str(beit3_src))

        from modeling_utils import BEiT3Wrapper, _get_large_config  # type: ignore

        # Tạo config BEiT-3 Large (img_size=224, embed_dim=1024)
        args = _get_large_config(img_size=_BEIT3_IMG_SIZE)
        args.normalize_output = True

        # Dùng BEiT3Wrapper (chỉ lấy encoder, không có downstream head)
        self._model = BEiT3Wrapper(args)
        state = torch.load(self.checkpoint_path, map_location="cpu")

        # Checkpoint có thể wrap trong 'model' key
        state_dict = state.get("model", state)
        missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[BEiT3Encoder] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[BEiT3Encoder] Unexpected keys: {len(unexpected)}")

        self._model.eval().to(self.device)
        print(f"[BEiT3Encoder] Loaded model on {self.device}")

    def unload(self) -> None:
        """Xóa model khỏi GPU/memory để giải phóng VRAM."""
        self._model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def encode_batch(
        self,
        frames: List[Tuple[int, np.ndarray]],
    ) -> Tuple[List[int], np.ndarray]:
        """
        Encode một batch frame ảnh thành CLS token embedding.

        Args:
            frames : List (frame_idx, BGR image numpy array).

        Returns:
            Tuple:
              - indices   : List frame_idx (giữ nguyên thứ tự).
              - embeddings: numpy array shape (N, 1024), L2-normalized, float32.
        """
        if self._model is None:
            raise RuntimeError("Gọi load() trước khi encode.")
        if not frames:
            return [], np.empty((0, 1024), dtype=np.float32)

        indices = [idx for idx, _ in frames]
        imgs = [img for _, img in frames]
        n = len(imgs)
        bs = self.batch_size

        def _prep_range(lo: int, hi: int) -> List[torch.Tensor]:
            return [self._preprocess(im) for im in imgs[lo:hi]]

        all_embeddings = []
        # GPU rảnh trong lúc preprocess CPU-bound và ngược lại — 1 thread nền
        # chuẩn bị batch kế tiếp trong lúc GPU forward batch hiện tại, thay vì
        # preprocess hết rồi mới forward tuần tự.
        with torch.no_grad(), ThreadPoolExecutor(max_workers=1) as prefetch:
            next_batch = prefetch.submit(_prep_range, 0, min(bs, n))
            for i in range(0, n, bs):
                tensors = next_batch.result()
                hi = min(i + bs, n)
                nxt_lo, nxt_hi = hi, min(hi + bs, n)
                if nxt_lo < n:
                    next_batch = prefetch.submit(_prep_range, nxt_lo, nxt_hi)

                batch = torch.stack(tensors).to(self.device)
                out = self._model.beit3(
                    textual_tokens=None,
                    visual_tokens=batch,
                    text_padding_position=None,
                )
                # Lấy CLS token (vị trí 0 trong encoder output)
                cls_emb = out["encoder_out"][:, 0, :]       # (B, 1024)
                cls_emb = F.normalize(cls_emb, dim=-1)      # L2 normalize
                all_embeddings.append(cls_emb.cpu().float().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        return indices, embeddings

    def _preprocess(self, bgr_img: np.ndarray) -> torch.Tensor:
        """
        Chuyển BGR numpy array → RGB PIL Image → tensor đã normalize.

        Args:
            bgr_img : Frame ảnh dạng BGR (từ OpenCV).

        Returns:
            Tensor shape (3, 224, 224).
        """
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self._transform(pil)
