"""DeepSolo (detection) + PARSeq-VN (recognition) stage-1 engine.

engine.run(path_or_bgr) -> list_of_line_dicts.

Only runnable inside the ocr-deepsolo-parseq image (detectron2/DeepSolo/
strhub on PYTHONPATH) -- launch via ./run.sh, which mounts
experiments/deepsolo_parseq at /exp (weights + vn_scenetext live there).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from adet.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint

EXP = Path(os.environ.get("DEEPSOLO_ROOT", "/exp"))


def _crop_word(img, bd, pad_y, pad_x):
    # DeepSolo was trained on Latin: its box hugs base letters, so Vietnamese
    # tone marks above/below need extra padding or they get clipped.
    pts = bd.reshape(-1, 2)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    h = y1 - y0
    x0, x1 = int(max(0, x0 - pad_x * h)), int(min(img.shape[1], x1 + pad_x * h))
    y0, y1 = int(max(0, y0 - pad_y * h)), int(min(img.shape[0], y1 + pad_y * h))
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def _dedupe(boxes, scores, max_overlap=0.5):
    # A low det threshold makes several queries fire on the same word -- keep
    # the best-scoring box when another box lies mostly inside a kept one.
    kept = []
    for i in np.argsort(-scores):
        x0, y0, x1, y1 = boxes[i]
        area = (x1 - x0) * (y1 - y0)
        if all(
            max(0, min(x1, k[2]) - max(x0, k[0])) * max(0, min(y1, k[3]) - max(y0, k[1]))
            <= max_overlap * min(area, (k[2] - k[0]) * (k[3] - k[1]))
            for k in (boxes[j] for j in kept)
        ):
            kept.append(i)
    return kept


def _group_lines(words):
    lines = []
    for w in sorted(words, key=lambda w: (w["box"][1] + w["box"][3]) / 2):
        cy, h = (w["box"][1] + w["box"][3]) / 2, w["box"][3] - w["box"][1]
        line = next((l for l in lines if abs(l["cy"] - cy) < 0.5 * min(l["h"], h)), None)
        if line is None:
            lines.append({"cy": cy, "h": h, "words": [w]})
        else:
            line["words"].append(w)
    out = []
    for l in lines:
        ws = sorted(l["words"], key=lambda w: w["box"][0])
        # One ticker row carries several headlines split by a logo -- break on wide gaps.
        segs = [[ws[0]]]
        for prev, w in zip(ws, ws[1:]):
            if w["box"][0] - prev["box"][2] > 1.5 * l["h"]:
                segs.append([])
            segs[-1].append(w)
        for seg in segs:
            boxes = [w["box"] for w in seg]
            out.append({
                "text": " ".join(w["text"] for w in seg),
                "confidence": round(float(np.mean([w["conf"] for w in seg])), 4),
                "box": [
                    min(b[0] for b in boxes),
                    min(b[1] for b in boxes),
                    max(b[2] for b in boxes),
                    max(b[3] for b in boxes),
                ],
            })
    return out


class DeepSoloParseqEngine:
    """DeepSolo detector + PARSeq-VN recognizer."""

    def __init__(
        self,
        det_threshold: float = 0.15,
        min_size: int = 1080,
        pad_y: float = 0.25,
        pad_x: float = 0.1,
    ) -> None:
        cfg = get_cfg()
        # config lấy trong image (DeepSolo được clone lúc build, sha khớp bản ở /exp),
        # còn weight thì mount từ /exp qua DEEPSOLO_ROOT
        cfg.merge_from_file("/workspace/DeepSolo/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml")
        cfg.MODEL.WEIGHTS = str(EXP / "weights/ic15_res50_finetune_synth-tt-mlt-13-15-textocr.pth")
        cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = det_threshold
        cfg.INPUT.MIN_SIZE_TEST = min_size
        self._det = DefaultPredictor(cfg)
        self._parseq = load_from_checkpoint(str(EXP / "vn_scenetext/weights/rec/best-parseq.ckpt")).eval().cuda()
        self._tf = SceneTextDataModule.get_transform(self._parseq.hparams.img_size)
        self._pad_y, self._pad_x = pad_y, pad_x
        self.det_ms = 0.0
        self.rec_ms = 0.0

    @torch.inference_mode()
    def run(self, source: "str | object") -> list[dict]:
        if isinstance(source, str):
            img = cv2.imread(source)
            if img is None:
                return []
        else:
            img = source

        t = time.time()
        inst = self._det(img)["instances"].to("cpu")
        self.det_ms += (time.time() - t) * 1000

        crops = [_crop_word(img, bd.numpy(), self._pad_y, self._pad_x) for bd in inst.bd]
        if crops:
            crops = [crops[i] for i in _dedupe(np.array([b for _, b in crops]).reshape(-1, 4), inst.scores.numpy())]
        crops = [(c, b) for c, b in crops if c.size]
        if not crops:
            return []

        t = time.time()
        batch = torch.stack([self._tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c, _ in crops]).cuda()
        preds, probs = self._parseq.tokenizer.decode(self._parseq(batch).softmax(-1))
        self.rec_ms += (time.time() - t) * 1000

        words = [
            {"text": p, "conf": pr.mean().item(), "box": b}
            for p, pr, (_, b) in zip(preds, probs, crops)
            if p
        ]
        return _group_lines(words)
