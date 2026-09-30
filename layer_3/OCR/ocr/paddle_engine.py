"""
paddle_engine.py -- PaddleOCR detection + two INDEPENDENT recognizers
(VietOCR + Paddle's own), no cross-recognizer merge logic here.

PaddleOCR's own recognizer (any version: v3/v5/v6, any `lang`) cannot emit 70
of the Vietnamese accented characters (U+1EA0-U+1EF9 block) -- see memory
paddleocr-vietnamese-charset-gap -- so text with those marks comes out with
the letter deleted entirely, not just mis-accented. This is kept as-is in
out_paddle (no ascii-stripping workaround) -- callers relying on Paddle's own
recognizer for Vietnamese-accented text should expect that gap. VietOCR's
vgg_seq2seq has a real Vietnamese vocabulary and doesn't have this gap
(measured live on 168 real crops: 54.8% word-drop rate for PaddleOCR's own
recognizer vs correctly restoring nearly all of the same lines with VietOCR,
e.g. "Thy din Tuyen Quang" -> "Thủy điện Tuyên Quang" -- exact). It also
doesn't hallucinate Vietnamese marks onto English text (measured: "WELCOME"/
"BREAKING NEWS"/"SUBSCRIBE" all passed through unchanged).

So detection stays PaddleOCR, recognition runs BOTH VietOCR and Paddle's own
recognizer on the detector's crops. Both recognizers' text is trusted
unconditionally (no confidence filter -- every non-empty box is kept). This
mirrors the original (retired) Paddle+VietOCR pipeline's detector reuse.
VietOCR's answer is kept per box: it measured better on every case tried.

Detection uses the full `PaddleOCR()` combined pipeline (not the standalone
`TextDetection` class) -- its own `rec_texts`/`rec_scores` are kept, not
discarded. Measured live: the standalone detector's raw polygons are NOT the
same boxes the combined pipeline uses (one frame: standalone gave a 322x166px
box spanning two unrelated lines merged together, and dropped the HTV7HD logo
entirely at the same confidence threshold that keeps it in the combined
pipeline) -- the combined pipeline applies text-line-level postprocessing the
bare detector doesn't. Reusing its boxes keeps that working detection
behavior.
"""

from __future__ import annotations

from PIL import Image


# Padding around each detected polygon before cropping for recognition --
# same margin used in the ad-hoc VietOCR re-test (experiments/vietocr_test),
# avoids clipping tall/italic glyphs right at the box edge.
CROP_PAD = 3


class PaddleEngine:
    """PaddleOCR detector + two independent recognizers (VietOCR, Paddle's own)."""

    def __init__(
        self,
        lang: str = "vi",  # unused, kept for config.yaml backward compat
        ocr_version: str | None = None,
        unclip_ratio: float | None = None,
    ) -> None:
        from paddleocr import PaddleOCR
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        det_kwargs = {"ocr_version": ocr_version} if ocr_version else {}
        if unclip_ratio is not None:
            det_kwargs["text_det_unclip_ratio"] = unclip_ratio
        self._det = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,  # CPU oneDNN path hits a PIR-attribute NotImplementedError on this build
            **det_kwargs,
        )

        cfg = Cfg.load_config_from_name("vgg_seq2seq")
        cfg["device"] = "cuda:0"
        cfg["predictor"]["beamsearch"] = False  # 3.3fps vs 0.9fps for beamsearch, no accuracy win -- see memory
        self._rec = Predictor(cfg)

    def run(self, source: "str | object") -> tuple[list[dict], list[dict]]:
        """
        Run detection + two independent recognizers on a file path or an
        in-memory BGR array (e.g. from frame_skip.preprocess).

        Returns
        -------
        tuple[list[dict], list[dict]]
            (vietocr_out, paddle_origin_out) -- both
            [{"text": str, "confidence": float, "box": [x1,y1,x2,y2]}, ...],
            each keeping every non-empty box regardless of confidence (both
            recognizers trusted unconditionally). A box can end up in one
            list, both, or neither, if one recognizer returns empty text
            where the other doesn't.
        """
        if isinstance(source, str):
            img = Image.open(source).convert("RGB")
        else:
            import numpy as np  # BGR (OpenCV convention) -> RGB
            img = Image.fromarray(np.asarray(source)[:, :, ::-1])

        det_result = self._det.predict(source)
        if not det_result:
            return [], []
        boxes = det_result[0].get("rec_boxes", [])
        paddle_texts = det_result[0].get("rec_texts", [])
        paddle_scores = det_result[0].get("rec_scores", [])

        w, h = img.size
        crops, coords, p_texts, p_scores = [], [], [], []
        for box, p_text, p_score in zip(boxes, paddle_texts, paddle_scores):
            bx1, by1, bx2, by2 = [int(v) for v in box]
            x1, y1 = max(0, bx1 - CROP_PAD), max(0, by1 - CROP_PAD)
            x2, y2 = min(w, bx2 + CROP_PAD), min(h, by2 + CROP_PAD)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
            coords.append((x1, y1, x2, y2))
            p_texts.append(p_text.strip())
            p_scores.append(float(p_score))

        if not crops:
            return [], []

        # One batched forward pass for all boxes in the frame instead of one
        # GPU call per box -- predict_batch buckets by resized width so
        # same-width crops share a single torch.cat'd forward pass.
        texts, probs = self._rec.predict_batch(crops, return_prob=True)

        out, out_paddle = [], []
        for (x1, y1, x2, y2), text, prob, p_text, p_score in zip(coords, texts, probs, p_texts, p_scores):
            text, prob = text.strip(), float(prob)
            if text:
                out.append({
                    "text": text,
                    "confidence": round(prob, 4),
                    "box": [x1, y1, x2, y2],
                })

            if p_text:
                out_paddle.append({
                    "text": p_text,
                    "confidence": round(p_score, 4),
                    "box": [x1, y1, x2, y2],
                })
        return out, out_paddle
