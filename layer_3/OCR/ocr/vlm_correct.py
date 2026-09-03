"""
vlm_correct.py -- targeted correction for boxes where the final merged text
(merge_recognizers.py output) is still a space-less blob: the residual case
neither recognizer split correctly, so there's nothing left to pick between.
Measured on batch1: 4,689/922,351 boxes (0.55%).

Uses Qwen3-VL-4B (huypl53/qwen3-vl-4b-vietnamese-ocr-merged, weights in
../OCR_v2/models/) via transformers+bitsandbytes NF4 -- vLLM 0.28.0 dropped
bitsandbytes quantization entirely (upstream regression, confirmed live),
so this runs on plain transformers instead. Only the flagged ~0.5% of boxes
need this, not the whole dataset, so the extra latency vs vLLM doesn't
matter here the way it would for a full-frame OCR pass.

Crops just the box (small image, ~0.4-0.5s/box batched) instead of feeding
the whole frame -- measured live: whole-frame inference was 12-46x slower
per item and OOM'd when batched on an 11GB shared-cluster GPU, plus it has
no natural way to match a regenerated multi-box answer back to one box.

Decoding is plain greedy (do_sample=False, no repetition_penalty or
no_repeat_ngram_size). Both were tried and made things worse: on a 60-box
sample, repetition_penalty=1.3 fixed 2 known loop cases but introduced
fabricated notes on 18/60 boxes (chasing repetition away pushes the model
into inventing new content instead of stopping); no_repeat_ngram_size=3
fixed the same 2 cases but corrupted 14/60 previously-correct pass-through
answers by blocking legitimate verbatim copies (brand names, URLs that
happen to repeat a substring). The model's own loop failure (~3% of boxes,
confirmed via crop inspection: happens on blurry logos / very small rotated
text) is instead caught AFTER generation by dedupe_consecutive(), which
keeps whatever the model read correctly before it started looping -- only
an empty result after deduping falls back to the original merged text.

Hallucination on genuinely unreadable crops (a separate failure mode from
looping) was measured to be much rarer than it first looked: of 8 boxes
whose corrected text looked suspicious by a raw-vs-corrected similarity
heuristic, manual inspection against the actual crop showed 6-7 were
correct or close reads of real (if blurry) content -- the similarity
heuristic can't tell "changed a lot because it's a legitimate fix" apart
from "changed a lot because it's invented", so it was dropped rather than
used as a filter. Only ~1-2/60 were genuine fabrication on clearly-legible
text. Given the residual box count is already tiny (0.55%), this risk is
accepted rather than guarded against (self-consistency resampling would
roughly 3x the runtime for the whole 4,689-box subset for a benefit this
narrow).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

CROP_PAD = 5

PROMPT_TMPL = (
    "Day la 1 doan text OCR bi loi tu anh nay: \"{raw}\". "
    "Van ban co the bi dinh lien tu (thieu khoang trang) hoac thieu dau tieng Viet. "
    "Dua vao anh, hay sua lai cho dung: tach tu va them dau neu day la tu/cau tieng Viet. "
    "Neu day la ten rieng, thuong hieu, URL, tu tieng Anh, ma nguon, hoac khong doc ro duoc trong anh: "
    "GIU NGUYEN y het, khong doi. Chi tra ve DUY NHAT ket qua da sua, khong giai thich, "
    "KHONG duoc them ghi chu, chu thich, hay bat ky chu nao ngoai ket qua. "
    "Neu khong doc ro duoc gi trong anh: tra ve DUNG Y HET chuoi goc, khong them bot gi."
)

GEN_KWARGS = dict(max_new_tokens=64, do_sample=False)


def looks_merged_blob(text: str) -> bool:
    """Same shape check used throughout the residual-merged-box measurement:
    a long space-less run of mostly letters. Deliberately simple/lenient --
    false positives just mean an already-fine box gets a second (cheap,
    grounded) opinion, false negatives mean it's silently left as-is."""
    return len(text) >= 12 and " " not in text and sum(c.isalpha() for c in text) >= 10


def looks_degenerate(text: str) -> bool:
    """Any non-trivial line (ignoring ``` fence markers) that appears more
    than once means the model looped or duplicated its own answer -- a
    correct short correction never needs to repeat a full line."""
    lines = [l.strip() for l in text.split("\n") if l.strip() and not l.strip().startswith("```")]
    seen = set()
    for l in lines:
        if l in seen:
            return True
        seen.add(l)
    return False


def dedupe_consecutive(text: str) -> str:
    """The part the model generates BEFORE it gets stuck looping is usually
    still a correct reading (e.g. 'SIU UNIVERSITY' before 'SIU\\nSIU\\n...'
    x19) -- drop repeat runs and code-fence markers instead of discarding
    the whole answer."""
    out, prev = [], None
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            continue
        if s == prev:
            continue
        out.append(line)
        prev = s
    return "\n".join(out).strip()


class VLMCorrector:
    """Loads Qwen3-VL-4B once; call .correct(items) for one batched pass."""

    def __init__(self, model_dir: str) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self._processor = AutoProcessor.from_pretrained(model_dir)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_dir, quantization_config=bnb, torch_dtype=torch.float16, device_map="cuda:0"
        ).eval()

    def correct(self, crops: list[Image.Image], raw_texts: list[str]) -> list[str]:
        """One batched forward pass over all given (crop, raw_text) pairs.
        Returns the corrected text per item, already deduped/loop-guarded."""
        import torch

        msgs = [[{"role": "user", "content": [
            {"type": "image", "image": crop},
            {"type": "text", "text": PROMPT_TMPL.format(raw=raw)},
        ]}] for crop, raw in zip(crops, raw_texts)]
        texts = [self._processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        images = [[c["image"] for c in m[0]["content"] if c["type"] == "image"] for m in msgs]
        enc = self._processor(text=texts, images=images, return_tensors="pt", padding=True).to("cuda:0")

        with torch.no_grad():
            gen = self._model.generate(**enc, **GEN_KWARGS)
        outs = self._processor.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

        final = []
        for raw, out in zip(raw_texts, outs):
            out = out.strip()
            if looks_degenerate(out):
                deduped = dedupe_consecutive(out)
                out = deduped if deduped and not looks_degenerate(deduped) else raw
            final.append(out if out else raw)
        return final


def crop_box(image: Image.Image, box: list[int]) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1 - CROP_PAD), max(0, y1 - CROP_PAD)
    x2, y2 = min(w, x2 + CROP_PAD), min(h, y2 + CROP_PAD)
    return image.crop((x1, y1, x2, y2))


def iter_merged_targets(frames: dict[str, str], records: list[dict], skip_frame_ids: set[str] = frozenset()):
    """Yields (record_idx, text_idx, crop, raw_text) for every box that still
    looks like a merged blob, skipping any frame_id in skip_frame_ids (e.g.
    already corrected in a prior checkpointed run). Opens each frame's image
    at most once (only for frames that actually have a flagged box -- a small
    fraction of the dataset) and lets it go out of scope right after cropping,
    instead of caching across frames -- at full-batch1 scale (195k+ frames)
    caching every opened frame would hold gigabytes of decoded images for no
    reason, since each frame's boxes are only ever needed once."""
    for ri, rec in enumerate(records):
        if rec["frame_id"] in skip_frame_ids:
            continue
        boxes = [(ti, t) for ti, t in enumerate(rec["texts"])
                 if "box" in t and looks_merged_blob(t["text"])]
        if not boxes:
            continue
        path = frames.get(rec["frame_id"])
        if path is None:
            continue
        img = Image.open(path).convert("RGB")
        for ti, t in boxes:
            yield ri, ti, crop_box(img, t["box"]), t["text"]


def correct_merged_boxes(frames: dict[str, str], records: list[dict], model_dir: str) -> tuple[list[dict], int]:
    """records: [{"frame_id", "texts": [{"text", "box", ...}]}] (merge_recognizers.py
    output format). frames: {frame_id: image_path}. Corrects only boxes whose
    text still looks like a merged blob, in place. Returns (records, n_corrected).
    Whole-input-in-one-batch -- fine for a handful of frames (CLI/manual use);
    large-scale runs should chunk via VLMCorrector + iter_merged_targets directly
    (see ocr/pipeline.py::run_vlm_correct_pipeline) to checkpoint progress."""
    targets = list(iter_merged_targets(frames, records))
    if not targets:
        return records, 0

    corrector = VLMCorrector(model_dir)
    corrected = corrector.correct([t[2] for t in targets], [t[3] for t in targets])
    for (ri, ti, _, _), new_text in zip(targets, corrected):
        records[ri]["texts"][ti]["text"] = new_text
    return records, len(targets)
