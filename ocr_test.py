"""
Đọc toàn bộ chữ (tiếng Việt) trong một ảnh bằng PaddleOCR-VL-1.6.

    from ocr_test import image_to_text
    print(image_to_text("ocr.jpg"))

Chạy trực tiếp:
    VENV=/workingspace_aiclub/WorkingSpace/Personal/vannk
    HF_HOME=$VENV/.hf-cache $VENV/.venv-ocrvl/bin/python ocr_test.py ocr.jpg

HF_HOME bắt buộc trỏ ra ngoài $HOME: ổ / trên máy này đã đầy 100%.

Hai lượt gọi model thay vì một: chạy thẳng "Spotting:" trên ảnh full thì model
nuốt mất dấu tiếng Việt khi dòng chữ nhỏ ("vị trí tốt nhất" -> "vi tri tot
nhat"), nên lượt 1 chỉ lấy toạ độ, lượt 2 đọc lại từng box đã phóng to.
"""

import re
import sys

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"
MIN_CROP_H = 48  # dưới ngưỡng này model bắt đầu rụng dấu -- đo trên ocr.jpg
PAD = 4
_cached = None


def _load():
    global _cached
    if _cached is None:
        _, gpu = max((torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count()))
        _cached = (
            AutoProcessor.from_pretrained(MODEL),
            AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float16).to(f"cuda:{gpu}").eval(),
        )
    return _cached


def _generate(img, task, max_new_tokens):
    proc, model = _load()
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": task}]}]
    inputs = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def image_to_text(image):
    """Nhận đường dẫn ảnh hoặc PIL.Image, trả về mọi chữ đọc được, mỗi vùng một dòng."""
    img = Image.open(image) if isinstance(image, str) else image
    img = img.convert("RGB")
    w, h = img.size

    lines = []
    for spotted in _generate(img, "Spotting:", 2048).split("\n"):
        locs = [int(v) for v in re.findall(r"<\|LOC_(\d+)\|>", spotted)]
        if len(locs) < 8:
            continue
        xs, ys = locs[0::2], locs[1::2]  # LOC chuẩn hoá 0-1000 theo width/height
        x1, y1 = max(0, min(xs) * w // 1000 - PAD), max(0, min(ys) * h // 1000 - PAD)
        x2, y2 = min(w, max(xs) * w // 1000 + PAD), min(h, max(ys) * h // 1000 + PAD)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img.crop((x1, y1, x2, y2))
        if crop.height < MIN_CROP_H:
            crop = crop.resize((round(crop.width * MIN_CROP_H / crop.height), MIN_CROP_H), Image.LANCZOS)
        text = _generate(crop, "OCR:", 256).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


if __name__ == "__main__":
    print(image_to_text(sys.argv[1]))
