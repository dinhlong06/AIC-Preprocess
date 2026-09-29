import json

from indexdb.ingest_batch1 import _load_gemma, _load_ocr


def test_load_ocr_removes_vietnamese_diacritics(tmp_path):
    path = tmp_path / "ocr.json"
    path.write_text(json.dumps([{
        "frame_id": "L22_V006_000000_kf0001",
        "texts": [
            {"text": "Đồng hồ: 18 giờ", "confidence": 0.9},
            {"text": "bỏ qua", "confidence": 0.4},
        ],
    }]), encoding="utf-8")

    assert _load_ocr(path) == {"L22_V006_000000_kf0001": "Dong ho: 18 gio"}


def test_load_gemma_keeps_diacritics_and_skips_empty(tmp_path):
    path = tmp_path / "gemma.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"frame_id": "L22_V006_000000_kf0001", "text": "Đồng hồ: 18 giờ"},
        {"frame_id": "L22_V006_000001_kf0001", "text": ""},
    ]), encoding="utf-8")

    assert _load_gemma(path) == {"L22_V006_000000_kf0001": "Đồng hồ: 18 giờ"}
