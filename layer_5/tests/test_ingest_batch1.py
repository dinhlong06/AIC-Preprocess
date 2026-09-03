import json

from indexdb.ingest_batch1 import _load_ocr


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
