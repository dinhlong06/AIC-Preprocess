from indexdb.builders import build_shot_doc, build_frame_doc, build_seg_doc

def test_build_shot_doc():
    rec = {"video_id": "K01_V001", "shot_id": "K01_V001_000007", "start_frame": 1520,
           "end_frame": 1655, "fps": 25.0}
    doc = build_shot_doc(rec, transcript="xin chào")
    assert doc["_id"] == "K01_V001_000007"
    assert doc["video_id"] == "K01_V001"
    assert doc["shot_id"] == "K01_V001_000007"
    assert doc["transcript"] == "xin chào"
    assert doc["start_ms"] == 60800
    assert doc["end_ms"] == 66240

def test_build_frame_doc_minimal():
    rec = {"video_id": "K01_V001", "keyframe_id": "K01_V001_000007_kf0001",
           "shot_id": "K01_V001_000007", "frame_idx": 1523, "timestamp_ms": 60920,
           "image_path": "/data/keyframes/K01_V001/K01_V001_000007_kf0001.jpg"}
    doc = build_frame_doc(rec)
    assert doc["_id"] == "K01_V001_000007_kf0001"
    assert doc["frame_idx"] == 1523
    assert doc["objects"] == {}
    assert doc["ocr_text"] == ""
    assert doc["ocr_api"] == ""
    assert doc["caption"] == ""
    assert doc["embeddings"] == {"beit3": False, "siglip2": False}
    assert doc["synced"] == {"elastic": False, "milvus": False}

def test_build_seg_doc():
    rec = {"video_id": "K01_V001", "seg_id": "K01_V001_000003", "start_ms": 60500,
           "end_ms": 66800, "text": "..."}
    doc = build_seg_doc(rec)
    assert doc["_id"] == "K01_V001_000003"
    assert doc["text"] == "..."
