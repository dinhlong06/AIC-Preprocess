import json
from compare_asr import compare


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                    encoding="utf-8")


def test_counts_segments_and_videos(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write(a, [{"video_id": "V1", "text": "xin chào"},
               {"video_id": "V1", "text": "hai"},
               {"video_id": "V2", "text": "ba"}])
    _write(b, [{"video_id": "V1", "text": "xin chao"}])
    out = compare(str(a), str(b))
    assert out["segments_a"] == 3
    assert out["segments_b"] == 1
    assert out["only_a"] == ["V2"]
    assert out["both"] == ["V1"]


def test_counts_empty_text(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write(a, [{"video_id": "V1", "text": "  "}, {"video_id": "V1", "text": "x"}])
    _write(b, [{"video_id": "V1", "text": "x"}])
    out = compare(str(a), str(b))
    assert out["empty_a"] == 1
    assert out["empty_b"] == 0
