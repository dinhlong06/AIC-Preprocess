from gpu_shot_and_asr import _build_entries


def test_timestamps_come_from_vad_not_model():
    segments = [{"start": 1.25, "end": 2.5}]
    entries = _build_entries("L23_V008", segments, [{"text": "xin chào"}])
    assert entries[0]["start_ms"] == 1250
    assert entries[0]["end_ms"] == 2500


def test_schema_and_seg_id_format():
    segments = [{"start": 0.0, "end": 1.0}]
    entries = _build_entries("L23_V008", segments, [{"text": "a"}])
    assert entries[0] == {
        "video_id": "L23_V008",
        "seg_id": "L23_V008_000000",
        "start_ms": 0,
        "end_ms": 1000,
        "text": "a",
    }


def test_empty_text_skipped_but_still_consumes_index():
    segments = [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]
    results = [{"text": "   "}, {"text": "hai"}]
    entries = _build_entries("L23_V008", segments, results)
    assert len(entries) == 1
    assert entries[0]["seg_id"] == "L23_V008_000001"


def test_text_is_stripped():
    entries = _build_entries("V", [{"start": 0.0, "end": 1.0}], [{"text": "  a  "}])
    assert entries[0]["text"] == "a"
