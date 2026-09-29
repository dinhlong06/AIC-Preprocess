import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cpu_map_transcript import run_shot_transcript_mapping


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def _map(tmp_path, shots, segs):
    _write(tmp_path / "shots.jsonl", [{"video_id": "V", "fps": 25.0, **s} for s in shots])
    _write(tmp_path / "whisper.jsonl", [{"video_id": "V", **s} for s in segs])
    out = tmp_path / "out.jsonl"
    run_shot_transcript_mapping(str(tmp_path / "shots.jsonl"), str(tmp_path / "whisper.jsonl"), str(out))
    return {r["shot_id"]: r["text"] for r in map(json.loads, open(out, encoding="utf-8"))}


def test_shot_in_short_gap_after_speech_gets_that_speech(tmp_path):
    # L23_V017: câu kết thúc 101.5s, shot kế bắt đầu 101.88s -> trước đây "No ASR".
    got = _map(tmp_path,
               [{"shot_id": "s9", "start_frame": 2402, "end_frame": 2546},
                {"shot_id": "s10", "start_frame": 2547, "end_frame": 2578}],
               [{"start_ms": 98600, "end_ms": 101500, "text": "đã vào góc đường hồ tùng mậu"},
                {"start_ms": 105900, "end_ms": 110900, "text": "câu sau"}])
    assert got == {"s9": "đã vào góc đường hồ tùng mậu", "s10": "đã vào góc đường hồ tùng mậu"}


def test_shot_with_overlap_does_not_pull_in_near_neighbours(tmp_path):
    got = _map(tmp_path,
               [{"shot_id": "s", "start_frame": 100, "end_frame": 200}],
               [{"start_ms": 3700, "end_ms": 3900, "text": "trước"},
                {"start_ms": 4000, "end_ms": 8000, "text": "trong"},
                {"start_ms": 8100, "end_ms": 9000, "text": "sau"}])
    assert got == {"s": "trong"}


def test_shot_far_from_speech_stays_empty(tmp_path):
    got = _map(tmp_path,
               [{"shot_id": "s", "start_frame": 1000, "end_frame": 1100}],
               [{"start_ms": 0, "end_ms": 1000, "text": "xa"}])
    assert got == {}
