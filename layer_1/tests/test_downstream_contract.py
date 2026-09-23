import json
import sys
from pathlib import Path

_L2 = Path(__file__).resolve().parents[2] / "layer_2" / "shot_transcript"
sys.path.insert(0, str(_L2))

from cpu_map_transcript import run_shot_transcript_mapping
from gpu_shot_and_asr import _build_entries


def test_new_schema_still_maps_to_shots(tmp_path):
    shots = tmp_path / "shots.jsonl"
    whisper = tmp_path / "whisper.jsonl"
    out = tmp_path / "shot_transcripts.jsonl"

    shots.write_text(json.dumps({
        "video_id": "V1", "shot_id": "V1_000001",
        "start_frame": 0, "end_frame": 50, "fps": 25.0,
    }) + "\n", encoding="utf-8")

    entries = _build_entries("V1", [{"start": 0.5, "end": 1.5}],
                             [{"text": "xin chào", "confidence": -0.3}])
    whisper.write_text(json.dumps(entries[0], ensure_ascii=False) + "\n",
                       encoding="utf-8")

    run_shot_transcript_mapping(str(shots), str(whisper), str(out))
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["shot_id"] == "V1_000001"
    assert "xin chào" in rows[0]["text"]
