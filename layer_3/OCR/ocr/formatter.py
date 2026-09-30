"""
formatter.py -- output JSON + checkpoint I/O for the OCR pipeline.

Output schema (per frame):
{
    "frame_id": "K01_V001_000001_kf0001",
    "texts": [{"text": "...", "confidence": 0.9823, "box": [x1, y1, x2, y2]}, ...]
}

`box` is DeepSolo's axis-aligned detection rectangle -- a box touching the frame
edge marks a line the scrolling ticker cut off mid-sentence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def load_checkpoint(output_file: str) -> dict[str, dict]:
    """Read a previous run's output (if any) keyed by frame_id, for --checkpoint-every resume."""
    path = Path(output_file)
    if not path.exists():
        return {}
    try:
        return {r["frame_id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
    except (json.JSONDecodeError, OSError):
        return {}


def save_output(records: list[dict], output_path: str) -> None:
    """Write .tmp then atomically replace -- a crash mid-write must not leave a truncated JSON."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(out)
