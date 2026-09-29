#!/usr/bin/env bash
# Layer 2 (CPU): ghép transcript (whisper.jsonl) vào shot (shots.jsonl) cho batch1.
# chạy thẳng bằng python3 hệ thống.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/cpu_map_transcript.py" \
    --shots_path "${L1_DIR:-$SCRIPT_DIR/../../layer_1/batch1}/shots.jsonl" \
    --whisper_path "${L1_DIR:-$SCRIPT_DIR/../../layer_1/batch1}/whisper.jsonl" \
    --output_path "${OUT:-$SCRIPT_DIR/shot_transcripts_batch1.jsonl}"
