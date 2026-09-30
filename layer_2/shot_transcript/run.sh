#!/usr/bin/env bash
# Layer 2 (CPU): ghép transcript (whisper.jsonl) của layer 1 vào shot (shots.jsonl),
# chạy thẳng bằng python3 hệ thống.
#
#   ./run.sh                          # batch2 (prefix K): layer_1/ -> shot_transcripts.jsonl
#   L1_DIR=../../layer_1/batch1 OUT=shot_transcripts_batch1.jsonl ./run.sh
#
# Env:
#   L1_DIR   thư mục layer_1 chứa shots.jsonl + whisper.jsonl (mặc định ../../layer_1)
#   OUT      file output (mặc định shot_transcripts.jsonl cạnh script)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L1_DIR="${L1_DIR:-$SCRIPT_DIR/../../layer_1}"
OUT="${OUT:-$SCRIPT_DIR/shot_transcripts.jsonl}"

python3 "$SCRIPT_DIR/cpu_map_transcript.py" \
    --shots_path "$L1_DIR/shots.jsonl" \
    --whisper_path "$L1_DIR/whisper.jsonl" \
    --output_path "$OUT"
