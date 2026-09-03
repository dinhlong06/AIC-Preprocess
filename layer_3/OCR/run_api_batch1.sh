#!/usr/bin/env bash
# Stage 2 (ising-calibration correction) cho dataset_batch1, dùng
# output_batch1/output_vietocr.json đã có sẵn (chạy xong ./run_paddle_batch1.sh).
# NVIDIA_API_KEY đọc từ .env (xem .env.example) hoặc từ biến môi trường đã export.
#
# Cách dùng:
#   cp .env.example .env && vi .env       # điền NVIDIA_API_KEY, làm 1 lần
#   ./run_api_batch1.sh
#
# Input:  output_batch1/output_vietocr.json
# Output: output_batch1/output_hybrid.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -z "${NVIDIA_API_KEY:-}" && -f "$SCRIPT_DIR/.env" ]]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi
if [[ -z "${NVIDIA_API_KEY:-}" ]]; then
    echo "[x] NVIDIA_API_KEY chưa có -- điền vào .env (xem .env.example) hoặc export trước khi chạy." >&2
    exit 1
fi

VIETOCR_OUTPUT="$SCRIPT_DIR/output_batch1/output_vietocr.json"
if [[ ! -f "$VIETOCR_OUTPUT" ]]; then
    echo "[x] $VIETOCR_OUTPUT không tồn tại -- chạy ./run_paddle_batch1.sh trước." >&2
    exit 1
fi

echo "== Stage 2: ising-calibration correction (host, batch1) =="
(cd "$SCRIPT_DIR" && python3 run_correct.py \
    --frames "$PROJECT_ROOT/layer_2/Keyframe_Extracting/benchmark_batch1/pipeline_g" \
    --paddle-output "$VIETOCR_OUTPUT" \
    --output "$SCRIPT_DIR/output_batch1/output_hybrid.json")

echo "== Xong. Kết quả cuối: $SCRIPT_DIR/output_batch1/output_hybrid.json =="
