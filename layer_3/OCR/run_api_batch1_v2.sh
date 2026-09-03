#!/usr/bin/env bash
# Stage 2 cho dataset_batch1_v2 (keyframe mới, xem run_paddle_batch1_shards.sh),
# dùng output_batch1_v2/output_vietocr.json đã có sẵn.
#
# Stage 2 nay chỉ là pass no-op qua correct_record_locally (xem ocr/pipeline.py
# run_correct_pipeline) -- stage 1 đã tự sửa lỗi cục bộ ngay khi OCR rồi, không
# còn gọi API ising-calibration nữa, không cần NVIDIA_API_KEY.
#
# Cách dùng:
#   ./run_api_batch1_v2.sh
#
# Input:  output_batch1_v2/output_vietocr.json
# Output: output_batch1_v2/output_hybrid.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VIETOCR_OUTPUT="$SCRIPT_DIR/output_batch1_v2/output_vietocr.json"
if [[ ! -f "$VIETOCR_OUTPUT" ]]; then
    echo "[x] $VIETOCR_OUTPUT không tồn tại -- chạy ./run_paddle_batch1_shards.sh + merge trước." >&2
    exit 1
fi

echo "== Stage 2: local correction pass (host, batch1_v2) =="
(cd "$SCRIPT_DIR" && python3 run_correct.py \
    --frames "$PROJECT_ROOT/layer_2/Keyframe_Extracting/benchmark_batch1_v2/pipeline_g" \
    --paddle-output "$VIETOCR_OUTPUT" \
    --output "$SCRIPT_DIR/output_batch1_v2/output_hybrid.json")

echo "== Xong. Kết quả cuối: $SCRIPT_DIR/output_batch1_v2/output_hybrid.json =="
