#!/usr/bin/env bash
# Build + chạy OCR stage 1 (PaddleOCR PP-OCRv6, GPU) trong Docker.
#
# Cách dùng:
#   ./run_paddle.sh                        # chạy full mọi video trong $FRAMES_DIR
#   ./run_paddle.sh --limit 60             # mọi flag thừa đều forward cho run_paddle.py
#
# Env:
#   FRAMES_DIR   thư mục keyframe nguồn
#   OUTPUT_DIR   nơi ghi JSON (mặc định ./output)
#
# Input: layer_2/Keyframe_Extracting/benchmark_batch1_v2/pipeline_g/ -- loader.py quét đệ quy
# nên tự gộp mọi video trong đó (frame_id đã unique toàn cục, không lo trùng).
# Output: output/output_vietocr.json (VietOCR + Paddle fallback)
#         output/output_paddle_origin.json (Paddle's own recognizer, uncorrected, for comparison)
#
# Host này là GPU server dùng chung -> mặc định chọn 1 GPU đang rảnh nhất
# (free memory cao nhất) qua nvidia-smi, có thể override bằng biến GPU_ID.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE_NAME="ocr-paddle"
FRAMES_DIR="${FRAMES_DIR:-$PROJECT_ROOT/layer_2/Keyframe_Extracting/benchmark_batch1_v2/pipeline_g}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"
mkdir -p "$OUTPUT_DIR" "$SCRIPT_DIR/cache/paddlex" "$SCRIPT_DIR/cache/torch"

GPU_ID="${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')}"

echo "== Build image $IMAGE_NAME =="
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

echo "== Chạy PaddleOCR trên GPU $GPU_ID (frames: $FRAMES_DIR) =="
docker run --rm \
    --gpus "device=$GPU_ID" \
    -v "$FRAMES_DIR:/data/frames:ro" \
    -v "$OUTPUT_DIR:/data/output" \
    -v "$SCRIPT_DIR/cache/paddlex:/root/.paddlex" \
    -v "$SCRIPT_DIR/cache/torch:/root/.cache/torch" \
    "$IMAGE_NAME" \
    --input /data/frames \
    --output /data/output/output_vietocr.json \
    --output-paddle-origin /data/output/output_paddle_origin.json \
    "$@"
