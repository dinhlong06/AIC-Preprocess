#!/usr/bin/env bash
# Build + chạy Layer 3 - Object Detection (YOLO/ultralytics) trong Docker,
# nguồn cố định là output pipeline_g của layer_2/Keyframe_Extracting (batch1).
#
# Cách dùng:
#   ./run_detect_batch1.sh                     # chạy full benchmark_batch1/pipeline_g
#   ./run_detect_batch1.sh --model yolo11m.pt  # đổi model
#   ./run_detect_batch1.sh --no-gpu            # ép chạy CPU
#
# Input: layer_2/Keyframe_Extracting/benchmark_batch1/pipeline_g/ — loader.py quét đệ quy
# nên tự gộp mọi video trong đó (tên file keyframe_id đã unique toàn cục, không lo trùng).
# Override: FRAMES_DIR=/path/khac ./run_detect_batch1.sh
# Output: detections.json ghi vào ./output_batch1/ (gộp chung mọi video).
#
# Host này là GPU server dùng chung -> mặc định chọn 1 GPU đang rảnh nhất
# (free memory cao nhất) qua nvidia-smi, có thể override bằng biến GPU_ID.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE_NAME="ai26-layer3-detect"
FRAMES_DIR="${FRAMES_DIR:-$PROJECT_ROOT/layer_2/Keyframe_Extracting/benchmark_batch1/pipeline_g}"
OUTPUT_DIR="$SCRIPT_DIR/output_batch1"
# Không mount cache thì ultralytics tải lại weight mỗi lần chạy vào lớp ghi của
# container, tức vào ổ / của host vốn đã đầy.
mkdir -p "$OUTPUT_DIR" "$SCRIPT_DIR/cache/ultralytics" "$SCRIPT_DIR/cache/weights"

GPU_ID="${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')}"

echo "== Build image $IMAGE_NAME =="
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

echo "== Chạy Object Detection trên GPU $GPU_ID =="
docker run --rm \
    --gpus "device=$GPU_ID" \
    -v "$FRAMES_DIR:/data/frames:ro" \
    -v "$OUTPUT_DIR:/data/output" \
    -v "$SCRIPT_DIR/cache/ultralytics:/root/.config/Ultralytics" \
    -v "$SCRIPT_DIR/cache/weights:/workspace/weights" \
    "$IMAGE_NAME" \
    --input /data/frames \
    --output /data/output/detections.json \
    "$@"
