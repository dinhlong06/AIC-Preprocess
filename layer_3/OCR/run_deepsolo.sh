#!/usr/bin/env bash
# Stage 1 with DeepSolo detect + PARSeq-VN recognize -- same flow as
# run_paddle.sh but uses the ocr-deepsolo-parseq image (detectron2 +
# DeepSolo CUDA ext + PARSeq) and mounts experiments/deepsolo_parseq at
# /exp (weights + vn_scenetext, see ocr/deepsolo_engine.py).
#
# Cách dùng: như run_paddle.sh
#   ./run_deepsolo.sh --limit 60
#   PIPELINE=pipeline_a ./run_deepsolo.sh
# Output: output/output_deepsolo_parseq.json (+ output_deepsolo_origin.json, luôn rỗng)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXP_DIR="$SCRIPT_DIR/experiments/deepsolo_parseq"

IMAGE_NAME="ocr-deepsolo-parseq"
PIPELINE="${PIPELINE:-pipeline_c}"
FRAMES_DIR="${FRAMES_DIR:-$PROJECT_ROOT/layer_2/Keyframe_Extracting/benchmark/$PIPELINE}"
OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"

GPU_ID="${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')}"

echo "== Build image $IMAGE_NAME (cached) =="
docker build -q -t "$IMAGE_NAME" "$EXP_DIR" >/dev/null

echo "== DeepSolo+PARSeq on GPU $GPU_ID (frames: $FRAMES_DIR) =="
docker run --rm \
    --gpus "device=$GPU_ID" \
    -e OMP_NUM_THREADS=2 \
    -v "$EXP_DIR":/exp \
    -v "$SCRIPT_DIR":/ocr \
    -v "$FRAMES_DIR":/data/frames:ro \
    -v "$OUTPUT_DIR":/data/output \
    -w /ocr "$IMAGE_NAME" \
    python3 run_paddle.py --engine deepsolo_parseq \
    --input /data/frames \
    --output /data/output/output_deepsolo_parseq.json \
    --output-paddle-origin /data/output/output_deepsolo_origin.json \
    "$@"
