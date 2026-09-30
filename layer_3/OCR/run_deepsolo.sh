#!/usr/bin/env bash
# Stage 1 with DeepSolo detect + PARSeq-VN recognize -- same flow as
# run_paddle.sh but uses the ocr-deepsolo-parseq image (detectron2 +
# DeepSolo + strhub) and mounts experiments/deepsolo_parseq at
# /exp (detection weights + vn_scenetext PARSeq checkpoint, see
# ocr/deepsolo_engine.py).
#
#   ./run_deepsolo.sh --limit 60
#   FRAMES_DIR=/path OUTPUT_DIR=/path ./run_deepsolo.sh
#
# Env:
#   FRAMES_DIR   keyframe source (default: batch1 pipeline_g)
#   OUTPUT_DIR   JSON output dir (default: ./output)
#
# Output: output_deepsolo_parseq.json (+ output_deepsolo_origin.json, always empty --
# PARSeq replaces both recognizers).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXP_DIR="$SCRIPT_DIR/experiments/deepsolo_parseq"

IMAGE_NAME="ocr-deepsolo-parseq"
FRAMES_DIR="${FRAMES_DIR:-$PROJECT_ROOT/layer_2/Keyframe_Extracting/benchmark_batch1_v2/pipeline_g}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"
mkdir -p "$OUTPUT_DIR"

for f in "$EXP_DIR/weights/ic15_res50_finetune_synth-tt-mlt-13-15-textocr.pth" \
         "$EXP_DIR/vn_scenetext/weights/rec/best-parseq.ckpt" \
         "$EXP_DIR/DeepSolo/DeepSolo/configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml"; do
    [[ -f "$f" ]] || { echo "Missing $f -- see README.md (DeepSolo+PARSeq weights)"; exit 1; }
done

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Missing image $IMAGE_NAME (12.7GB, detectron2+DeepSolo+torch)."
    echo "Rebuild is not scripted -- it was built once from nvidia/cuda with"
    echo "detectron2 v0.6 + DeepSolo CUDA ext. Ask the team for the image tarball."
    exit 1
fi

GPU_ID="${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')}"

echo "== DeepSolo+PARSeq on GPU $GPU_ID (frames: $FRAMES_DIR) =="
docker run --rm \
    --gpus "device=$GPU_ID" \
    -e OMP_NUM_THREADS=2 \
    -v "$EXP_DIR":/exp \
    -v "$SCRIPT_DIR":/ocr \
    -v "$FRAMES_DIR:/data/frames:ro" \
    -v "$OUTPUT_DIR:/data/output" \
    -w /ocr "$IMAGE_NAME" \
    python3 run_paddle.py --engine deepsolo_parseq \
    --input /data/frames \
    --output /data/output/output_deepsolo_parseq.json \
    --output-paddle-origin /data/output/output_deepsolo_origin.json \
    "$@"
