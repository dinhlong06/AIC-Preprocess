#!/usr/bin/env bash
# Chạy stage 3 (VLM correction) trong Docker.
#
# Cách dùng:
#   ./run_vlm_correct.sh                       # dùng config.yaml mặc định
#   FRAMES_DIR=/path ./run_vlm_correct.sh      # override frames_dir
#   ./run_vlm_correct.sh --merged in.json      # mọi flag thừa forward cho run_vlm_correct.py
#
# Dùng lại image ocr-v2-vllm (../OCR_v2, đã có torch+transformers+CUDA) thay
# vì build image riêng -- root disk host này chronically gần đầy (xem
# Dockerfile chính), build thêm 1 image torch+cuda mới (~5-8GB) từng thất bại
# vì hết chỗ. bitsandbytes (~50MB) được cài lúc chạy thay vì bake vào image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

FRAMES_DIR="${FRAMES_DIR:-$PROJECT_ROOT/layer_2/Keyframe_Extracting/benchmark_batch1_v2/pipeline_g}"
OUTPUT_DIR="$SCRIPT_DIR/output"
MODEL_DIR="$SCRIPT_DIR/../OCR_v2/models/qwen3-vl-4b-vi-ocr"

[ -d "$MODEL_DIR" ] || { echo "Chưa có model, xem layer_3/OCR_v2/run_vllm.sh"; exit 1; }
mkdir -p "$OUTPUT_DIR"

GPU_ID="${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')}"

echo "== Chạy VLM correct trên GPU $GPU_ID (frames: $FRAMES_DIR) =="
docker run --rm \
    --gpus "device=$GPU_ID" \
    --shm-size=8g \
    -v "$FRAMES_DIR:/data/frames:ro" \
    -v "$OUTPUT_DIR:/data/output" \
    -v "$MODEL_DIR:/model:ro" \
    -v "$SCRIPT_DIR:/workspace:ro" \
    --entrypoint bash \
    ocr-v2-vllm \
    -c "pip install -q bitsandbytes && cd /workspace && python3 run_vlm_correct.py \
        --frames /data/frames \
        --merged /data/output/output_vietocr_merged.json \
        --output /data/output/output_vlm_corrected.json \
        --model-dir /model \
        $*"
