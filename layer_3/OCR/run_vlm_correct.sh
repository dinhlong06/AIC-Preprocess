#!/usr/bin/env bash
# Chạy stage 3 (VLM correction) trong Docker.
#
#   ./run_vlm_correct.sh                       # 1 lần, dùng config.yaml mặc định
#   MAX_ATTEMPTS=15 ./run_vlm_correct.sh       # tự thử lại khi container chết sớm
#   FRAMES_DIR=/path ./run_vlm_correct.sh      # override frames_dir
#   ./run_vlm_correct.sh --merged in.json      # mọi flag thừa forward cho run_vlm_correct.py
#
# Vòng lặp thử lại chọn lại GPU rảnh nhất TẠI THỜI ĐIỂM ĐÓ (tranh chấp VRAM đổi
# liên tục), nếu container thoát sớm — OOM vì process khác chiếm giữa chừng —
# thì thử lại. An toàn vì run_vlm_correct.py checkpoint theo frame_id
# (output + .progress.json), lần sau resume đúng chỗ dừng.
#
# Dùng lại image ocr-v2-vllm (../OCR_v2, đã có torch+transformers+CUDA) thay
# vì build image riêng -- root disk host này gần đầy, build thêm 1 image
# torch+cuda mới (~5-8GB) từng thất bại vì hết chỗ. bitsandbytes (~50MB) được
# cài lúc chạy thay vì bake vào image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

FRAMES_DIR="${FRAMES_DIR:-$PROJECT_ROOT/layer_2/Keyframe_Extracting/benchmark_batch1_v2/pipeline_g}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"
MODEL_DIR="$SCRIPT_DIR/../OCR_v2/models/qwen3-vl-4b-vi-ocr"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"

[ -d "$MODEL_DIR" ] || { echo "Chưa có model, xem layer_3/OCR_v2/run_vllm.sh"; exit 1; }
mkdir -p "$OUTPUT_DIR"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    GPU_ID="${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')}"

    echo "== VLM correct trên GPU $GPU_ID (lần $attempt/$MAX_ATTEMPTS, frames: $FRAMES_DIR) =="
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
            $*" && exit 0

    echo "-- Lần thử $attempt thất bại (khả năng OOM do tranh chấp GPU), chờ 15s rồi thử lại --" >&2
    sleep 15
done

echo "Hết $MAX_ATTEMPTS lần thử vẫn chưa xong." >&2
exit 1
