#!/usr/bin/env bash
# Vòng lặp tự phục hồi cho run_vlm_correct.py trên cụm GPU dùng chung: mỗi
# lần thử chọn lại GPU rảnh nhất TẠI THỜI ĐIỂM ĐÓ (tranh chấp VRAM đổi liên
# tục), nếu container thoát sớm (OOM do process khác chiếm giữa chừng) thì
# thử lại -- an toàn vì run_vlm_correct.py đã checkpoint theo frame_id
# (output_file + .progress.json), lần thử sau resume đúng chỗ dừng lại,
# không làm lại từ đầu. Cùng ý tưởng OCR_v2/run_resilient.sh.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMES_DIR="${FRAMES_DIR:?set FRAMES_DIR}"
MERGED_JSON="${MERGED_JSON:?set MERGED_JSON}"
OUTPUT_JSON="${OUTPUT_JSON:?set OUTPUT_JSON}"
MODEL_DIR="$SCRIPT_DIR/../OCR_v2/models/qwen3-vl-4b-vi-ocr"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-15}"

OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_JSON")" && pwd)"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    GPU_ID="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')"
    echo "== Lần thử $attempt/$MAX_ATTEMPTS trên GPU $GPU_ID ==" >&2

    docker run --rm --gpus "device=$GPU_ID" --shm-size=8g \
        -v "$FRAMES_DIR:/data/frames:ro" \
        -v "$OUTPUT_DIR:/data/output" \
        -v "$MODEL_DIR:/model:ro" \
        -v "$SCRIPT_DIR:/workspace:ro" \
        --entrypoint bash \
        ocr-v2-vllm \
        -c "pip install -q bitsandbytes && cd /workspace && python3 run_vlm_correct.py \
            --frames /data/frames --merged /data/output/$(basename "$MERGED_JSON") \
            --output /data/output/$(basename "$OUTPUT_JSON") --model-dir /model"

    if [ $? -eq 0 ]; then
        echo "== Xong sau $attempt lần thử ==" >&2
        exit 0
    fi
    echo "-- Lần thử $attempt thất bại (khả năng OOM do tranh chấp GPU), chờ 15s rồi thử lại --" >&2
    sleep 15
done

echo "Hết $MAX_ATTEMPTS lần thử vẫn chưa xong." >&2
exit 1
