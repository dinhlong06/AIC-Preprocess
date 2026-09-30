#!/usr/bin/env bash
# OCR stage 1: DeepSolo detect + PARSeq-VN recognize, in Docker.
#
#   ./run.sh                               # 1 GPU, container đầy đủ
#   FRAMES_DIR=/path OUTPUT_DIR=/path ./run.sh
#   ./run.sh --limit 60                    # mọi flag thừa forward cho run_ocr.py
#
# Env:
#   FRAMES_DIR   thư mục keyframe nguồn (mặc định batch1 pipeline_g)
#   OUTPUT_DIR   nơi ghi JSON (mặc định ./output)
#
# Dựng image bằng ./download_weights.sh + docker build -t ocr-deepsolo-parseq .
# Weight (163MB + 274MB) vượt giới hạn 100MB của GitHub nên tải qua script.
# Host này là GPU server dùng chung -> mặc định chọn 1 GPU đang rảnh nhất
# (free memory cao nhất) qua nvidia-smi, có thể override bằng biến GPU_ID.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXP_DIR="$SCRIPT_DIR/experiments/deepsolo_parseq"

IMAGE_NAME="ocr-deepsolo-parseq"
FRAMES_DIR="${FRAMES_DIR:-$PROJECT_ROOT/layer_2/Keyframe_Extracting/benchmark_batch1_v2/pipeline_g}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"
mkdir -p "$OUTPUT_DIR"

"$SCRIPT_DIR/download_weights.sh"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Missing image $IMAGE_NAME -- build lại từ repo:" >&2
    echo "  docker build -t $IMAGE_NAME $SCRIPT_DIR" >&2
    echo "hoặc khôi phục từ backup: docker load -i $PROJECT_ROOT/ocr-deepsolo-parseq-backup.tar" >&2
    exit 1
fi

GPU_ID="${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')}"

echo "== DeepSolo+PARSeq trên GPU $GPU_ID (frames: $FRAMES_DIR) =="
docker run --rm \
    --gpus "device=$GPU_ID" \
    -e OMP_NUM_THREADS=2 \
    -v "$EXP_DIR:/exp" \
    -v "$SCRIPT_DIR:/ocr" \
    -v "$FRAMES_DIR:/data/frames:ro" \
    -v "$OUTPUT_DIR:/data/output" \
    -w /ocr "$IMAGE_NAME" \
    python3 run_ocr.py \
    --input /data/frames \
    --output /data/output/output_ocr.json \
    "$@"
