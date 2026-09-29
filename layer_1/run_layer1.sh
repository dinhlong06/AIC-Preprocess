#!/usr/bin/env bash
# Build + chạy Layer 1 (TransNetV2 shot detection + ChunkFormer ASR) trong Docker.
#
# Cách dùng:
#   ./run_layer1.sh                        # chạy toàn bộ video trong $VIDEO_DIR
#   ./run_layer1.sh --skip_asr             # chỉ chạy shot detection (bỏ ASR)
#   ./run_layer1.sh --skip_shots           # chỉ chạy ASR (bỏ shot detection,
#                                          #   không load TensorFlow/TransNetV2)
#   ./run_layer1.sh --videos K01_V001.mp4,K01_V002.mp4
#   ./run_layer1.sh --skip_shots --videos K01_V001.mp4 --force
#                                          # làm lại 1 video đã có (bỏ qua resume),
#                                          #   tự xóa dòng cũ của nó, không đụng video khác
#
# Env:
#   VIDEO_DIR   thư mục video (mặc định dataset/video)
#   BATCH_DIR   nơi ghi shots.jsonl/whisper.jsonl (mặc định $SCRIPT_DIR)
#
# Host này là GPU server dùng chung (8x RTX 2080 Ti) -> mặc định chọn 1 GPU
# đang rảnh nhất (free memory cao nhất) qua nvidia-smi, có thể override bằng
# biến môi trường GPU_ID.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="ai26-layer1"
INPUT_DIR="${VIDEO_DIR:-$PROJECT_ROOT/dataset/video}"
# Override khi chạy nhiều shard song song: mỗi shard một OUTPUT_DIR riêng, vì
# nhiều process cùng append một jsonl trên NFS sẽ xé dòng giữa chừng.
OUTPUT_DIR="${OUTPUT_DIR:-${BATCH_DIR:-$SCRIPT_DIR}}"
CACHE_DIR="$SCRIPT_DIR/cache"
# Thư mục claim dùng chung khi chạy nhiều shard (run_shards.sh đặt); rỗng = chạy đơn.
CLAIMS_DIR="${CLAIMS_DIR:-}"
MOUNTS=()
[[ -n "$CLAIMS_DIR" ]] && MOUNTS+=(-v "$CLAIMS_DIR:/data/claims")
SHOT_THRESHOLD="0.5"   # ngưỡng ranh giới shot của TransNetV2, sửa ở đây nếu cần

# --- Tham số Silero VAD (cắt audio theo đoạn có tiếng nói cho ASR), sửa ở đây ---
VAD_THRESHOLD="0.5"               # ngưỡng xác suất "có tiếng nói" (giảm nếu sót lời, tăng nếu nhiễu)
VAD_MAX_SPEECH_DURATION_S="28"    # giới hạn độ dài 1 đoạn, giây
VAD_MIN_SILENCE_DURATION_MS="300" # khoảng lặng tối thiểu để tách 2 đoạn, ms
VAD_SPEECH_PAD_MS="100"           # đệm 2 đầu mỗi đoạn để không cụt âm đầu/cuối từ, ms

GPU_ID="${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')}"

echo "== Build image $IMAGE_NAME =="
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

echo "== Chạy Layer 1 trên GPU $GPU_ID =="
# shm-size: wav tạm ghi vào /dev/shm thay vì NFS; mặc định của docker chỉ 64 MB.
# cache: ChunkFormer ~1,5 GB, không mount thì mỗi lần chạy tải lại vào lớp ghi
# của container, tức vào ổ / của host vốn đã đầy.
docker run --rm \
    --gpus "device=$GPU_ID" \
    --shm-size=2g \
    -e PYTHONUNBUFFERED=1 \
    -v "$INPUT_DIR:/data/video:ro" \
    -v "$OUTPUT_DIR:/data/output" \
    -v "$CACHE_DIR:/root/.cache" \
    "${MOUNTS[@]}" \
    "$IMAGE_NAME" \
    --input_dir /data/video \
    --output_dir /data/output \
    --shot_threshold "$SHOT_THRESHOLD" \
    --vad_threshold "$VAD_THRESHOLD" \
    --vad_max_speech_duration_s "$VAD_MAX_SPEECH_DURATION_S" \
    --vad_min_silence_duration_ms "$VAD_MIN_SILENCE_DURATION_MS" \
    --vad_speech_pad_ms "$VAD_SPEECH_PAD_MS" \
    "$@"
