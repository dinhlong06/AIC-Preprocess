#!/usr/bin/env bash
# Build + chạy Layer 1 (TransNetV2 shot detection + PhoWhisper ASR) trong Docker,
# cho dataset_batch1 (video prefix L, BTC AIC25). Ghi ra layer_1/batch1/ để không
# đụng shots.jsonl/whisper.jsonl của batch2 (prefix K) ở layer_1/.
#
# Cách dùng: giống run_layer1.sh, xem đó để biết các flag --skip_asr/--skip_shots/
# --videos/--force.
#
# Host này là GPU server dùng chung (8x RTX 2080 Ti) -> mặc định chọn 1 GPU
# đang rảnh nhất (free memory cao nhất) qua nvidia-smi, có thể override bằng
# biến môi trường GPU_ID.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="ai26-layer1"
INPUT_DIR="${VIDEO_DIR:-$PROJECT_ROOT/dataset_batch1/videos/video}"
# Override khi chạy nhiều shard song song: mỗi shard một OUTPUT_DIR riêng, vì
# nhiều process cùng append một jsonl trên NFS sẽ xé dòng giữa chừng.
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/batch1}"
CACHE_DIR="$SCRIPT_DIR/cache"
# Thư mục claim dùng chung khi chạy nhiều shard (run_shards_batch1.sh đặt); rỗng = chạy đơn.
CLAIMS_DIR="${CLAIMS_DIR:-}"
# Backend ASR: chunkformer là đích đến, phowhisper chỉ còn để đối chiếu.
ASR_BACKEND="${ASR_BACKEND:-chunkformer}"
MOUNTS=()
[[ -n "$CLAIMS_DIR" ]] && MOUNTS+=(-v "$CLAIMS_DIR:/data/claims")
SHOT_THRESHOLD="0.5"   # ngưỡng ranh giới shot của TransNetV2, sửa ở đây nếu cần

# --- Tham số Silero VAD (cắt audio theo đoạn có tiếng nói cho ASR), sửa ở đây ---
VAD_THRESHOLD="0.5"               # ngưỡng xác suất "có tiếng nói" (giảm nếu sót lời, tăng nếu nhiễu)
VAD_MAX_SPEECH_DURATION_S="28"    # giới hạn độ dài 1 đoạn, giây (giữ <30 = cửa sổ Whisper)
VAD_MIN_SILENCE_DURATION_MS="300" # khoảng lặng tối thiểu để tách 2 đoạn, ms
VAD_SPEECH_PAD_MS="100"           # đệm 2 đầu mỗi đoạn để không cụt âm đầu/cuối từ, ms

GPU_ID="${GPU_ID:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')}"

echo "== Build image $IMAGE_NAME =="
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

echo "== Chạy Layer 1 (batch1) trên GPU $GPU_ID =="
# shm-size: wav tạm ghi vào /dev/shm thay vì NFS; mặc định của docker chỉ 64 MB.
# cache: PhoWhisper-large ~3 GB, không mount thì mỗi lần chạy tải lại vào lớp ghi
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
    --asr_backend "$ASR_BACKEND" \
    "$@"
