#!/usr/bin/env bash
# run.sh — chạy Keyframe Extractor bằng Docker (mặc định) hoặc trực tiếp host.
#
# Input thật lấy từ project root (chung với Layer 1):
#   video   : $PROJECT_ROOT/dataset/video
#   shots   : $PROJECT_ROOT/layer_1/shots.jsonl (1 file gộp mọi video)
# Layer 2 (runner.py) đòi 1 file shots/video, nên script tự tách shots.jsonl
# theo video_id ra dataset/shots/<video_id>.jsonl trước khi gọi cli.py.
#
# Dùng:
#   ./run.sh                         # pipeline_g, Docker
#   ./run.sh all                     # chạy cả 4 pipeline
#   ./run.sh pipeline_b --device cpu # truyền thêm flag CLI tuỳ ý
#   MODE=host ./run.sh pipeline_g    # chạy python3 cli.py thẳng, không Docker
#
# Đổi dataset bằng env, phải là ĐƯỜNG DẪN TUYỆT ĐỐI vì chúng được docker mount
# (run_shards_batch1.sh dùng cách này):
#   VIDEO_DIR, SHOTS_SRC, SHOTS_SPLIT_DIR, OUTPUT_DIR, GT_DIR
#
# Mọi tham số sau tên pipeline được chuyển thẳng cho cli.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

PIPELINE="${1:-pipeline_g}"; shift || true
MODE="${MODE:-docker}"          # docker | host
IMAGE="${IMAGE:-ai26-layer2}"

# Mặc định = batch2 (video prefix K). run_shards_batch1.sh override cả 4 biến này
# để trỏ sang dataset_batch1 mà không phải nhân bản script.
VIDEO_DIR_HOST="${VIDEO_DIR:-$PROJECT_ROOT/dataset/video}"
GT_DIR_HOST="${GT_DIR:-$PROJECT_ROOT/dataset/gt_keyframes}"
SHOTS_SRC="${SHOTS_SRC:-$PROJECT_ROOT/layer_1/shots.jsonl}"
SHOTS_SPLIT_DIR="${SHOTS_SPLIT_DIR:-$SCRIPT_DIR/dataset/shots}"
KEYFRAME_OUT_HOST="${OUTPUT_DIR:-$SCRIPT_DIR/benchmark}"

if [[ ! -d "$SHOTS_SPLIT_DIR" ]]; then
  rm -rf "$SHOTS_SPLIT_DIR" 2>/dev/null || true
  mkdir -p "$SHOTS_SPLIT_DIR"
fi
if [[ ! -d "$KEYFRAME_OUT_HOST/$PIPELINE" ]]; then
  rm -rf "$KEYFRAME_OUT_HOST/$PIPELINE" 2>/dev/null || true
  mkdir -p "$KEYFRAME_OUT_HOST/$PIPELINE"
fi

HAS_GT_DIR=0
for arg in "$@"; do
  if [[ "$arg" == "--gt_dir="* ]] || [[ "$arg" == "--gt_dir" ]]; then
    HAS_GT_DIR=1
    break
  fi
done

if [[ -f "$SHOTS_SRC" ]]; then
  python3 - "$SHOTS_SRC" "$SHOTS_SPLIT_DIR" <<'PYEOF'
import collections, json, sys

src, out_dir = sys.argv[1], sys.argv[2]
by_video = collections.defaultdict(list)
with open(src, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            by_video[json.loads(line)["video_id"]].append(line)

for video_id, lines in by_video.items():
    with open(f"{out_dir}/{video_id}.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
PYEOF
else
  echo "[run.sh] WARNING: Không thấy $SHOTS_SRC — Layer 2 sẽ tự chunk video (không dùng shot thật)."
fi

EXTRA_ARGS=()
if [[ $HAS_GT_DIR -eq 0 && -d "$GT_DIR_HOST" ]]; then
  if [[ "$MODE" == "host" ]]; then
    EXTRA_ARGS+=(--gt_dir "$GT_DIR_HOST")
  else
    EXTRA_ARGS+=(--gt_dir "dataset/gt_keyframes")
  fi
fi

if [[ "$MODE" == "host" ]]; then
  exec python3 cli.py --pipeline "$PIPELINE" \
    --video_dir "$VIDEO_DIR_HOST" \
    --shots_dir "$SHOTS_SPLIT_DIR" \
    --output_dir "$KEYFRAME_OUT_HOST" \
    "${EXTRA_ARGS[@]}" \
    "$@"
fi

echo "[run.sh] Build image $IMAGE..."
[[ -n "${SKIP_BUILD:-}" ]] || docker build -t "$IMAGE" "$SCRIPT_DIR"

GT_VOL_MOUNT=()
if [[ -d "$GT_DIR_HOST" ]]; then
  GT_VOL_MOUNT=(-v "$GT_DIR_HOST:/app/dataset/gt_keyframes:ro")
fi

# Máy dùng chung 8 GPU: ghim đúng 1 GPU thay vì --gpus all.
MIN_FREE_MB="${MIN_FREE_MB:-4000}"
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', *' -v m="$MIN_FREE_MB" '$2 >= m' | sort -t',' -k2 -n -r \
    | head -1 | cut -d',' -f1 | tr -d ' '
}

# BEiT-3 Large chiếm 3552 MiB đo bằng nvidia-smi (statistics.json ghi 2.97 GB vì
# torch.cuda chỉ đếm tensor, bỏ qua CUDA context + phần allocator giữ sẵn).
# Giữa lúc chọn GPU và lúc model nạp xong có vài chục giây build/khởi động, đủ để
# người khác chiếm chỗ -> đã OOM 3 lần theo đúng kịch bản này. Chọn lại GPU ở mỗi
# lần thử thay vì bám GPU đã hết chỗ.
for attempt in 1 2 3; do
  gpu="${GPU_ID:-$(pick_gpu)}"
  if [[ -z "$gpu" ]]; then
    echo "[run.sh] Lần $attempt: không GPU nào còn >= ${MIN_FREE_MB} MB, đợi 60s..."
    sleep 60
    GPU_ID=""; continue
  fi
  echo "[run.sh] Lần $attempt: GPU $gpu, DAKE_THREADS=${DAKE_THREADS:-2}"

  # Nhãn cố định để tìm lại container: --filter ancestor=<tên ảnh> phân giải sang
  # image ID hiện tại, nên build lại ảnh là mất dấu mọi container đang chạy.
  docker run --rm \
    --label app=ai26-layer2 \
    --shm-size=8g \
    --gpus "device=$gpu" \
    -e PYTHONUNBUFFERED=1 \
    -e DAKE_THREADS="${DAKE_THREADS:-2}" \
    -e OMP_NUM_THREADS=1 \
    -e OPENCV_FOR_THREADS_NUM=1 \
    -e SHARD_ID="${SHARD_ID:-}" \
    -e CLAIMS_DIR="${CLAIMS_DIR:-}" \
    -v "$VIDEO_DIR_HOST:/app/dataset/raw_video:ro" \
    -v "$SHOTS_SPLIT_DIR:/app/dataset/shots" \
    -v "$KEYFRAME_OUT_HOST:/app/benchmark" \
    "${GT_VOL_MOUNT[@]}" \
    "$IMAGE" \
    --pipeline "$PIPELINE" \
    --video_dir dataset/raw_video \
    --shots_dir dataset/shots \
    "${EXTRA_ARGS[@]}" \
    "$@" && exit 0

  # GPU_ID do người gọi ghim đã mất chỗ; lần sau tự chọn GPU khác.
  GPU_ID=""
  echo "[run.sh] Lần $attempt thất bại, chọn GPU khác sau 30s..."
  sleep 30
done

echo "[run.sh] Thất bại sau 3 lần thử."
exit 1
