#!/usr/bin/env bash
# Worker-pool song song cho OCR stage 1 (DeepSolo+PARSeq). Mỗi container tự
# claim video CHƯA ai làm qua claims dir dùng chung (tạo file O_EXCL, atomic
# kể cả trên NFS) -- nên thêm/bớt container bất cứ lúc nào KHÔNG cần tính lại
# từ đầu (video không gán cứng theo index, NSHARDS đổi lúc nào cũng được).
#
# Cách dùng:
#   ./run_shards.sh                             # launch N container nền (N = số GPU rảnh)
#   NSHARDS=4 ./run_shards.sh
#   FRAMES_DIR=/path/khac OUTPUT_DIR=/path/khac ./run_shards.sh
#   ./run_shards.sh add                         # thêm 1 container vào GPU rảnh nhất hiện tại
#   ./run_shards.sh add 3                       # thêm 1 container, ép GPU 3
#   ./run_shards.sh status                      # xem tiến độ (video xong / tổng / đang claim)
#   ./run_shards.sh release                     # nhả claim dở dang (container chết giữa chừng, chưa ghi done)
#   ./run_shards.sh merge                       # gộp claims/done/*.json -> output_ocr.json
#
# Chạy merge sau khi status báo done == tổng số video.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXP_DIR="$SCRIPT_DIR/experiments/deepsolo_parseq"

IMAGE_NAME="ocr-deepsolo-parseq"
FRAMES_DIR="${FRAMES_DIR:-$PROJECT_ROOT/layer_2/Keyframe_Extracting/benchmark_batch1_v2/pipeline_g}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"
CLAIMS_DIR="$OUTPUT_DIR/claims"
# DeepSolo R50 + PARSeq trên 1 GPU 2080 Ti (11 GB) dùng chung máy -- để chỗ cho
# process khác, đòi tối thiểu 3 GB free trước khi giao shard vào GPU đó.
MIN_FREE_MB="${MIN_FREE_MB:-3000}"
# Bỏ qua GPU đang tính toán bận dù còn dư VRAM, để không giành tài nguyên với
# job khác đang chạy trên máy chung này.
MAX_UTIL="${MAX_UTIL:-50}"
# Stage này là vòng lặp Python đơn luồng (không multi-thread), CPU chỉ dùng để
# decode/resize ảnh -- 2 core/shard là đủ dư. Ép ở tầng biến môi trường (không
# dùng docker --cpus/--cpuset-cpus): host này chạy kernel không có CFS
# bandwidth controller / cpuset cgroup nên cả hai flag đó đều bị Docker từ
# chối hoặc discard.
CPUS_PER_SHARD="${CPUS_PER_SHARD:-2}"

mkdir -p "$OUTPUT_DIR" "$CLAIMS_DIR/claimed" "$CLAIMS_DIR/done"

if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Missing image $IMAGE_NAME -- khôi phục từ ocr-deepsolo-parseq-backup.tar:" >&2
    echo "  docker load -i <backup>/ocr-deepsolo-parseq-backup.tar" >&2
    exit 1
fi

pick_gpus() {
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
        | awk -F', *' -v m="$MIN_FREE_MB" -v u="$MAX_UTIL" '$2 >= m && $3 <= u' | sort -t',' -k2 -n -r \
        | head -"$1" | cut -d',' -f1 | tr -d ' '
}

run_shard() {
    local gpu="$1" log="$OUTPUT_DIR/shard_$(date +%s%N).log"
    echo "  -> GPU $gpu, log: $log"
    docker run --rm \
        --gpus "device=$gpu" \
        -e OMP_NUM_THREADS="$CPUS_PER_SHARD" \
        -e OPENBLAS_NUM_THREADS="$CPUS_PER_SHARD" \
        -e MKL_NUM_THREADS="$CPUS_PER_SHARD" \
        -v "$EXP_DIR:/exp" \
        -v "$SCRIPT_DIR:/ocr" \
        -v "$FRAMES_DIR:/data/frames:ro" \
        -v "$CLAIMS_DIR:/data/claims" \
        -w /ocr "$IMAGE_NAME" \
        python3 run_ocr.py \
        --input /data/frames \
        --claims-dir /data/claims \
        > "$log" 2>&1 &
    disown
}

case "${1:-}" in
add)
    gpu="${2:-$(pick_gpus 1)}"
    [[ -z "$gpu" ]] && { echo "TỪ CHỐI: không GPU nào còn >= ${MIN_FREE_MB} MB trống / util <= ${MAX_UTIL}%." >&2; exit 1; }
    echo "== Thêm 1 container =="
    run_shard "$gpu"
    exit 0
    ;;
status)
    total=$(ls "$FRAMES_DIR" | wc -l)
    done_n=$(ls "$CLAIMS_DIR/done" 2>/dev/null | wc -l)
    claimed_n=$(ls "$CLAIMS_DIR/claimed" 2>/dev/null | wc -l)
    echo "$done_n/$total video xong, $claimed_n đã claim (gồm cả đang xử lý dở)."
    exit 0
    ;;
release)
    n=0
    for f in "$CLAIMS_DIR/claimed"/*; do
        [[ -f "$f" ]] || continue
        v="$(basename "$f")"
        [[ -f "$CLAIMS_DIR/done/$v.json" ]] && continue
        rm -f "$f"
        n=$((n + 1))
    done
    echo "đã nhả $n claim dở dang -- container mới/đang chạy sẽ nhận lại."
    exit 0
    ;;
merge)
    python3 -c "
import glob, json
recs = []
for f in sorted(glob.glob('$CLAIMS_DIR/done/*.json')):
    d = json.load(open(f))
    recs.extend(d.get('vietocr', d.get('texts', [])))
json.dump(recs, open('$OUTPUT_DIR/output_ocr.json', 'w'), ensure_ascii=False, indent=2)
print(f'{len(recs)} record -> $OUTPUT_DIR/output_ocr.json')
"
    exit 0
    ;;
esac

NSHARDS="${NSHARDS:-8}"
mapfile -t GPUS < <(pick_gpus "$NSHARDS")
if [[ ${#GPUS[@]} -eq 0 ]]; then
    echo "TỪ CHỐI: không GPU nào còn >= ${MIN_FREE_MB} MB trống / util <= ${MAX_UTIL}%." >&2
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader >&2
    exit 1
fi

echo "== Khởi động ${#GPUS[@]} container: GPU ${GPUS[*]} =="
for gpu in "${GPUS[@]}"; do
    run_shard "$gpu"
done
echo "== Đã launch nền. Theo dõi: ./run_shards.sh status, hoặc tail -f $OUTPUT_DIR/shard_*.log"
echo "== Xong hết thì: ./run_shards.sh merge =="
