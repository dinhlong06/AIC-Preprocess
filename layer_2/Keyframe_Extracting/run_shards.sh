#!/usr/bin/env bash
# Chạy layer_2 song song nhiều shard.
#
#   ./run_shards.sh                    # 3 shard, pipeline_c
#   NSHARDS=4 ./run_shards.sh
#   DAKE_THREADS=4 NSHARDS=3 ./run_shards.sh
#
# Khác layer_1: output là MỘT THƯ MỤC CON MỖI VIDEO nên mọi shard dùng chung
# --output_dir, không cần tách rồi merge. Chỉ benchmark_summary.csv là điểm ghi
# chung, tách theo SHARD_ID. Phân việc bằng thư mục claim (O_EXCL, NFSv4).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PIPELINE="${PIPELINE:-pipeline_g}"
NSHARDS="${NSHARDS:-3}"
CLAIMS="$SCRIPT_DIR/benchmark/claims_$PIPELINE"
SHOTS_SPLIT_DIR="$SCRIPT_DIR/dataset/shots"

# Video thiếu file shots sẽ bị _load_shots âm thầm chunk 300 frame với shot_id
# "S0001" (mất prefix video) -> keyframe trông bình thường nhưng sai định danh.
# Chặn ở đây thay vì sửa _load_shots, vì fallback đó là cố ý cho pipeline khác.
n_video=$(cd "$PROJECT_ROOT/dataset/video" && ls *.mp4 | wc -l)
n_shots=$(ls "$SHOTS_SPLIT_DIR" 2>/dev/null | wc -l)
if [[ "$n_shots" -lt "$n_video" ]]; then
    echo "TỪ CHỐI: mới có shots cho $n_shots/$n_video video."
    echo "Chạy layer_1/run_shards.sh merge trước, rồi ./run.sh một lần để tách shots."
    exit 1
fi

[[ ! -d "$CLAIMS" ]] && { rm -rf "$CLAIMS" 2>/dev/null || true; mkdir -p "$CLAIMS"; }
# Gieo claim cho video đã xong để shard khác không làm lại.
for d in "$SCRIPT_DIR/benchmark/$PIPELINE"/*/; do
    [[ -f "$d/statistics.json" ]] && : > "$CLAIMS/$(basename "$d")"
done

mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -"$NSHARDS" | cut -d',' -f1 | tr -d ' ')

echo "== $PIPELINE: $n_video video / $NSHARDS shard / GPU ${GPUS[*]} / đã gieo $(ls "$CLAIMS" | wc -l) claim =="

for ((i = 0; i < NSHARDS; i++)); do
    log="$SCRIPT_DIR/benchmark/shard_$i.log"
    GPU_ID="${GPUS[i]}" SHARD_ID="_$i" CLAIMS_DIR="/app/benchmark/claims_$PIPELINE" \
        "$SCRIPT_DIR/run.sh" "$PIPELINE" "$@" > "$log" 2>&1 &
    echo "  shard $i -> GPU ${GPUS[i]}, log: $log"
done

wait
echo "== $PIPELINE xong: $(ls -d "$SCRIPT_DIR/benchmark/$PIPELINE"/*/ 2>/dev/null | wc -l) video có output =="
