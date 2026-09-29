#!/usr/bin/env bash
# Chạy SigLIP embedding song song nhiều shard, mỗi shard một GPU riêng, cho
# layer_2/Keyframe_Extracting/benchmark_batch1/pipeline_g.
#
# Khác run_shards_batch1.sh của layer_2: extract_siglip_dataset đã tự resume
# theo .npy có sẵn, và job chỉ tốn vài chục phút nên không cần add/drain/stop —
# chỉ cần chia video ra N thư mục symlink riêng để N container không đụng nhau.
#
#   ./run_siglip_shards_batch1.sh                 # 4 shard
#   NSHARDS=8 ./run_siglip_shards_batch1.sh
#   ./run_siglip_shards_batch1.sh --overwrite --batch-size 64
#
# Input : layer_2/Keyframe_Extracting/benchmark_batch1/pipeline_g/<VIDEO_ID>/*.jpg
# Output: ./artifacts/siglip_batch1/<VIDEO_ID>.npy + <VIDEO_ID>_ids.json — dùng
#         chung OUTPUT_DIR vì mỗi video chỉ thuộc đúng một shard, không tranh ghi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="ai26-siglip"
FRAMES_DIR="${FRAMES_DIR:-$PROJECT_ROOT/layer_2/Keyframe_Extracting/benchmark_batch1/pipeline_g}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/artifacts/siglip_batch1}"
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/.model_cache/huggingface}"
SHARDS_DIR="${SHARDS_DIR:-$SCRIPT_DIR/.shard_frames_batch1}"
NSHARDS="${NSHARDS:-4}"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

# so400m ~1,5 GB VRAM ở fp16, không phải BEiT-3 Large 3 GB, nên ngưỡng thấp hơn
# nhiều so với layer_2 (4000 MB) là đủ an toàn.
MIN_FREE_MB="${MIN_FREE_MB:-1500}"
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', *' -v m="$MIN_FREE_MB" '$2 >= m' | sort -t',' -k2 -n -r \
    | head -"$NSHARDS" | cut -d',' -f1 | tr -d ' ')

if [[ ${#GPUS[@]} -eq 0 ]]; then
    echo "TỪ CHỐI: không GPU nào còn >= ${MIN_FREE_MB} MB trống."
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
    exit 1
fi
if [[ ${#GPUS[@]} -lt $NSHARDS ]]; then
    echo "Chỉ ${#GPUS[@]}/$NSHARDS GPU đủ chỗ -> chạy ${#GPUS[@]} shard."
    NSHARDS=${#GPUS[@]}
fi

# Chỉ lấy video layer_2 đã ghi xong (có statistics.json), để chạy được nhiều vòng song
# song với layer_2; video đã có .npy thì siglip-dataset tự bỏ qua.
mapfile -t VIDEOS < <(find "$FRAMES_DIR" -mindepth 2 -maxdepth 2 -name statistics.json -printf '%h\n' \
    | xargs -rn1 basename | grep -E '^[A-Za-z]+[0-9]+[_-]V[0-9]+$' | sort)
if [[ ${#VIDEOS[@]} -eq 0 ]]; then
    echo "TỪ CHỐI: không tìm thấy thư mục video nào dưới $FRAMES_DIR"
    exit 1
fi

# Chia round-robin theo thứ tự tên (thay vì chunk liên tiếp) để các shard đều
# nhau về prefix/độ dài video, tránh một shard toàn L26 (video dài nhất).
rm -rf "$SHARDS_DIR"
for ((i = 0; i < NSHARDS; i++)); do
    mkdir -p "$SHARDS_DIR/shard_$i"
done
for ((v = 0; v < ${#VIDEOS[@]}; v++)); do
    shard=$((v % NSHARDS))
    # Target là path TRONG container (/data/frames_all, mount riêng bên dưới), không
    # phải path host — symlink tuyệt đối theo host sẽ gãy vì host không mount cùng chỗ.
    ln -s "/data/frames_all/${VIDEOS[v]}" "$SHARDS_DIR/shard_$shard/${VIDEOS[v]}"
done

echo "== Build image $IMAGE_NAME =="
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

echo "== ${#VIDEOS[@]} video / $NSHARDS shard / GPU ${GPUS[*]} =="
for ((i = 0; i < NSHARDS; i++)); do
    n=$(find "$SHARDS_DIR/shard_$i" -mindepth 1 -maxdepth 1 -type l | wc -l)
    log="$OUTPUT_DIR/shard_$i.log"
    echo "  shard $i -> GPU ${GPUS[i]}, $n video, log: $log"
    docker run --rm \
        --gpus "device=${GPUS[i]}" \
        --shm-size=2g \
        -v "$SHARDS_DIR/shard_$i:/data/frames:ro" \
        -v "$FRAMES_DIR:/data/frames_all:ro" \
        -v "$OUTPUT_DIR:/data/output" \
        -v "$CACHE_DIR:/root/.cache/huggingface" \
        -e HF_HOME=/root/.cache/huggingface \
        "$IMAGE_NAME" \
        siglip-dataset \
        --dataset-root /data/frames \
        --output-dir /data/output \
        --device cuda:0 \
        "$@" > "$log" 2>&1 &
done

wait
echo "== xong: $(find "$OUTPUT_DIR" -maxdepth 1 -name '*.npy' | wc -l) file .npy =="
