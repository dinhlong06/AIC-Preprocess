#!/usr/bin/env bash
# Build + chạy SigLIP embedding của keyframe_pipeline trong Docker, chia shard
# song song — mỗi shard một GPU riêng.
#
# Cách dùng:
#   ./run.sh                          # 4 shard song song (mặc định)
#   NSHARDS=1 ./run.sh                # 1 GPU, 1 process
#   NSHARDS=8 ./run.sh --overwrite
#   FRAMES_DIR=/path OUTPUT_DIR=/path ./run.sh
#
# Env:
#   FRAMES_DIR   nguồn keyframe (mặc định benchmark_batch1/pipeline_g)
#   OUTPUT_DIR   nơi ghi .npy + _ids.json (mặc định artifacts/siglip_batch1_v2)
#   NSHARDS      số shard = số GPU dùng song song (mặc định 4)
#   MIN_FREE_MB  VRAM trống tối thiểu để chọn GPU (mặc định 1500)
#
# .model_cache/huggingface giữ weight SigLIP (~3,5 GB) ngoài image, bind-mount
# vào container. Lần chạy đầu tải model, các lần sau dùng lại.
# Video đã có .npy thì siglip-dataset tự bỏ qua, nên chạy lại là resume.
#
# extract_siglip_dataset tự resume theo .npy có sẵn nên chỉ cần chia video ra
# N thư mục symlink để N container không đụng nhau. Chia round-robin theo thứ
# tự tên (thay vì chunk liên tiếp) để các shard đều nhau, tránh một shard toàn
# L26 (video dài nhất).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="ai26-siglip"
FRAMES_DIR="${FRAMES_DIR:-$PROJECT_ROOT/layer_2/Keyframe_Extracting/benchmark_batch1/pipeline_g}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/artifacts/siglip_batch1_v2}"
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/.model_cache/huggingface}"
SHARDS_DIR="${SHARDS_DIR:-$SCRIPT_DIR/.shard_frames}"
NSHARDS="${NSHARDS:-4}"
# so400m fp16 ~1,5 GB VRAM, ngưỡng 1500 MB là đủ an toàn.
MIN_FREE_MB="${MIN_FREE_MB:-1500}"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

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

mapfile -t VIDEOS < <(find "$FRAMES_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | grep -E '^[A-Za-z]+[0-9]+[_-]V[0-9]+$' | sort)
if [[ ${#VIDEOS[@]} -eq 0 ]]; then
    echo "TỪ CHỐI: không tìm thấy thư mục video nào dưới $FRAMES_DIR"
    exit 1
fi

rm -rf "$SHARDS_DIR"
for ((i = 0; i < NSHARDS; i++)); do
    mkdir -p "$SHARDS_DIR/shard_$i"
done
for ((v = 0; v < ${#VIDEOS[@]}; v++)); do
    # Target là path TRONG container (/data/frames_all, mount riêng bên dưới),
    # không phải path host — symlink tuyệt đối theo host sẽ gãy.
    ln -s "/data/frames_all/${VIDEOS[v]}" "$SHARDS_DIR/shard_$((v % NSHARDS))/${VIDEOS[v]}"
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
        --ipc=host \
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
