#!/usr/bin/env bash
# Chạy Layer 1 song song nhiều shard trên nhiều GPU.
#
#   ./run_shards.sh shots          # TransNetV2, nghẽn ở CPU (ffmpeg decode)
#   ./run_shards.sh asr            # PhoWhisper, nghẽn ở GPU
#   NSHARDS=6 ./run_shards.sh shots
#   ./run_shards.sh merge          # gộp jsonl các shard về shots.jsonl / whisper.jsonl
#
# Mỗi shard ghi vào thư mục riêng: nhiều process cùng append một jsonl trên NFS
# sẽ xé dòng giữa chừng. Video chia kiểu round-robin cho đều độ dài.
#
# GPU được cấp phát MỘT LẦN ở đây rồi truyền xuống, không để từng container tự
# chọn — nvidia-smi không kịp phản ánh chỗ vừa bị chiếm, mọi shard sẽ chọn trùng.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAGE="${1:-shots}"
NSHARDS="${NSHARDS:-4}"
SHARD_ROOT="$SCRIPT_DIR/shards"

if [[ "$STAGE" == "merge" ]]; then
    n_total=$(cd "$PROJECT_ROOT/dataset/video" && ls *.mp4 | wc -l)
    for name in shots whisper; do
        out="$SCRIPT_DIR/$name.jsonl"
        found=("$SHARD_ROOT"/*/"$name.jsonl")
        [[ -e "${found[0]}" ]] || continue
        n_done=$(cat "$SHARD_ROOT"/*/"$name.jsonl.done" 2>/dev/null | sort -u | wc -l)
        # Gộp khi còn thiếu video sẽ tạo ra file trông bình thường nhưng khuyết dữ
        # liệu, không báo gì. FORCE=1 nếu thật sự muốn gộp bản dở.
        if [[ "$n_done" -lt "$n_total" && "${FORCE:-0}" != "1" ]]; then
            echo "$name.jsonl: TỪ CHỐI gộp — mới $n_done/$n_total video. FORCE=1 để ghi đè."
            continue
        fi
        cat "${found[@]}" > "$out.tmp" && mv "$out.tmp" "$out"
        cat "$SHARD_ROOT"/*/"$name.jsonl.done" 2>/dev/null | sort -u > "$out.done"
        echo "$name.jsonl: $(wc -l < "$out") dòng, $n_done/$n_total video"
    done
    exit 0
fi

mapfile -t VIDEOS < <(cd "$PROJECT_ROOT/dataset/video" && ls *.mp4 | sort)
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -"$NSHARDS" | cut -d',' -f1 | tr -d ' ')

# Mọi shard duyệt CHUNG một danh sách và giành việc qua thư mục claim, thay vì
# chia cứng từ đầu: shard nào xong sớm thì lấy tiếp, không ai nằm chơi.
CLAIMS="$SHARD_ROOT/claims_$STAGE"
NAME=$( [[ "$STAGE" == "shots" ]] && echo shots || echo whisper )
mkdir -p "$CLAIMS"
# Claim chỉ có hiệu lực trong một lần chạy; .done mới là trạng thái hoàn thành bền vững.
find "$CLAIMS" -mindepth 1 -maxdepth 1 -type f -delete
# Gieo claim cho video đã xong ở BẤT KỲ shard nào. Mỗi shard chỉ đọc .done của
# riêng nó, nên thiếu bước này là shard khác làm lại và ghi trùng vào jsonl.
cat "$SHARD_ROOT/${STAGE}"_*/"$NAME.jsonl.done" 2>/dev/null | sort -u \
    | while read -r v; do [[ -n "$v" ]] && : > "$CLAIMS/$v"; done

echo "== $STAGE: ${#VIDEOS[@]} video / $NSHARDS shard / GPU ${GPUS[*]} / đã gieo $(ls "$CLAIMS" | wc -l) claim =="

for ((i = 0; i < NSHARDS; i++)); do
    out="$SHARD_ROOT/${STAGE}_$i"
    mkdir -p "$out"
    # Backend ASR: mặc định chunkformer, truyền xuống cho khỏi phải set tay.
    OUTPUT_DIR="$out" GPU_ID="${GPUS[i]}" CLAIMS_DIR="$CLAIMS" \
        ASR_BACKEND="${ASR_BACKEND:-chunkformer}" \
        "$SCRIPT_DIR/run_layer1.sh" \
            --claims_dir /data/claims \
            "$( [[ "$STAGE" == "shots" ]] && echo --skip_asr || echo --skip_shots )" \
            "${@:2}" \
        > "$out/log.txt" 2>&1 &
    echo "  shard $i -> GPU ${GPUS[i]}, log: $out/log.txt"
done

wait
echo "== $STAGE xong. Chạy './run_shards.sh merge' để gộp. =="
