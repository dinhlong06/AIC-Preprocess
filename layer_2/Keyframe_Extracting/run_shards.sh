#!/usr/bin/env bash
# Chạy layer_2 song song nhiều shard.
#
#   ./run_shards.sh                       # pipeline_g, 3 shard
#   NSHARDS=4 ./run_shards.sh
#   ./run_shards.sh --videos L25_V054.mp4 # pilot vài video
#   ./run_shards.sh add 5 2               # bật thêm shard 5 trên GPU 2 (chạy nền)
#   ./run_shards.sh add 5 ""             # như trên, để run.sh tự chọn GPU
#   ./run_shards.sh drain 5               # dừng ÊM: xong video hiện tại rồi thoát
#   ./run_shards.sh stop 5                # dừng NGAY, giết ngang (mất video đang dở)
#   nohup ./run_shards.sh watchdog &      # tự phát hiện shard treo (NFS hang), đánh lỗi, bật lại
#
# Env (mặc định = batch1; đổi sang batch2 bằng VIDEO_DIR/SHOTS_SRC/OUTPUT_DIR):
#   VIDEO_DIR       thư mục video
#   SHOTS_SRC       shots.jsonl của layer 1
#   SHOTS_SPLIT_DIR nơi tách shots.json thành 1 file/video
#   OUTPUT_DIR      thư mục output (mọi shard dùng chung)
#
# Chế độ 'add' dùng khi có GPU rảnh mới hoặc để thay shard đã chết: nó KHÔNG xoá
# claim, nên an toàn khi các shard khác đang chạy.
#
# Output mỗi video một thư mục con nên mọi shard dùng chung OUTPUT_DIR; chỉ
# benchmark_summary.csv tách theo SHARD_ID.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PIPELINE="${PIPELINE:-pipeline_g}"
NSHARDS="${NSHARDS:-3}"

export VIDEO_DIR="${VIDEO_DIR:-$PROJECT_ROOT/dataset_batch1/videos/video}"
export SHOTS_SRC="${SHOTS_SRC:-$PROJECT_ROOT/layer_1/batch1/shots.jsonl}"
export SHOTS_SPLIT_DIR="${SHOTS_SPLIT_DIR:-$SCRIPT_DIR/dataset/shots_batch1}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/benchmark_batch1}"
CLAIMS="$OUTPUT_DIR/claims_$PIPELINE"

# Tìm container của mình theo TÊN ẢNH ghi trong config, không dùng --filter ancestor
# (phân giải sang image ID hiện tại nên build lại ảnh là mất dấu container đang chạy)
# cũng không dùng --filter label (container khởi động trước khi thêm nhãn thì không có).
docker_ours() {
    for c in $(docker ps -q); do
        [[ "$(docker inspect -f '{{.Config.Image}}' "$c" 2>/dev/null)" == "${IMAGE:-ai26-layer2}" ]] && echo "$c"
    done
}

# Bật thêm 1 shard vào một GPU cụ thể, dùng chung claims/output với các shard đang
# chạy. Đặt trước phần xoá claim vì thêm shard giữa chừng không được đụng vào chúng.
if [[ "${1:-}" == "add" ]]; then
    shard="$2"; gpu="$3"
    mkdir -p "$OUTPUT_DIR"
    log="$OUTPUT_DIR/shard_$shard.log"
    GPU_ID="$gpu" SHARD_ID="_$shard" CLAIMS_DIR="/app/benchmark/claims_$PIPELINE" \
        nohup "$SCRIPT_DIR/run.sh" "$PIPELINE" >> "$log" 2>&1 &
    disown
    echo "shard $shard -> GPU $gpu, log: $log"
    exit 0
fi

# Dừng ÊM: đặt cờ, shard làm nốt video hiện tại rồi tự thoát. Không mất video đang
# dở, không sinh claim mồ côi. Dùng cái này thay 'stop' trừ khi shard đã treo.
if [[ "${1:-}" == "drain" ]]; then
    mkdir -p "$OUTPUT_DIR/$PIPELINE"
    : > "$OUTPUT_DIR/$PIPELINE/_stop_$2"
    echo "đã đặt cờ dừng êm cho shard $2 — nó thoát sau khi xong video đang chạy"
    echo "theo dõi: tail -f $OUTPUT_DIR/shard_$2.log"
    exit 0
fi

# Dừng NGAY, giết ngang: mất video đang làm dở. Chỉ dùng khi shard đã treo hoặc cần
# giải phóng GPU gấp; bình thường dùng 'drain'.
if [[ "${1:-}" == "stop" ]]; then
    shard="_$2"
    # Giết wrapper TRƯỚC container: run.sh có vòng thử lại 3 lần, giết mỗi container
    # thì nó tưởng lỗi GPU và bật shard mới ngay. SHARD_ID nằm trong environment chứ
    # không trong command line, nên phải đọc /proc chứ pkill -f không khớp.
    for p in $(pgrep -f "run.sh $PIPELINE" 2>/dev/null); do
        if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "SHARD_ID=$shard"; then
            echo "dừng wrapper pid $p (shard $shard)"
            kill "$p" 2>/dev/null || true
        fi
    done
    for c in $(docker_ours); do
        if docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$c" | grep -qx "SHARD_ID=$shard"; then
            echo "dừng container $c (shard $shard)"
            docker kill "$c" > /dev/null
        fi
    done
    sleep 3
    n=0
    for f in "$CLAIMS"/*; do
        [[ -f "$f" ]] || continue
        v=$(basename "$f")
        [[ -f "$OUTPUT_DIR/$PIPELINE/$v/statistics.json" ]] && continue
        [[ "$(cat "$f")" == "$shard" ]] && { rm -f "$f"; n=$((n + 1)); }
    done
    echo "đã trả lại $n video cho shard khác nhận"
    exit 0
fi

# Theo dõi shard treo (NFS đọc video bị kẹt vô thời hạn, kiểu D-state không cách
# nào kill được từ bên trong container): log không nhích quá TIMEOUT giây thì coi
# video hiện tại là hỏng, dừng shard, đánh dấu lỗi vĩnh viễn (khớp _MAX_VIDEO_RETRIES
# trong runner.py để lần sau mọi shard tự bỏ qua), rồi bật lại shard trên GPU khác.
#   nohup ./run_shards.sh watchdog &        # mặc định: check mỗi 5', kẹt >20'
#   nohup ./run_shards.sh watchdog 300 900 & # tự chọn interval/timeout (giây)
if [[ "${1:-}" == "watchdog" ]]; then
    INTERVAL="${2:-300}"
    TIMEOUT="${3:-1200}"
    echo "watchdog: check mỗi ${INTERVAL}s, coi là kẹt nếu log đứng yên > ${TIMEOUT}s"
    while true; do
        sleep "$INTERVAL"
        for log in "$OUTPUT_DIR"/shard_*.log; do
            [[ -f "$log" ]] || continue
            shard="$(basename "$log" .log | sed 's/^shard_//')"
            # Shard đã tự thoát (xong việc / bị stop tay) thì log đứng yên là bình
            # thường — chỉ báo kẹt khi container của nó vẫn đang chạy.
            container=""
            for c in $(docker_ours); do
                docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$c" 2>/dev/null \
                    | grep -qx "SHARD_ID=_$shard" && { container="$c"; break; }
            done
            [[ -z "$container" ]] && continue
            age=$(( $(date +%s) - $(stat -c %Y "$log") ))
            (( age < TIMEOUT )) && continue
            video="$(grep -oE 'Processing [^ ]*\.(mp4|mov)' "$log" | tail -1 | awk '{print $2}')"
            [[ -z "$video" ]] && continue
            video_id="${video%.*}"
            [[ -f "$OUTPUT_DIR/$PIPELINE/$video_id/statistics.json" ]] && continue
            echo "watchdog: shard $shard kẹt ${age}s ở $video -> dừng, đánh lỗi vĩnh viễn, bật lại"
            "$0" stop "$shard"
            mkdir -p "$OUTPUT_DIR/$PIPELINE/_failures"
            echo -n 2 > "$OUTPUT_DIR/$PIPELINE/_failures/$video_id"
            "$0" add "$shard" ""
        done
    done
    exit 0
fi

# Tách shots.jsonl một lần ở đây thay vì để 3 shard cùng ghi đè lên nhau.
mkdir -p "$SHOTS_SPLIT_DIR"
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

# Video thiếu file shots sẽ bị _load_shots âm thầm chunk 300 frame với shot_id
# "S0001" (mất prefix video) -> keyframe trông bình thường nhưng sai định danh.
# Thư mục con = một video keyframe BTC cắt sẵn (kf_batch2), xem cli.py.
n_video=$(find "$VIDEO_DIR" -mindepth 1 -maxdepth 1 \( -name '*.mp4' -o -name '*.mov' -o -type d \) | wc -l)
n_shots=$(ls "$SHOTS_SPLIT_DIR" | wc -l)
if [[ "$n_shots" -lt "$n_video" ]]; then
    echo "TỪ CHỐI: mới có shots cho $n_shots/$n_video video."
    echo "Chạy layer_1/run_shards.sh merge trước."
    exit 1
fi

# Script này xoá sạch claim rồi gieo lại, nên chạy nó trong lúc còn shard đang làm
# việc sẽ khiến hai shard cùng ghi vào một thư mục video.
if docker_ours | grep -q .; then
    echo "TỪ CHỐI: còn container layer_2 đang chạy."
    echo "Đợi chúng xong, hoặc dừng chúng trước khi chạy lại script này."
    exit 1
fi

mkdir -p "$CLAIMS"
# Claim chỉ có hiệu lực trong một lần chạy; statistics.json mới là trạng thái hoàn
# thành bền vững. Không xoá thì video đang dở lúc job chết sẽ bị bỏ qua vĩnh viễn.
find "$CLAIMS" -mindepth 1 -maxdepth 1 -type f -delete
# Gieo claim cho video đã xong để shard khác không làm lại.
for d in "$OUTPUT_DIR/$PIPELINE"/*/; do
    [[ -f "$d/statistics.json" ]] && : > "$CLAIMS/$(basename "$d")"
done

# BEiT-3 Large chiếm ~3 GB VRAM. Máy này dùng chung nên phần lớn GPU chỉ còn vài
# trăm MB; lấy top-N mà không lọc là shard cuối chết CUDA OOM ngay lúc load model.
MIN_FREE_MB="${MIN_FREE_MB:-4000}"
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

echo "== $PIPELINE: $n_video video / $NSHARDS shard / GPU ${GPUS[*]} / đã gieo $(ls "$CLAIMS" | wc -l) claim =="

for ((i = 0; i < NSHARDS; i++)); do
    log="$OUTPUT_DIR/shard_$i.log"
    GPU_ID="${GPUS[i]}" SHARD_ID="_$i" CLAIMS_DIR="/app/benchmark/claims_$PIPELINE" \
        "$SCRIPT_DIR/run.sh" "$PIPELINE" "$@" > "$log" 2>&1 &
    echo "  shard $i -> GPU ${GPUS[i]}, log: $log"
done

wait
echo "== $PIPELINE xong: $(ls -d "$OUTPUT_DIR/$PIPELINE"/*/ 2>/dev/null | wc -l) video có output =="
