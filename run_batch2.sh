#!/usr/bin/env bash
# Batch2: tái dùng keyframe + shot BTC, KHÔNG chạy lại shot detection.
# Mọi output nằm trong output_batch2/.
#
#   ./run_batch2.sh all        # shots rồi driver
#   ./run_batch2.sh driver     # nền (sống sót khi đóng session): mỗi 5' thêm shard khi GPU trống,
#                              # chạy vòng OCR/SigLIP/Gemma trên video layer_2 đã xong, gộp kết quả
#   ./run_batch2.sh status     # tiến độ từng bước
#   ./run_batch2.sh <bước>     # chạy riêng 1 bước (foreground): shots|asr|filter|ocr|gemma_ocr|gemma_ocr_n|siglip
#
# Mọi bước chạy chồng: asr || filter || (ocr, gemma_ocr, siglip trên video filter đã xong)
#   shots     : kf_batch2/*/metadata.json -> shots.jsonl, shot_id = <video>_<shot BTC 6 số>
#   asr       : layer_1 chunkformer greedy (--skip_shots) -> whisper.jsonl -> shot_transcripts.jsonl
#   filter    : layer_2 pipeline_h trên webp BTC -> keyframes/pipeline_h/<video>/<video>_<shot>_kfNNNN.jpg
#   ocr       : layer_3 PaddleOCR shards + local correction -> ocr/output_hybrid.json
#   gemma_ocr : OCR qua API Gemma UIT -> gemma_ocr.jsonl (song song Paddle, ingest nhận cả hai), trừ video N*
#   gemma_ocr_n: video N* (camera giao thông) chỉ OCR frame đầu -> gemma_ocr_n.jsonl, frame_id = video_id
#   siglip    : layer 4 SigLIP2 -> siglip/<video>.npy
#   ingest    : (nền) đợi filter + SigLIP + Gemma OCR xong -> gộp ASR, layer_5 ingest-batch2, thumbnail UI
#   proxy     : bản phát 720p H.264 faststart cho M/N (GPU) -> video_720/<video>.mp4, backend ưu tiên khi phát

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/output_batch2"
# Đọc thẳng nguồn thay vì kf_batch2: kf_batch2 là bản rclone chép dở từ chính thư mục
# này (~9 file/s, hàng chục giờ), thư mục chưa chép tới thì rỗng.
KF_SRC="${KF_SRC:-$(readlink -f "$ROOT/kf_batch2_src")}"
# Không dùng video_batch2 (symlink /mlcv1): NFS đó treo hẳn, ffmpeg đứng ở trạng thái D
# hàng giờ. Bản tải về workspace đọc được; video chưa tải xong thì chạy lại 'asr' sau
# (resume theo .done, không làm lại video đã xong).
VIDEO_SRC="$ROOT/video_batch2_dl"
FRAMES="$OUT/keyframes/pipeline_h"
mkdir -p "$OUT/logs"

step_shots() {
    python3 - "$KF_SRC" "$OUT/shots.jsonl" <<'PYEOF'
import collections, json, os, sys

src, out = sys.argv[1], sys.argv[2]
meta = json.load(open(os.path.join(src, "metadata.json"), encoding="utf-8"))
with open(out + ".tmp", "w", encoding="utf-8") as f:
    for vid in sorted(meta):
        frames = meta[vid].values()
        by_shot = collections.defaultdict(list)
        for fr in frames:
            by_shot[fr["shot"]].append(fr["id"])
        shots = sorted((min(ids), max(ids), s) for s, ids in by_shot.items())
        fps = next(iter(frames))["fps"]
        # Kéo end_frame tới sát shot sau: BTC chỉ cắt mỗi >= 5 frame, để nguyên max(id)
        # thì lời nói rơi vào khe giữa hai shot bị mất khi map transcript.
        for i, (start, last, s) in enumerate(shots):
            end = shots[i + 1][0] - 1 if i + 1 < len(shots) else last + 4
            f.write(json.dumps({"video_id": vid, "shot_id": f"{vid}_{s:06d}",
                                "start_frame": start, "end_frame": end, "fps": fps}) + "\n")
os.replace(out + ".tmp", out)
PYEOF
    echo "shots.jsonl: $(wc -l < "$OUT/shots.jsonl") shot / $(cut -d'"' -f4 "$OUT/shots.jsonl" | sort -u | wc -l) video"
}

step_asr() {
    export BATCH_DIR="$OUT/asr" VIDEO_DIR="$VIDEO_SRC"
    NSHARDS="${ASR_SHARDS:-2}" "$ROOT/layer_1/run_shards_batch1.sh" asr
    FORCE=1 "$ROOT/layer_1/run_shards_batch1.sh" merge
    python3 "$ROOT/layer_2/shot_transcript/cpu_map_transcript.py" \
        --shots_path "$OUT/shots.jsonl" \
        --whisper_path "$OUT/asr/whisper.jsonl" \
        --output_path "$OUT/shot_transcripts.jsonl"
}

# Đủ cả 5 biến: thiếu SHOTS_* thì run_shards_batch1.sh rơi về shots của batch1.
filter_env() {
    VIDEO_DIR="$KF_SRC" SHOTS_SRC="$OUT/shots.jsonl" SHOTS_SPLIT_DIR="$OUT/shots_split" \
        OUTPUT_DIR="$OUT/keyframes" PIPELINE=pipeline_h SKIP_BUILD=1 "$@"
}

step_filter() {
    NSHARDS="${FILTER_SHARDS:-3}" filter_env "$ROOT/layer_2/Keyframe_Extracting/run_shards_batch1.sh" "$@"
}

# Đếm từ shots.jsonl chứ không find trên NAS nguồn: NAS bị layer_2 đọc nặng, find treo vài phút.
n_videos() { cut -d'"' -f4 "$OUT/shots.jsonl" | sort -u | wc -l; }

step_ocr() {
    local paddle="$ROOT/layer_3/OCR/run_paddle_batch1_shards.sh" total
    export FRAMES_DIR="$FRAMES" OUTPUT_DIR="$OUT/ocr"
    NSHARDS="${OCR_SHARDS:-3}" "$paddle"
    total=$(find "$FRAMES" -mindepth 1 -maxdepth 1 -type d -name '*_V*' | wc -l)
    # Container Paddle chạy nền (disown): đợi đủ file done thay vì wait.
    until [[ $(ls "$OUT/ocr/claims/done" | wc -l) -ge $total ]]; do
        sleep 120
        docker ps --format '{{.Image}}' | grep -qx ocr-paddle || { echo "container Paddle đã thoát hết nhưng mới $(ls "$OUT/ocr/claims/done" | wc -l)/$total video" >&2; break; }
    done
    "$paddle" merge
    (cd "$ROOT/layer_3/OCR" && python3 run_correct.py --frames "$FRAMES" \
        --paddle-output "$OUT/ocr/output_vietocr.json" --output "$OUT/ocr/output_hybrid.json")
}

# OCR chỉ dùng Gemma: UIT (32 luồng, trần 40 request đồng thời/key) chạy từ đầu danh sách,
# AI Studio 26B chạy từ cuối. AI Studio có trần CHUNG ~60 frame/phút cho mọi key (thêm key còn
# chậm đi) nên đúng 3 process k7/k9/k10. Hai bên --skip frame của nhau nên không làm trùng.
gemma_loop() {
    local last=0
    while true; do
        # Layer_2 đã xong từ trước lượt này thì lượt này đã thấy mọi frame: thoát sau khi chạy xong.
        (( $(count_filter) >= $(n_videos) )) && last=1
        python3 -u "$ROOT/layer_3/OCR_gemma/gemma_ocr.py" --task ocr --frames "$FRAMES" --scan "$OUT/scan" "$@" || true
        (( last )) && return
        sleep 120
    done
}

step_gemma_ocr() {
    set -a; . "$ROOT/retrieval_system/.env"; set +a
    local i=0 k
    for k in 7 9 10; do
        gemma_loop --model ais26 --key "$k" --reverse --shard "$i/3" --out "$OUT/gemma_ocr_ais_k$k.jsonl" \
            --skip "$OUT/gemma_ocr.jsonl" "$OUT/gemma_ocr_ais_*.jsonl" >> "$OUT/logs/gemma_ais_k$k.log" 2>&1 &
        i=$((i + 1))
    done
    step_gemma_ocr_n
    GEMMA_WORKERS=32 gemma_loop --out "$OUT/gemma_ocr.jsonl" --skip "$OUT/gemma_ocr_ais_*.jsonl"
    wait
}

# Camera giao thông đứng yên: chữ chỉ là overlay tên giao lộ + đồng hồ (và biển hiệu cố định) nên frame
# 0 của webp BTC là đủ, không phải đợi layer_2. Ảnh đặt tên <video>.jpg để frame_id = video_id.
step_gemma_ocr_n() {
    set -a; . "$ROOT/retrieval_system/.env"; set +a
    python3 - "$KF_SRC" "$OUT/n_first" <<'PYEOF'
import os, sys
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for v in sorted(d for d in os.listdir(src) if d.startswith("N")):
    if not os.path.exists(f"{dst}/{v}.jpg"):
        Image.open(f"{src}/{v}/frame_001.webp").convert("RGB").save(f"{dst}/{v}.jpg", quality=95)
PYEOF
    # 4 luồng: job chính giữ 32/40 request đồng thời của key UIT.
    GEMMA_WORKERS=4 python3 -u "$ROOT/layer_3/OCR_gemma/gemma_ocr.py" --task ocr --images "$OUT/n_first/*.jpg" --out "$OUT/gemma_ocr_n.jsonl"
}

# Một số file trên NFS workspace đọc bị treo VĨNH VIỄN (trạng thái D, kill không được): SigLIP
# gặp là cả container kẹt và giữ ~3 GB VRAM mãi. Quét trước trên host, mỗi file tối đa 30s;
# file treo ghi vào scan/<video>.bad, xong ghi scan/<video>.ok. Luồng treo bị bỏ lại trong
# process quét này (không GPU, không chặn gì); đọc trước còn nạp ảnh vào page cache.
step_scan() {
    python3 - "$FRAMES" "$OUT/scan" <<'PYEOF'
import os, sys, time
from concurrent.futures import ThreadPoolExecutor, wait
frames, scan = sys.argv[1], sys.argv[2]
os.makedirs(scan, exist_ok=True)
pool = ThreadPoolExecutor(64)
while True:
    todo = sorted(v for v in os.listdir(frames) if os.path.exists(f"{frames}/{v}/statistics.json")
                  and not os.path.exists(f"{scan}/{v}.ok"))
    for v in todo:
        futs = {pool.submit(lambda p: open(p, "rb").read(), f"{frames}/{v}/{f}"): f
                for f in os.listdir(f"{frames}/{v}") if f.endswith(".jpg")}
        _, stuck = wait(futs, timeout=30 + len(futs) * 0.05)
        bad = sorted(futs[f] for f in stuck)
        with open(f"{scan}/{v}.bad", "w") as f:
            f.write("".join(b + "\n" for b in bad))
        open(f"{scan}/{v}.ok", "w").close()
        print(f"{time.strftime('%H:%M:%S')} scan {v}: {len(futs)} ảnh, {len(bad)} treo {bad}", flush=True)
    time.sleep(60)
PYEOF
}

# SigLIP chỉ nhận video đã quét: mỗi video là thư mục thật trong /dev/shm chứa symlink tới từng
# ảnh trừ ảnh treo (discovery.py liệt kê file trong thư mục đó). Model nạp từ RAM, ngưỡng VRAM
# 3,5 GB vì SigLIP thực dùng ~3,3 GB.
step_siglip() {
    local sh=/dev/shm/ai26_siglip_shards n="${SIGLIP_SHARDS:-2}" gpus i v f
    mapfile -t gpus < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', *' '$2 >= 3500' | sort -t',' -k2 -n -r | head -"$n" | cut -d',' -f1)
    (( ${#gpus[@]} )) || { echo "TỪ CHỐI: không GPU nào còn >= 3500 MB trống."; return 0; }
    rm -rf "$sh"; i=0
    for f in "$OUT"/scan/*.ok; do
        v=$(basename "$f" .ok); [[ -f "$OUT/siglip/$v.npy" ]] && continue
        mkdir -p "$sh/shard_$((i % ${#gpus[@]}))/$v"
        for f in "$FRAMES/$v"/*.jpg; do
            grep -qxF "$(basename "$f")" "$OUT/scan/$v.bad" || ln -s "/data/frames_all/$v/$(basename "$f")" "$sh/shard_$((i % ${#gpus[@]}))/$v/"
        done
        i=$((i + 1))
    done
    echo "$(date +%T) siglip: $i video / GPU ${gpus[*]}"
    for i in "${!gpus[@]}"; do
        [[ -d "$sh/shard_$i" ]] || continue
        docker run --rm --gpus "device=${gpus[i]}" --shm-size=2g \
            -v "$sh/shard_$i:/data/frames:ro" -v "$FRAMES:/data/frames_all:ro" -v "$OUT/siglip:/data/output" \
            -v /dev/shm/ai26_siglip_cache:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface \
            ai26-siglip siglip-dataset --dataset-root /data/frames --output-dir /data/output --device cuda:0 \
            > "$OUT/siglip/shard_$i.log" 2>&1 &
    done
    wait
}

n_running() { for c in $(docker ps -q); do docker inspect -f '{{.Config.Image}}' "$c"; done | grep -cx "$1" || true; }
free_gpu() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', *' -v m="$1" '$2 >= m' | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1
}
# "|| true": chưa có file nào thì ls lỗi, pipefail + set -e sẽ giết cả driver.
count_asr() { { cat "$OUT"/asr/shards/*/whisper.jsonl.done 2>/dev/null || true; } | sort -u | wc -l; }
count_filter() { { ls "$FRAMES"/*/statistics.json 2>/dev/null || true; } | wc -l; }
count_paddle() { { ls "$OUT/ocr/claims/done" 2>/dev/null || true; } | wc -l; }
# PID rỗng = chưa chạy; không dùng mặc định 0 vì "kill -0 0" hỏi cả process group, luôn đúng.
alive() { [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null; }
count_siglip() { { ls "$OUT"/siglip/*.npy 2>/dev/null || true; } | wc -l; }
count_scan() { { ls "$OUT"/scan/*.ok 2>/dev/null || true; } | wc -l; }

add_asr_shard() {
    local i; i=$(ls -d "$OUT"/asr/shards/asr_* 2>/dev/null | wc -l)
    mkdir -p "$OUT/asr/shards/asr_$i"
    echo "$(date +%T) driver: thêm ASR shard $i -> GPU $1"
    OUTPUT_DIR="$OUT/asr/shards/asr_$i" GPU_ID="$1" CLAIMS_DIR="$OUT/asr/shards/claims_asr" \
        VIDEO_DIR="$VIDEO_SRC" ASR_BACKEND=chunkformer \
        "$ROOT/layer_1/run_layer1_batch1.sh" --claims_dir /data/claims --skip_shots \
        > "$OUT/asr/shards/asr_$i/log.txt" 2>&1 &
}

add_filter_shard() {
    local i claims="$OUT/keyframes/claims_pipeline_h" v; i=$(ls "$OUT"/keyframes/shard_*.log 2>/dev/null | wc -l)
    # Không còn shard nào sống mà vẫn có claim của video chưa xong = claim mồ côi (shard chết
    # giữa chừng); không nhả thì video đó bị bỏ qua vĩnh viễn.
    if (( $(n_running ai26-layer2) == 0 )); then
        for v in $(ls "$claims" 2>/dev/null); do [[ -f "$FRAMES/$v/statistics.json" ]] || rm -f "$claims/$v"; done
    fi
    echo "$(date +%T) driver: thêm layer_2 shard $i -> GPU $1"
    filter_env "$ROOT/layer_2/Keyframe_Extracting/run_shards_batch1.sh" add "$i" "$1"
}

asr_finish() {
    BATCH_DIR="$OUT/asr" VIDEO_DIR="$VIDEO_SRC" FORCE=1 "$ROOT/layer_1/run_shards_batch1.sh" merge
    python3 "$ROOT/layer_2/shot_transcript/cpu_map_transcript.py" --shots_path "$OUT/shots.jsonl" \
        --whisper_path "$OUT/asr/whisper.jsonl" --output_path "$OUT/shot_transcripts.jsonl"
}

ocr_finish() {
    FRAMES_DIR="$FRAMES" OUTPUT_DIR="$OUT/ocr" "$ROOT/layer_3/OCR/run_paddle_batch1_shards.sh" merge
    (cd "$ROOT/layer_3/OCR" && python3 run_correct.py --frames "$FRAMES" \
        --paddle-output "$OUT/ocr/output_vietocr.json" --output "$OUT/ocr/output_hybrid.json")
}

# Một vòng điều phối mọi bước, chạy chồng lên nhau để kịp giờ: OCR/SigLIP/Gemma chạy từng
# vòng trên video layer_2 đã xong (có statistics.json), không chờ layer_2 xong toàn bộ.
# Layer_2 là bước dài nhất nên được thêm shard trước ASR khi có GPU trống.
step_driver() {
    local total asr_total gpu gemma_pid="" siglip_pid="" scan_pid="" asr_merged=-1 f
    total=$(n_videos)
    while true; do
        asr_total=$(find "$VIDEO_SRC" -maxdepth 1 \( -name '*.mp4' -o -name '*.mov' \) | wc -l)
        f=$(count_filter)
        echo "$(date +%T) driver: asr $(count_asr)/$asr_total filter $f/$total paddle $(count_paddle) siglip $(count_siglip)"

        if (( f < total )) && (( $(n_running ai26-layer2) < ${FILTER_MAX:-6} )) && gpu=$(free_gpu 4500) && [[ -n "$gpu" ]]; then
            add_filter_shard "$gpu"
        elif (( $(count_asr) < asr_total )) && (( $(n_running ai26-layer1) < ${ASR_MAX:-3} )) && gpu=$(free_gpu 4000) && [[ -n "$gpu" ]]; then
            add_asr_shard "$gpu"
        fi
        if (( $(count_asr) != asr_merged )) && (( $(n_running ai26-layer1) == 0 )); then
            asr_merged=$(count_asr); asr_finish >> "$OUT/logs/asr.log" 2>&1 || true
        fi

        if ! alive "$scan_pid"; then
            step_scan >> "$OUT/logs/scan.log" 2>&1 & scan_pid=$!
        fi
        if (( $(count_siglip) < $(count_scan) )) && ! alive "$siglip_pid"; then
            step_siglip >> "$OUT/logs/siglip.log" 2>&1 & siglip_pid=$!
        fi
        if (( f > 0 )) && ! alive "$gemma_pid"; then
            step_gemma_ocr >> "$OUT/logs/gemma_ocr.log" 2>&1 & gemma_pid=$!
        fi

        # OCR chỉ dùng Gemma (API, không GPU): bỏ Paddle để dồn GPU cho layer_2/ASR/SigLIP.
        if (( f >= total && $(count_siglip) >= total )) && ! alive "$siglip_pid"; then
            echo "$(date +%T) driver: layer_2 + SigLIP xong"; step_status
            (( $(count_asr) >= asr_total )) && return
        fi
        sleep 300
    done
}

# Chỉ CPU/I-O, không GPU: đọc trước webp của các video layer_2 sắp nhận (cùng thứ tự sort)
# để chúng nằm sẵn trong page cache; shard đọc từ RAM thay vì chờ NAS (~17 file/s mỗi shard,
# trong khi NAS chịu ~70 file/s). Chỉ đi trước tối đa WARM_AHEAD video để không ôm RAM.
step_warm() {
    python3 - "$KF_SRC" "$FRAMES" "${WARM_AHEAD:-30}" <<'PYEOF'
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
src, frames, ahead = sys.argv[1], sys.argv[2], int(sys.argv[3])
done = lambda v: os.path.exists(f"{frames}/{v}/statistics.json")
videos = sorted(d for d in os.listdir(src) if os.path.isdir(f"{src}/{d}"))
with ThreadPoolExecutor(256) as pool:
    for i, v in enumerate(videos):
        while sum(not done(x) for x in videos[:i]) > ahead:
            time.sleep(20)
        if done(v):
            continue
        t = time.time()
        n = sum(1 for _ in pool.map(lambda f: open(f"{src}/{v}/{f}", "rb").read(), os.listdir(f"{src}/{v}")))
        print(f"{time.strftime('%H:%M:%S')} warm {v}: {n} file {time.time() - t:.0f}s", flush=True)
PYEOF
}

# Không đợi đủ ASR: vài video ASR chết giữa chừng có thể không bao giờ xong, ingest lại (idempotent) khi có.
# Gemma coi là xong khi còn <= 50 frame chưa có: frame lỗi 5 lần bị bỏ qua, lượt sau lại thử và lại lỗi.
step_ingest() {
    local left
    while true; do
        left=$(python3 - "$FRAMES" "$OUT" <<'PYEOF'
import glob, json, sys
from pathlib import Path
frames, out = Path(sys.argv[1]), Path(sys.argv[2])
done = {json.loads(l)["frame_id"] for f in glob.glob(f"{out}/gemma_ocr*.jsonl") for l in open(f, encoding="utf-8")}
print(sum(f.stem not in done for ok in (out / "scan").glob("*.ok") if not ok.stem.startswith("N")
          for bad in [set(ok.with_suffix(".bad").read_text().split())]
          for f in (frames / ok.stem).glob("*.jpg") if f.name not in bad))
PYEOF
)
        echo "$(date +%T) ingest: filter $(count_filter)/$(n_videos) scan $(count_scan) siglip $(count_siglip), gemma còn $left frame"
        (( $(count_filter) >= $(n_videos) && $(count_scan) >= $(n_videos) && $(count_siglip) >= $(n_videos) && left <= 50 )) && break
        sleep 600
    done
    # Không tự asr_finish/make_thumbs: NFS sát quota thì file lớn ghi ra toàn byte 0 mà không báo lỗi
    # (2026-09-25). Gộp ASR tay rồi kiểm; thumbnail chạy tay khi có quota, thiếu thì backend tự resize.
    "$ROOT/layer_5/run.sh" ingest-batch2
    echo "$(date +%T) ingest: xong"
}

# Bản phát cho UI: M/N gốc 1080p 2,5-5 Mbps, header moov ở cuối file nên trình duyệt tua chậm
# hơn batch1 (720p ~1,7 Mbps). Giải mã + thu nhỏ + mã hoá đều trên GPU (cuvid/nvenc, <1 nhân CPU),
# fps giữ nguyên nên frame_idx không lệch. S* là AV1, ffmpeg host (4.2) không giải mã được trên GPU:
# giải mã bằng dav1d trong ai26-layer1 (ffmpeg 7), ép CFR đúng fps gốc, pipe yuv sang nvenc host.
# Ghi .tmp.mp4 rồi mới đổi tên khi độ dài khớp gốc: NFS hết quota thì file ra toàn byte 0 mà ffmpeg
# không báo lỗi. GeForce chỉ cho 8 phiên NVENC CẢ MÁY và -gpu của ffmpeg 4.2 không có tác dụng (mọi
# phiên dồn về GPU 0), nên mỗi worker ghim một GPU bằng CUDA_VISIBLE_DEVICES, tổng 8 worker.
# Worker nhận việc bằng mkdir .claim/<video> (không tự xoá .claim: chạy thêm worker giữa chừng thì
# không giành video đang làm; claim mồ côi của lần bị kill thì xoá tay). Worker AV1 làm S* dài nhất
# trước, worker M/N chỉ làm M*/N* để tổng CPU ≤ ~10 nhân (AV1 ~4,5 nhân/worker, M/N GPU ~0,2).
proxy_one() {
    local v=$1 gpu=$2 dst="$OUT/video_720" src="$VIDEO_SRC/$1" out a b codec
    out="$dst/${v%.*}.mp4"; codec=$(proxy_codec "$v")
    # N011/N023/N078 là HEVC dù cùng họ N*: bộ giải mã GPU chọn theo codec (h264_cuvid/hevc_cuvid).
    if [[ $codec == av1 ]]; then
        local fps fifo="/dev/shm/proxy_fifo/${v%.*}.yuv"
        fps=$(ffprobe -v error -select_streams v -show_entries stream=r_frame_rate -of csv=p=0 "$src")
        # yuv đi qua FIFO chứ không qua stdout của docker: pipe của Docker rootless chỉ ~37 MB/s (~27 fps
        # 720p), FIFO thì dav1d chạy ~225 fps với 5 thread. Rootless không có cgroup CPU (--cpus bị từ
        # chối) nên giới hạn CPU bằng số thread của dav1d.
        mkdir -p /dev/shm/proxy_fifo; rm -f "$fifo"; mkfifo "$fifo"
        docker run --rm --network none -v "$VIDEO_SRC:/v:ro" -v /dev/shm/proxy_fifo:/p --entrypoint ffmpeg \
            ai26-layer1 -hide_banner -loglevel error -threads "${AV1_THREADS:-4}" -c:v libdav1d -i "/v/$v" \
            -map 0:v:0 -vf scale=1280:720:flags=fast_bilinear -fps_mode cfr -r "$fps" -pix_fmt yuv420p \
            -y -f rawvideo "/p/${fifo##*/}" &
        local enc=(-c:v h264_nvenc -preset slow -rc vbr_hq)
        [[ ${PROXY_ENC:-} == x264 ]] && enc=(-c:v libx264 -preset veryfast -threads "${X264_THREADS:-4}")
        CUDA_VISIBLE_DEVICES=$gpu ffmpeg -hide_banner -loglevel error -y -f rawvideo -pix_fmt yuv420p \
            -s 1280x720 -r "$fps" -i "$fifo" -i "$src" -map 0:v -map '1:a?' "${enc[@]}" \
            -b:v 1200k -maxrate 2000k -bufsize 4000k -c:a copy -movflags +faststart "$out.tmp.mp4" || true
        wait $! || true; rm -f "$fifo"
    elif [[ ${PROXY_ENC:-} == x264 ]]; then
        # Worker phụ khi 8 phiên NVENC đã dùng hết: NVDEC không giới hạn phiên nên vẫn giải mã + thu nhỏ trên
        # GPU, chỉ mã hoá bằng x264 (~3 nhân cho ~3,8x thời gian thực, bằng ~1/4 một worker NVENC).
        CUDA_VISIBLE_DEVICES=$gpu ffmpeg -hide_banner -loglevel error -y -c:v "${codec}_cuvid" -resize 1280x720 -i "$src" \
            -c:v libx264 -preset veryfast -threads "${X264_THREADS:-4}" -b:v 1200k -maxrate 2000k -bufsize 4000k \
            -c:a copy -movflags +faststart "$out.tmp.mp4" 2> >(grep -v 'SEI type' >&2) || true
    else
        CUDA_VISIBLE_DEVICES=$gpu nice -n 10 ffmpeg -hide_banner -loglevel error -y -hwaccel cuvid -c:v "${codec}_cuvid" \
            -resize 1280x720 -i "$src" -c:v h264_nvenc -preset slow -rc vbr_hq \
            -b:v 1200k -maxrate 2000k -bufsize 4000k -c:a copy -movflags +faststart "$out.tmp.mp4" \
            2> >(grep -v 'SEI type' >&2) || true
    fi
    # So độ dài LUỒNG HÌNH, không phải cả file: âm thanh chép nguyên nên file hỏng hình (S01-V012 là
    # H.264 bị đưa nhầm vào dav1d) vẫn có độ dài file khớp gốc.
    a=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 "$src" || echo -1)
    b=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 "$out.tmp.mp4" 2>/dev/null || echo 0)
    if python3 -c "import sys; sys.exit(abs(float('$a') - float('${b:-0}')) > 1)"; then
        mv "$out.tmp.mp4" "$out"; echo "$(date +%T) proxy $v ok GPU $gpu ($(du -m "$out" | cut -f1) MB)"
    else
        rm -f "$out.tmp.mp4"; echo "$(date +%T) proxy $v LỖI GPU $gpu: gốc ${a}s, ra ${b}s"
    fi
}

proxy_codec() { ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$VIDEO_SRC/$1"; }

# kind=av1: chỉ nhận video AV1; kind=gpu: chỉ nhận video giải mã được trên GPU. Không suy từ tên:
# S01-V012 là H.264 dù cùng họ S* với 11 video AV1.
proxy_worker() {
    local kind=$1 gpu=$2 v; shift 2
    for v in "$@"; do
        [[ -f "$OUT/video_720/${v%.*}.mp4" ]] && continue
        if [[ $v == S* ]]; then
            if [[ $(proxy_codec "$v") == av1 ]]; then [[ $kind == av1 ]] || continue
            else [[ $kind == gpu ]] || continue; fi
        fi
        mkdir "$OUT/video_720/.claim/$v" 2>/dev/null || continue
        proxy_one "$v" "$gpu"
    done
}

step_proxy() {
    local g mn s
    mkdir -p "$OUT/video_720/.claim"
    mn=$(ls "$VIDEO_SRC" | grep -E '^[MN].*\.(mp4|mov)$' | sort)
    s=$(cd "$VIDEO_SRC" && ls -S S*.mp4)
    for g in ${AV1_GPUS-2 5}; do proxy_worker av1 "$g" $s & done
    for g in ${PROXY_GPUS-0 1 3 4 6 7}; do proxy_worker gpu "$g" $mn $s & done
    wait
}

step_status() {
    echo "shots     : $(wc -l < "$OUT/shots.jsonl" 2>/dev/null || echo 0) shot"
    echo "asr       : $(cat "$OUT"/asr/shards/*/whisper.jsonl.done 2>/dev/null | sort -u | wc -l)/$(n_videos) video"
    echo "filter    : $(ls "$FRAMES"/*/statistics.json 2>/dev/null | wc -l)/$(n_videos) video"
    echo "ocr       : $(ls "$OUT/ocr/claims/done" 2>/dev/null | wc -l) video paddle xong, hybrid: $([[ -f "$OUT/ocr/output_hybrid.json" ]] && echo có || echo chưa)"
    echo "gemma_ocr : $(wc -l < "$OUT/gemma_ocr.jsonl" 2>/dev/null || echo 0) frame, N: $(wc -l < "$OUT/gemma_ocr_n.jsonl" 2>/dev/null || echo 0) video"
    echo "siglip    : $(ls "$OUT"/siglip/*.npy 2>/dev/null | wc -l) video"
    echo "log       : $OUT/logs/"
}

# Tách hẳn khỏi terminal/session (setsid + nohup): đóng Claude Code hay SSH đều không giết job.
detach() {
    local name="$1"; shift
    setsid nohup "$0" "$@" >> "$OUT/logs/$name.log" 2>&1 < /dev/null &
    echo "  $name -> nền, log: $OUT/logs/$name.log"
}

main() {
    case "${1:-}" in
    all)
        step_shots
        detach driver _driver
        ;;
    driver)
        detach driver _driver
        ;;
    _driver)
        step_driver
        ;;
    shots|asr|filter|ocr|gemma_ocr|gemma_ocr_n|siglip|warm|scan|ingest|proxy|status)
        s="$1"; shift; "step_$s" "$@"
        ;;
    *)
        sed -n '2,17p' "$0"; exit 1
        ;;
    esac
}

# Cùng một dòng: bash đọc script dần theo byte offset, sửa file khi job nền còn chạy thì
# sau khi main xong nó sẽ đọc tiếp giữa chừng file mới. exit ngay chặn chuyện đó.
main "$@"; exit $?
