#!/usr/bin/env bash
# Caption batch1 chia 3 nguồn: AI Studio 26B + 31B chạy từ cuối danh sách (bắt đầu ngay, song song OCR),
# UIT Gemma 1120 chạy từ đầu sau khi OCR (PID truyền vào) xong và bỏ qua phần AI Studio đã làm.
# AI Studio hết quota ngày thì python thoát -> chờ 30 phút thử lại (quota reset theo giờ Mỹ).
#   ./run_caption_batch1.sh <ocr_pid>
set -euo pipefail
cd "$(dirname "$0")"
set -a; . ../../retrieval_system/.env; set +a
F=../../layer_2/Keyframe_Extracting/benchmark_batch1_v2/pipeline_g
ALL="ais26_caption_batch1.jsonl ais31_caption_batch1.jsonl gemma_caption_batch1.jsonl"

for m in 26 31; do
    shard=$([ $m = 26 ] && echo 0/2 || echo 1/2)
    setsid nohup bash -c "until python3 -u gemma_ocr.py --task caption --model ais$m --reverse --shard $shard \
        --frames $F --out ais${m}_caption_batch1.jsonl --skip $ALL; do sleep 1800; done" \
        >> ais${m}_caption_batch1.log 2>&1 < /dev/null &
done

if [ -n "${1:-}" ]; then
    setsid nohup bash -c "while kill -0 $1 2>/dev/null; do sleep 60; done; \
        until python3 -u gemma_ocr.py --task caption --frames $F --out gemma_caption_batch1.jsonl --skip $ALL; do sleep 300; done" \
        >> gemma_caption_batch1.log 2>&1 < /dev/null &
fi
