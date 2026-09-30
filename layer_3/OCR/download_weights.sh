#!/usr/bin/env bash
# Tải weight OCR (163MB DeepSolo + 274MB PARSeq-VN) -- vượt 100MB/file nên không nằm trong git.
#   ./download_weights.sh        # idempotent: file đúng sha256 thì bỏ qua
set -euo pipefail
cd "$(dirname "$0")"

# file -> "sha256 url"
FILES=(
    "experiments/deepsolo_parseq/weights/ic15_res50_finetune_synth-tt-mlt-13-15-textocr.pth 4f8ef40ee1df3535e8965a7ce064ee9e1d7f143164d04eb43af60b131bf00927 https://huggingface.co/rwood-97/DeepSolo_ic15_res50/resolve/main/ic15_res50_finetune_synth-tt-mlt-13-15-textocr.pth"
    "experiments/deepsolo_parseq/vn_scenetext/weights/rec/best-parseq.ckpt 6cdd37ecbdbc2615958c68187705b2c21240a0ccb2943f29454c514595a2bd21 https://github.com/dinhlong06/retrieval_system/releases/download/weights/best-parseq.ckpt"
)

for entry in "${FILES[@]}"; do
    read -r path sha url <<<"$entry"
    if [ -f "$path" ] && echo "$sha  $path" | sha256sum -c --status -; then
        continue
    fi
    mkdir -p "$(dirname "$path")"
    echo "Tải $url"
    curl -fL --retry 3 -o "$path.part" "$url"
    echo "$sha  $path.part" | sha256sum -c -
    mv "$path.part" "$path"
    echo "OK: $path"
done
