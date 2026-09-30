#!/usr/bin/env bash
# Tải weight BEiT-3 (1.5GB) -- không nằm trong git vì vượt giới hạn 100MB/file của GitHub.
#   ./download_weights.sh        # idempotent: file đúng sha256 thì bỏ qua
set -euo pipefail
cd "$(dirname "$0")"

W=checkpoint/beit-3/beit3_large_patch16_224.pth
SHA=b76916cb3008d5e27519884b256dd1698faeb45ea7ab3950f319353e5051bd60
URL=https://github.com/addf400/files/releases/download/beit3/beit3_large_patch16_224.pth

[ -f "$W" ] && { echo "$SHA  $W" | sha256sum -c --status - && exit 0; }

mkdir -p "$(dirname "$W")"
echo "Tải $URL"
curl -fL --retry 3 -o "$W.part" "$URL"
echo "$SHA  $W.part" | sha256sum -c -
mv "$W.part" "$W"
echo "OK: $W"
