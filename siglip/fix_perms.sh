#!/usr/bin/env bash
# Sửa quyền các artifact đã ghi ra trước khi io.py::_atomic_write được vá.
#
# Nguyên nhân: tempfile.mkstemp() tạo file mode 600, os.replace() giữ nguyên mode
# đó -> artifact không đọc được từ container chạy user khác (layer_5 ingest chạy
# bằng appuser uid 10001). Chỉ cần chạy MỘT LẦN cho data cũ; artifact mới đã 644.
#
#   ./fix_perms.sh          # sửa artifacts/ của siglip
#   ./fix_perms.sh <dir>    # sửa thư mục khác

set -euo pipefail

TARGET="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/artifacts}"

echo "== Quét $TARGET =="
mapfile -t FILES < <(find "$TARGET" -type f ! -perm -o=r)
if [ ${#FILES[@]} -eq 0 ]; then
    echo "   không có file nào thiếu quyền đọc."
    exit 0
fi

printf '%s\n' "${FILES[@]}"
chmod 644 "${FILES[@]}"
find "$TARGET" -type d ! -perm -o=x -exec chmod 755 {} +
echo "== Đã sửa ${#FILES[@]} file =="
