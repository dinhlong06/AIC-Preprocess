"""Thêm pts_ms (pts_time thật × 1000) vào keyframes.jsonl của video VFR (N*).

frame_idx là thứ tự hiển thị của OpenCV, tức chỉ số trong danh sách pts đã sort.
"""
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VIDEOS, KF_DIR = Path(sys.argv[1]), Path(sys.argv[2])


def add_pts(video_id):
    jsonl = KF_DIR / video_id / "keyframes.jsonl"
    if (bak := jsonl.with_suffix(".jsonl.nopts")).exists():
        return video_id, "skip"
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0",
         str(next(VIDEOS.glob(video_id + ".*")))],
        capture_output=True, text=True, check=True, env={"LC_ALL": "C", "PATH": "/usr/bin:/usr/local/bin"}).stdout
    pts = sorted(float(x) for x in out.split() if x != "N/A")
    rows = [json.loads(l) for l in jsonl.read_text().splitlines()]
    shutil.copy(jsonl, bak)
    for r in rows:
        r["pts_ms"] = round(pts[r["frame_idx"]] * 1000)
    jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return video_id, f"{len(pts)} frames"


ids = sorted(p.name for p in KF_DIR.glob("N*-V*") if p.is_dir())
with ThreadPoolExecutor(2) as ex:
    for vid, msg in ex.map(add_pts, ids):
        print(vid, msg, flush=True)
