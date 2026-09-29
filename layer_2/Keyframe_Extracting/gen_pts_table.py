"""Sinh bảng pts_ms đầy đủ (mọi frame, không chỉ keyframe) cho các video VFR (N*).

Một file gộp duy nhất — tránh nhiều file nhỏ trên NFS. Mỗi dòng:
{"video_id": "...", "pts_ms": [pts frame 0, pts frame 1, ...]}
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VIDEOS, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def pts_of(video_id):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0",
         str(next(VIDEOS.glob(video_id + ".*")))],
        capture_output=True, text=True, check=True, env={"LC_ALL": "C", "PATH": "/usr/bin:/usr/local/bin"}).stdout
    pts = sorted(round(float(x) * 1000) for x in out.split() if x != "N/A")
    return video_id, pts


ids = sorted({p.name.rsplit(".", 1)[0] for p in VIDEOS.glob("N*.*") if p.suffix in (".mov", ".mp4")})
done = set()
if OUT.exists():
    done = {json.loads(l)["video_id"] for l in OUT.read_text().splitlines()}
todo = [v for v in ids if v not in done]
print(f"{len(done)} đã có, {len(todo)} cần làm", flush=True)

with OUT.open("a") as f, ThreadPoolExecutor(2) as ex:
    for vid, pts in ex.map(pts_of, todo):
        f.write(json.dumps({"video_id": vid, "pts_ms": pts}) + "\n")
        f.flush()
        print(vid, len(pts), "frames", flush=True)
