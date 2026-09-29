from __future__ import annotations

DEFAULT_SIGLIP_MODEL_ID = "google/siglip2-so400m-patch14-384"
SIGLIP_EMBEDDING_DIM = 1152
# Keyframe nằm trên NFS: decode tốn 160 ms/ảnh vì chờ IO, GPU chỉ chiếm 7% thời gian.
# Đo trên 714 ảnh: 0 worker 125s, 4 worker ~20s, 16 worker chậm lại (tranh CPU
# trên máy dùng chung 40 core). 4 để còn chỗ chạy song song nhiều shard.
DEFAULT_SIGLIP_NUM_WORKERS = 4
