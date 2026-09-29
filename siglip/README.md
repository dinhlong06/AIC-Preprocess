# SigLIP Embedding Pipeline

Package `keyframe_pipeline` đọc keyframe ảnh của layer 2 và tạo **embedding
SigLIP2 1152 chiều** cho mỗi keyframe (chạy trong Docker, GPU).

## 1. Chạy bằng Docker (đường chạy chính)

```bash
./run.sh                               # 4 shard song song, mỗi shard 1 GPU
NSHARDS=1 ./run.sh                     # 1 GPU, 1 process
FRAMES_DIR=/path OUTPUT_DIR=/path ./run.sh
MIN_FREE_MB=3000 ./run.sh              # siết ngưỡng VRAM trống để chọn GPU
```

- Weight SigLIP (~3,5 GB) nằm ở `.model_cache/huggingface` và được bind-mount vào
  container, không nhét vào image. Lần chạy đầu tải model, các lần sau dùng lại.
- Script tự chọn GPU còn nhiều VRAM nhất qua `nvidia-smi` — máy dùng chung,
  đừng mặc định GPU 0.

## 2. Input

Mỗi thư mục video chứa trực tiếp các ảnh keyframe:

```text
<dataset-root>/
└── L21_V001/
    ├── 001.jpg
    └── ...
```

- Hỗ trợ `.jpg`, `.jpeg`, `.png`, `.webp`; natural order.
- `video_id` là tên thư mục, phải khớp `[A-Z]+<số>_V<số>`.
- `--dataset-root` nhận cả thư mục chứa trực tiếp các thư mục video, lẫn thư mục
  cha có `keyframes/` bên trong.

## 3. Output

```text
artifacts/siglip_batch1_v2/
├── L21_V001.npy
└── L21_V001_ids.json
```

- `.npy`: NumPy array shape `(N, 1152)`, dtype `float32`, mỗi hàng L2-normalize;
  hàng thứ `i` ứng với ID thứ `i` trong `_ids.json`.
- File ID là list phẳng JSON: `["001","002",...]`.
- Cùng format với embedding BEiT-3 của layer 2 (khác số chiều: 1152 vs 1024).
- Downstream: `layer_5/indexdb/ingest_batch1.py` đọc `--siglip2-dir` từ đây.

## 4. CLI

Entrypoint của image là `python3 -m keyframe_pipeline`:

| Subcommand | Việc |
|---|---|
| `siglip-dataset` | embed cả dataset |
| `siglip-video` | embed một video |
| `run` | alias siglip-dataset qua facade (`KeyframePipeline`) |

## 5. Public Python API

```python
from pathlib import Path

from keyframe_pipeline import KeyframePipeline, discover_video

video = discover_video(Path("/path/frames/L21_V001"))
pipeline = KeyframePipeline(siglip_device=None)  # None = tự chọn CUDA nếu có

siglip = pipeline.run_siglip(
    video.keyframes,
    output_dir=Path("artifacts/siglip"),
    batch_size=32,
)
print(siglip.embeddings.shape)   # (N, 1152)
```

Đọc lại artifact đã lưu:

```python
from keyframe_pipeline import load_siglip_result

siglip = load_siglip_result(
    Path("artifacts/siglip/L21_V001.npy"),
    Path("artifacts/siglip/L21_V001_ids.json"),
)
```

## 6. Lỗi thường gặp

- `Output already exists`: thêm `--overwrite` nếu thực sự muốn thay artifact cũ.
- `CUDA was requested ... but CUDA PyTorch is unavailable`: bỏ `--device cuda:0`.
- `Expected .../keyframes or direct <PREFIX>nn_Vnnn video directories`: sai
  `--dataset-root`, hoặc tên thư mục video không khớp pattern ở mục 2.
