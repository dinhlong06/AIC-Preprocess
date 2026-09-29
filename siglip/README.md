# SigLIP Embedding Pipeline

The `keyframe_pipeline` package reads image keyframes from layer 2 and produces
a **SigLIP2 1152-dim embedding** for each keyframe (runs in Docker on GPU).

## 1. Running via Docker (main path)

```bash
./run.sh                               # 4 parallel shards, one GPU per shard
NSHARDS=1 ./run.sh                     # 1 GPU, 1 process
FRAMES_DIR=/path OUTPUT_DIR=/path ./run.sh
MIN_FREE_MB=3000 ./run.sh              # tighten free-VRAM threshold for GPU picking
```

- SigLIP weights (~3.5 GB) live in `.model_cache/huggingface` and are bind-mounted
  into the container, not baked into the image. First run downloads the model,
  later runs reuse it.
- The script picks the GPU with the most free VRAM via `nvidia-smi` — shared
  machine, do not default to GPU 0.
- Re-running `./run.sh` resumes: videos that already have a `.npy` are skipped.

## 2. Input

Each video directory contains its keyframe images directly:

```text
<dataset-root>/
└── L21_V001/
    ├── 001.jpg
    └── ...
```

- Supports `.jpg`, `.jpeg`, `.png`, `.webp`; natural sort order.
- `video_id` is the directory name and must match `[A-Z]+<digits>_V<digits>`.
- `--dataset-root` accepts either a directory holding video directories
  directly, or a parent that contains a `keyframes/` subdirectory.

## 3. Output

```text
artifacts/siglip_batch1_v2/
├── L21_V001.npy
└── L21_V001_ids.json
```

- `.npy`: NumPy array of shape `(N, 1152)`, dtype `float32`, each row
  L2-normalized; row `i` corresponds to ID `i` in `_ids.json`.
- The IDs file is a flat JSON list: `["001","002",...]`.
- Same format as layer 2's BEiT-3 embeddings (differs only in dimension:
  1152 vs 1024).
- Downstream: `layer_5/indexdb/ingest_batch1.py` reads `--siglip2-dir` from here.

## 4. CLI

The image entrypoint is `python3 -m keyframe_pipeline`:

| Subcommand | Purpose |
|---|---|
| `siglip-dataset` | embed a whole dataset (auto-resumes: videos with an existing `.npy` are skipped) |
| `siglip-video` | embed a single video |

## 5. Public Python API

```python
from pathlib import Path

from keyframe_pipeline import discover_video, extract_siglip

video = discover_video(Path("/path/frames/L21_V001"))

siglip = extract_siglip(
    video.keyframes,
    output_dir=Path("artifacts/siglip"),
    batch_size=32,
)
print(siglip.embeddings.shape)   # (N, 1152)
```

Reading back a saved artifact:

```python
from keyframe_pipeline import load_siglip_result

siglip = load_siglip_result(
    Path("artifacts/siglip/L21_V001.npy"),
    Path("artifacts/siglip/L21_V001_ids.json"),
)
```

## 6. Common errors

- `Output already exists`: pass `--overwrite` if you really want to replace the artifact.
- `CUDA was requested ... but CUDA PyTorch is unavailable`: drop `--device cuda:0`.
- `Expected .../keyframes or direct <PREFIX>nn_Vnnn video directories`: wrong
  `--dataset-root`, or the video directory name doesn't match the pattern in
  section 2.
