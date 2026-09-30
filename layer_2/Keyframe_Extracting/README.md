# Keyframe Extractor — AI Challenge 2026

Extracts keyframes from video using shot detection results.
Supports **2 pipelines**: `pipeline_g` (default, production) and
`pipeline_h` (G + text-region awareness).

---

## System overview

```
Shot Detector
     ↓
shot.jsonl / shots.json
     ↓
Keyframe Extraction   ← this module
     ↓
keyframes.jsonl + .jpg images
     ↓
Embedding → Milvus
```

---

## The 2 pipelines

| Pipeline | DAKE | Encoder | Goal |
|----------|------|---------|------|
| **G** | ✅ | BEiT-3 Large (1024-dim) | **Default** — 7 stages, tuned params v3.1 |
| **H** | ✅ | BEiT-3 Large | G + `text_prescan`: prefers frames with changing text regions |

Both share the G parameter set; H adds a `text_prescan` block in config.

### What is DAKE?

DAKE is a **Coarse Temporal Filter** — not a semantic model.
It works on JPEG file sizes (no CNN, no GPU).
Its job: shrink the frame set before encoding (keeps `candidate_ratio × total_frames`).

---

## Directory layout

```
Keyframe_Extracting/
├── checkpoint/
│   └── beit-3/
│       ├── beit3_large_patch16_224.pth   ← BEiT-3 weights (1.5 GB, ./download_weights.sh)
│       └── beit3.spm                      ← SentencePiece tokenizer
│
├── configs/
│   ├── pipeline_g.yaml    ← Pipeline G config (default)
│   └── pipeline_h.yaml    ← Pipeline H config (G + text_prescan)
│
├── src/
│   ├── core/
│   │   ├── models.py      ← Dataclasses: ShotRecord, Keyframe, ShotKeyframes, PipelineStatistics
│   │   ├── interfaces.py  ← BaseKeyframeExtractor ABC
│   │   ├── runner.py      ← KeyframeBenchmarkRunner (orchestrates everything)
│   │   └── metrics.py     ← Diversity, Coverage, Precision/Recall vs GT
│   │
│   ├── components/
│   │   ├── frame_loader.py      ← Reads frames from video or pre-cut keyframe dirs
│   │   ├── dake.py              ← DAKE: JPEG steepness → sliding window → top-k
│   │   ├── beit3_encoder.py     ← BEiT-3 Large visual encoder (1024-dim CLS)
│   │   ├── mobilenet_encoder.py ← MobileNetV3-Large, GT-evaluation only
│   │   ├── semantic_filter.py   ← Cosine similarity sequential filter
│   │   ├── text_prescan.py      ← Text-region change detector (H only)
│   │   ├── diversity_filter.py  ← Spreads keyframes within a shot
│   │   ├── transition_selector.py, blank_veto.py, sharpness_selector.py
│   │
│   └── extractors/
│       ├── pipeline_g.py  ← 7 stages, default
│       └── pipeline_h.py  ← G + text_prescan
│
├── beit3_src/               ← vendored BEiT-3 wrapper (microsoft/unilm, MIT)
│   └── modeling_utils.py
│
├── dataset/
│   ├── raw_video/         ← Input .mp4 videos
│   └── shots/             ← shots.json from Shot Detector (1 folder/video)
│
├── benchmark*/            ← Auto-generated output (git-ignored)
│   ├── pipeline_g/
│   ├── pipeline_h/
│   └── benchmark_summary.csv
│
├── run.sh / run_shards.sh ← entry points (Docker)
├── download_weights.sh    ← fetch beit3_large_patch16_224.pth (over GitHub's 100 MB limit)
├── cli.py                 ← CLI (invoked inside the image by run.sh)
└── requirements.txt
```

## Module I/O (interface between modules)

### 1. INPUT

*   **Raw video (`.mp4`)**: path to the video to process.
*   **`shots.json`**: shot list produced by the Shot Detector module. Each shot has:
    *   `video_id`: video name.
    *   `shot_id`: shot code (e.g. `S0001`).
    *   `start_frame` & `end_frame`: first and last frames.
    *   *Without this file, the system chunks the video into small pieces (300 frames/chunk) to avoid RAM overflow.*

### 2. OUTPUT

One directory per video containing:
*   **Keyframe images (`.jpg`)**: stored as `<video_id>/<shot_id>/kf0001.jpg`. Later modules (e.g. SigLIP) read these into the vector database.
*   **`keyframes.jsonl`**: metadata for every keyframe in the video, one JSON object per line:
    ```json
    {"video_id": "L30_V001", "shot_id": "S0001", "keyframe_id": "S0001_kf0001", "frame_idx": 42, "timestamp_ms": 1680, "image_path": "S0001/kf0001.jpg"}
    ```
    *Field meanings:*
    *   `keyframe_id`: unique ID.
    *   `frame_idx`: true frame position in the source video.
    *   `timestamp_ms`: timestamp in milliseconds (used by the video player).
    *   `image_path`: relative path to the cropped `.jpg`.

---

## Running

### 1. Prepare the dataset

```
dataset/
├── raw_video/
│   ├── video1.mp4
│   └── video2.mp4
└── shots/
    ├── video1/
    │   └── shots.json    ← from Shot Detector
    └── video2/
        └── shots.json
```

> **Note**: without `shots.json`, the module treats the whole video as 1 shot.

### 2. Run Pipeline G (default)

`run.sh` builds the Docker image, downloads the BEiT-3 weight if missing and
splits the shared `shots.jsonl` per video. Paths default to the batch2 layout
at project root; override with `VIDEO_DIR`, `SHOTS_SRC`, `OUTPUT_DIR` (absolute
paths — they are mounted into the container).

```bash
./run.sh                    # pipeline_g in Docker
./run.sh all                # G and H
MODE=host ./run.sh pipeline_g   # host python3 (needs deps + weights installed)

NSHARDS=3 ./run_shards.sh   # fan out across GPUs (add/drain/stop/watchdog)
```

### 3. Override params via CLI (forwarded by run.sh)

```bash
./run.sh pipeline_g --threshold 0.85 --candidate_ratio 0.05 --device cuda --batch_size 16
```

---

## Output

### Output directory layout

```
benchmark/
├── pipeline_g/
│   ├── video1/
│   │   ├── keyframes.jsonl     ← metadata per keyframe (1 line/KF)
│   │   ├── statistics.json     ← per-video stats
│   │   ├── S0001/
│   │   │   ├── kf0001.jpg
│   │   │   └── kf0002.jpg
│   │   └── S0002/
│   │       └── kf0001.jpg
│   └── video2/
│       └── ...
└── benchmark_summary.csv       ← all pipelines × videos
```

### keyframes.jsonl (one line = one keyframe)

```json
{"video_id": "video1", "shot_id": "S0001", "keyframe_id": "S0001_kf0001", "frame_idx": 42, "timestamp_ms": 1680, "image_path": "S0001/kf0001.jpg"}
{"video_id": "video1", "shot_id": "S0001", "keyframe_id": "S0001_kf0002", "frame_idx": 120, "timestamp_ms": 4800, "image_path": "S0001/kf0002.jpg"}
```

### benchmark_summary.csv

Example:
| Pipeline | Video | #KF | KF/Shot | Time(s) | FPS | VRAM(GB) | Storage(MB) | Diversity | Coverage |
|----------|-------|-----|---------|---------|-----|----------|-------------|-----------|----------|
| pipeline_g | video1 | 820 | 2.5 | 145 | 62 | 4.2 | 312 | 0.0 | 0.91 |
| pipeline_h | video1 | 650 | 2.0 | 52 | 170 | 2.1 | 248 | 0.0 | 0.94 |



---

## Benchmark metrics

### 1. Keyframe quality (vs Ground Truth)

| Metric | Description | Needs GT? |
|--------|-------------|-----------|
| Recall | % of GT keyframes found | ✅ |
| Precision | % of predicted keyframes correct | ✅ |
| F1-score | Harmonic mean | ✅ |
| Diversity score | 1 - avg cosine sim between KFs (higher = better) | ❌ |
| Coverage score | % of shot time represented | ❌ |

> GT matching uses nearest neighbor (cosine similarity ≥ 0.80).

### 2. Computational efficiency

| Metric | Description |
|--------|-------------|
| Processing time (s) | Time to process the whole video |
| Processing FPS | Frames/second |
| Peak VRAM (GB) | GPU memory peak |

### 3. Storage cost

| Metric | Description |
|--------|-------------|
| #Keyframes | Total keyframes |
| KF/Shot | Average keyframes/shot |
| Storage (MB) | .jpg size on disk |

### 4. Downstream retrieval impact

Evaluated later after Embedding + Milvus integration:
- mAP@K, R@1/R@5/R@10 on the test query set.
---

## DAKE config (candidate_ratio benchmark)

Try these values to find the best balance:

| candidate_ratio | Meaning | Speed |
|----------------|---------|--------|
| 0.01 | Keep 1% of frames | Fastest |
| 0.02 | Keep 2% of frames | **Default** |
| 0.05 | Keep 5% of frames | Medium |
| 0.10 | Keep 10% of frames | Slower |
| 0.20 | Keep 20% of frames | Best quality |

Suggestions only — feel free to try more.
---
