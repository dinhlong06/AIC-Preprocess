# Video Retrieval System — AIC 2026

**Finalist — AI Challenge 2026, Ho Chi Minh City.** This is the end-to-end
retrieval pipeline our team took to the contest finals, covering all three
query types of the challenge: Textual KIS, QA, and TRAKE.

From raw videos to a searchable index: shot segmentation, keyframe
extraction, OCR + ASR text, SigLIP2 embeddings, and a hybrid
vector (Milvus) + BM25 (Elasticsearch) search API.

## Pipeline

```
video/*.mp4
  │
  ├─ layer_1   TransNetV2 ──────────────────► shots.jsonl      (shot boundaries)
  │            ffmpeg + Silero VAD + ChunkFormer ─► whisper.jsonl (ASR segments)
  │
  ├─ layer_2   Keyframe_Extracting (BEiT-3, pipeline G/H) ─► keyframes/*.jpg + keyframes.jsonl
  │            shot_transcript (ASR ∩ shots) ─────────────► shot_transcripts.jsonl
  │
  ├─ layer_3   OCR_gemma (Gemma 4 over API) ──► gemma_ocr*.jsonl
  │            OCR (DeepSolo + PARSeq-VN, GPU) ► output_ocr.json
  │            siglip (SigLIP2 embeddings) ───► <video>.npy + <video>_ids.json
  │
  └─ layer_5   ingest ──► Milvus (vector) + Elasticsearch (BM25) + MongoDB (source of truth)
               api ─────► HTTP /search/* (FastAPI, key auth)
```

Each `layer_N/` folder is self-contained and has its own README with details.

## Setup

Prerequisites: Linux with NVIDIA drivers + Docker (nvidia-container-toolkit),
`python3.8+` on the host, ~60 GB free disk.

1. **Build the Docker images** (one per layer; scripts build them on demand too):

   ```bash
   docker build -t ai26-layer1 layer_1
   docker build -t ai26-layer2 layer_2/Keyframe_Extracting
   docker build -t ocr-deepsolo-parseq layer_3/OCR
   # layer_5 / siglip images are built automatically by their run.sh
   ```

2. **Download model weights** — files over GitHub's 100 MB limit are not in
   git; each script is idempotent and verifies sha256:

   ```bash
   layer_2/Keyframe_Extracting/download_weights.sh   # BEiT-3, 1.5 GB
   layer_3/OCR/download_weights.sh                   # DeepSolo 163 MB + PARSeq-VN 274 MB
   ```

   Downloaded automatically on first run (via HuggingFace): ChunkFormer ASR
   (~1.5 GB) and SigLIP2 (~3.5 GB). TransNetV2 weights are vendored in the repo.
   The PARSeq-VN checkpoint is hosted on this repo's
   [releases](https://github.com/dinhlong06/retrieval_system/releases).

3. **Run the pipeline** (order matters):

   ```bash
   cd layer_1 && ./run_shards.sh shots && ./run_shards.sh asr && ./run_shards.sh merge
   cd ../layer_2/Keyframe_Extracting && ./run_shards.sh
   cd ../../layer_3/OCR && ./run.sh                  # or OCR_gemma for the API path
   cd ../../siglip && ./run.sh
   cd ../layer_5 && ./run.sh up && ./run.sh init && ./run.sh ingest-batch1
   ```

4. **Search**: `./run.sh up` in `layer_5` starts the API on port 8021
   (header `x-api-key: 123` by default, override with `export API_KEY=...`),
   plus Attu (Milvus UI) and Elasticvue (ES UI).

## Folder structure

```
├── layer_1/                  # shot segmentation (TransNetV2) + ASR (ChunkFormer)
├── layer_2/
│   ├── Keyframe_Extracting/  # BEiT-3 keyframe selection, pipeline G/H
│   └── shot_transcript/      # map ASR segments onto shots
├── layer_3/
│   ├── OCR/                  # DeepSolo + PARSeq-VN local GPU OCR
│   └── OCR_gemma/            # Gemma 4 OCR + captions over HTTP API
├── siglip/                   # SigLIP2 embeddings for keyframes
└── layer_5/                  # Milvus + Elasticsearch + MongoDB ingest & search API
```

Model weights, datasets, and outputs are git-ignored (see `.gitignore`); the
download scripts place weights where each layer expects them.
