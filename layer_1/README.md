# Layer 1 — Shot Segmentation + ASR

Turns raw videos into two JSONL files that every later layer consumes:

```
video.mp4 ──► TransNetV2 ──► shots.jsonl     (shot boundaries, per video)
      └─────► ffmpeg ──► Silero VAD ──► ChunkFormer ──► whisper.jsonl  (ASR segments)
```

- `shots.jsonl` — one row per shot: `video_id`, `shot_id`, `start_frame`, `end_frame`, `fps`.
- `whisper.jsonl` — one row per speech segment: `video_id`, `seg_id`, `start_ms`, `end_ms`, `text`, `confidence`.

Both are resume-safe (`.done` markers per video) and shard-safe (O_EXCL claim
files let N containers share one work queue without double-processing).

## Production runs

| Batch | Input | Output | Status |
|---|---|---|---|
| batch1 (L21–L30, 873 videos) | `dataset_batch1/videos/video` | `batch1/shots.jsonl`, `batch1/whisper.jsonl` | complete (873/873) |
| batch2 (M/N/S, 614 videos) | `dataset/video` | `shots.jsonl`, `whisper.jsonl` (repo root) | complete (614/614) |

## Files

| File | Purpose |
|---|---|
| `gpu_shot_and_asr.py` | The only entry point: shot detection + ASR, resumable, claim-based sharding |
| `run_layer1.sh` | Build + run in Docker on the freest GPU. Env: `VIDEO_DIR` (input), `BATCH_DIR`/`OUTPUT_DIR` (output) |
| `run_shards.sh` | Multi-GPU fanout: `shots` / `asr` / `merge` stages, work-stealing via claim dir |
| `Dockerfile` | TF 2.13 GPU + torch cu118 + static ffmpeg (AV1) + TransNetV2 |
| `tests/` (untracked) | Unit tests for entry building, ChunkFormer wiring, and the layer-2 contract |

## Usage

```bash
./run_shards.sh shots        # TransNetV2, CPU-bound, all GPUs
./run_shards.sh asr          # ChunkFormer, GPU-bound
./run_shards.sh merge        # merge shard jsonl -> shots.jsonl + whisper.jsonl
```

Defaults point at batch2; for batch1:
`VIDEO_DIR=$PWD/../dataset_batch1/videos/video BATCH_DIR=$PWD/batch1 ./run_shards.sh shots`

## Notes

- ASR backend is **ChunkFormer CTC (khanhld/chunkformer-ctc-large-vie)** — one
  `endless_decode` call per VAD segment. PhoWhisper was measured against it and
  removed (history in git).
- Videos with no audio get a `.done` marker with 0 segments — otherwise merge
  would wait forever.
- `merge` refuses to produce a file missing videos (progress < total) unless
  `FORCE=1`.
- Model weights live in `cache/huggingface` (bind-mounted, not baked into the
  image); first run downloads them.
