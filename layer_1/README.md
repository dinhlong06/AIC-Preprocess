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

## Setup and run

```bash
cd layer_1
./run_shards.sh shots   # TransNetV2 shot detection, CPU-bound, all GPUs
./run_shards.sh asr     # ChunkFormer ASR, GPU-bound
./run_shards.sh merge   # merge shard jsonl -> shots.jsonl + whisper.jsonl
```

- `./run_layer1.sh` runs the same pipeline on a single GPU
  (`--skip_shots`, `--skip_asr`, `--videos ...`, `--force` supported).
- Defaults point at batch2; for batch1:
  `VIDEO_DIR=../dataset_batch1/videos/video BATCH_DIR=$PWD/batch1 ./run_shards.sh shots`
- `merge` refuses to write a file missing videos unless `FORCE=1`.
- ChunkFormer weights live in `cache/huggingface` (bind-mounted, downloaded on
  first run).

## Folder structure

```
layer_1/
├── gpu_shot_and_asr.py   # the only entry point (shots + ASR, resumable, claim-based sharding)
├── run_layer1.sh         # build + run in Docker on the freest GPU
├── run_shards.sh         # multi-GPU fanout: shots / asr / merge
├── transnetv2/inference  # vendored TransNetV2 code + TF weights (~35MB, copied into the image)
├── Dockerfile            # TF 2.13 GPU + torch cu118 + static ffmpeg
└── requirements.txt
```
