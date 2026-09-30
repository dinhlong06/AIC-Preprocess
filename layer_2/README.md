# Layer 2 — Keyframe Extraction + Transcript Mapping

Turns layer 1's shots + ASR into keyframe images with metadata, and maps
transcripts onto shots:

```
shots.jsonl ──► Keyframe_Extracting (pipeline G/H) ──► keyframes.jsonl + .jpg per video
whisper.jsonl ─► shot_transcript (cpu_map_transcript.py) ──► shot_transcripts.jsonl
```

- `pipeline_g` (default): 7-stage BEiT-3 semantic filter, params tuned v3.1.
- `pipeline_h`: G + `text_prescan`, prefers frames with changing text regions.
- `cpu_map_transcript.py`: CPU-only join of ASR segments to shots by time
  overlap (500ms gap tolerance), one text per shot.

See [Keyframe_Extracting/README.md](Keyframe_Extracting/README.md) for the
extractor stages and params.

## Setup and run

```bash
# keyframes
cd Keyframe_Extracting
./run_shards.sh                        # pipeline_g, 3 shards across GPUs
PIPELINE=pipeline_h ./run_shards.sh    # text-aware variant
# add / drain / stop / watchdog subcommands manage background shards

# shot transcripts
cd ../shot_transcript
./run.sh                                                          # batch2
L1_DIR=../../layer_1/batch1 OUT=shot_transcripts_batch1.jsonl ./run.sh   # batch1
```

## Folder structure

```
layer_2/
├── Keyframe_Extracting/
│   ├── cli.py                 # CLI: --pipeline {g,h,all} + param overrides
│   ├── run.sh                 # build + run (docker or host mode)
│   ├── run_shards.sh          # multi-GPU fanout with add/drain/stop/watchdog
│   ├── configs/               # pipeline_g.yaml, pipeline_h.yaml
│   ├── src/core/              # runner, models, metrics, interfaces
│   ├── src/extractors/        # pipeline_g.py, pipeline_h.py
│   ├── src/components/        # dake, beit3_encoder, semantic_filter, text_prescan, ...
│   ├── checkpoint/beit-3/     # BEiT-3 weights + tokenizer
│   ├── Dockerfile
│   └── requirements.txt
└── shot_transcript/
    ├── cpu_map_transcript.py  # ASR segments → one text per shot
    └── run.sh                 # thin wrapper (env: L1_DIR, OUT)
```
