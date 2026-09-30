# Layer 2 — Keyframe Extraction + Transcript Mapping

Turns layer 1's shots + ASR into keyframe images with metadata, and maps
transcripts onto shots:

```
shots.jsonl ──► Keyframe_Extracting (pipeline G/H) ──► keyframes.jsonl + .jpg per video
whisper.jsonl ─► shot_transcript (cpu_map_transcript.py) ──► shot_transcripts.jsonl
```

Two components:

| Component | What it does | Docs |
|---|---|---|
| `Keyframe_Extracting/` | Pick representative frames per shot (BEiT-3 semantic filtering, 7-stage pipeline G; H adds text-prescan). Runs on GPU. | [Keyframe_Extracting/README.md](Keyframe_Extracting/README.md) |
| `shot_transcript/` | CPU-only: join ASR segments to shots by time overlap (500ms gap tolerance), one text per shot. | [shot_transcript/](shot_transcript/) |

## Production runs

| Batch | Keyframes | Shot transcripts |
|---|---|---|
| batch1 (873 videos) | `Keyframe_Extracting/benchmark_batch1_v2/pipeline_g` | `shot_transcript/shot_transcripts_batch1.jsonl` |
| batch2 (614 videos) | `output_batch2/keyframes/pipeline_h` | `output_batch2/shot_transcripts.jsonl` |

## Downstream

- `layer_3` OCRs the keyframe images (`frame_id` = `keyframe_id`).
- `siglip` embeds them (same ID).
- `layer_5` ingests keyframes.jsonl + shot transcripts + OCR + captions into
  Mongo/Elasticsearch/Milvus.

## Usage

```bash
# keyframes (batch1 defaults)
cd Keyframe_Extracting && ./run_shards.sh

# shot transcripts
cd shot_transcript && ./run.sh                              # batch2
L1_DIR=../../layer_1/batch1 OUT=shot_transcripts_batch1.jsonl ./run.sh   # batch1
```
