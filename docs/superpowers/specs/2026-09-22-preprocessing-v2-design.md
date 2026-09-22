# Preprocessing Pipeline V2 Design

**Date:** 2026-09-22  
**Status:** Awaiting user review  
**Scope:** Layers 1-3 of the offline preprocessing pipeline

## 1. Goal

Add a clearly separated preprocessing v2 that can run end to end or by
individual stage, while leaving the existing Layer 1, Layer 2, and Layer 3
pipelines unchanged. V2 must make it easy to compare old and new outputs and
must add trajectory analysis for configured video categories.

V2 covers:

- Layer 1a: TransNetV2 shot detection with configurable thresholds.
- Layer 1b: Silero VAD plus ChunkFormer Vietnamese ASR.
- Layer 1c: trajectory extraction and conservative motion-event labelling.
- Layer 2: the existing Pipeline H keyframe path.
- Layer 3: PaddleOCR detection plus a configurable PARSeq-Vietnamese
  recognizer, with the legacy recognizer as an explicit fallback.

Layer 4 embeddings, indexing, and online retrieval are outside this design,
matching `preprocessing_layer_expansion_plan.md`.

## 2. Compatibility and isolation

The following existing files and behaviors remain untouched:

- `layer_1/gpu_shot_and_asr.py` and all `layer_1/run_*.sh` scripts.
- Existing Layer 1 outputs such as `layer_1/shots.jsonl` and
  `layer_1/whisper.jsonl`.
- Existing Pipeline G and all existing Layer 2 entrypoints.
- Existing Layer 3 OCR entrypoints and outputs.

V2 lives under a new top-level `preprocessing_v2/` package. It calls stable
existing entrypoints through adapters where reuse is appropriate, rather than
copying or modifying v1 code. All generated files live under a separate v2
artifact root.

## 3. Proposed layout

```text
preprocessing_v2/
├── configs/
│   ├── pipeline_v2.yaml
│   └── trajectory.yaml
├── preprocessing_v2/
│   ├── __init__.py
│   ├── config.py
│   ├── orchestrator.py
│   ├── manifest.py
│   ├── shot.py
│   ├── asr_chunkformer.py
│   ├── trajectory.py
│   ├── keyframes.py
│   ├── ocr.py
│   └── compare.py
├── tests/
├── Dockerfile
├── requirements.txt
├── run_v2.py
├── run_v2.sh
└── README.md
```

Default output layout:

```text
artifacts/preprocessing_v2/
├── manifest.json
├── layer_1/
│   ├── shots.jsonl
│   ├── shots.jsonl.done
│   ├── whisper.jsonl
│   ├── whisper.jsonl.done
│   ├── trajectories.jsonl
│   └── trajectory_tracks.jsonl
├── layer_2/pipeline_h/
└── layer_3/ocr/
    ├── output.json
    └── candidates.json
```

## 4. CLI and stage selection

`run_v2.sh` is the end-to-end entrypoint and forwards arguments to
`run_v2.py`. Stage selection is available at both layer and component level.

Examples:

```bash
./preprocessing_v2/run_v2.sh
./preprocessing_v2/run_v2.sh --skip-layer1
./preprocessing_v2/run_v2.sh --skip-layer2 --skip-ocr
./preprocessing_v2/run_v2.sh --skip-shots --skip-asr
./preprocessing_v2/run_v2.sh --only trajectory
./preprocessing_v2/run_v2.sh --videos L23_V001.mp4,L23_V002.mp4
```

Supported component flags are `--skip-shots`, `--skip-asr`,
`--skip-trajectory`, `--skip-keyframes`, and `--skip-ocr`. `--skip-layer1`
expands to the first three. `--skip-layer2` expands to `--skip-keyframes`.
`--only STAGE` is mutually exclusive with skip flags.

Before execution, the orchestrator resolves dependencies. A skipped producer
is valid only when its required v2 output already exists. For example, running
keyframes with `--skip-layer1` requires the v2 `shots.jsonl`; running OCR with
`--skip-keyframes` requires the v2 Pipeline H frame directory. Missing inputs
cause an actionable error before any model is loaded.

## 5. Configuration

`pipeline_v2.yaml` owns dataset paths, artifact paths, model identifiers,
resource settings, and stage defaults. CLI arguments override configuration
without editing YAML.

`trajectory.yaml` separately owns category routing and trajectory-specific
settings so that category policy can change without touching code:

```yaml
trajectory:
  enabled_prefixes: ["L23"]
  sample_interval_seconds: 0.25
  backend: "ultralytics"
  force_all: false

  classes: ["person", "bicycle", "motorcycle", "car"]
  min_track_points: 8
  min_event_duration_seconds: 1.0

  ultralytics:
    model_path: "yolov8x-oiv7.pt"
    tracker: "botsort.yaml"
    compensate_camera_motion: true

  otvision:
    executable: null
    transform_to_world: false
    reference_points_dir: null
```

Prefix matching uses the portion before `_V`, so `L23_V008` maps to `L23`.
`--trajectory-all` overrides `enabled_prefixes` for an intentional one-off
run. An empty `enabled_prefixes` list disables automatic trajectory routing.

## 6. Layer 1 stages

### 6.1 Shot detection

V2 adapts the existing TransNetV2 runner without modifying it. The threshold
is read from v2 configuration and defaults to the current `0.5`. V2 preserves
the existing output contract:

```json
{"video_id":"L23_V008","shot_id":"L23_V008_000001","start_frame":10,"end_frame":95,"fps":25.0}
```

Configurable thresholds support A/B evaluation; v2 does not claim an improved
threshold without data.

### 6.2 ChunkFormer ASR

V2 retains Silero VAD so every recognized segment has reliable source timing.
Each speech segment is passed to the configured ChunkFormer Vietnamese model,
defaulting to `khanhld/chunkformer-ctc-large-vie`. The output remains compatible
with the existing transcript mapper:

```json
{"video_id":"L23_V008","seg_id":"L23_V008_000003","start_ms":1200,"end_ms":8450,"text":"..."}
```

`asr.backend` accepts `chunkformer` or `phowhisper`. This is an explicit
comparison switch, not an automatic silent fallback. Model initialization or
inference failure fails the ASR stage unless the user configured a fallback.

### 6.3 Trajectory extraction

Trajectory sampling is independent of Pipeline H keyframes. Frames are read at
a fixed interval so velocity and curvature operate on regular observations.
GPU stages are scheduled sequentially by default to avoid VRAM contention;
"parallel" here means a separate data branch, not concurrent GPU processes.

The default backend uses Ultralytics with BoT-SORT. It retains raw detections,
track IDs, normalized bounding-box centers, source frame numbers, and
timestamps. Tracks are filtered by configurable class, confidence, length,
and continuity thresholds.

When enabled, camera-motion compensation estimates a robust background affine
transform between samples and subtracts global motion before event inference.
If there are too few reliable background correspondences, the sample is marked
uncompensated and cannot produce a high-confidence directional event.

Track centers are smoothed before computing displacement, speed change, and
signed curvature. The event vocabulary is deliberately conservative:

- `turn_left`
- `turn_right`
- `accelerating`
- `decelerating`
- `steady_motion`
- `unknown`

V2 does not emit `hard_brake`: image-plane motion alone cannot distinguish
braking from camera movement and perspective changes reliably.

Raw track records are stored separately from derived events, allowing event
thresholds to be retuned without rerunning detection and tracking. Each event
contains `video_id`, source class, track ID, start/end frames and milliseconds,
label, confidence, and quality flags.

## 7. OpenTrafficCam extension

OpenTrafficCam is an optional extension only. It is not installed or used by
the default pipeline.

The optional `otvision` adapter invokes a separately installed OTVision CLI and
converts its track output to the same internal trajectory contract used by the
Ultralytics backend. No OpenTrafficCam source is copied into this repository.
This preserves dependency isolation and makes the GPL-3.0 component explicit.

OTVision pixel-to-world transformation is disabled by default because it needs
reference points and is primarily appropriate for calibrated, fixed traffic
cameras. It may be enabled for compatible footage by providing a reference
points directory. Broadcast footage with camera pan, zoom, or cuts continues
to use normalized image coordinates plus camera-motion compensation.

## 8. Layer 2 and OCR

### 8.1 Pipeline H

The Layer 2 adapter invokes the existing `pipeline_h` entrypoint and directs
its inputs and outputs to v2 paths. It does not alter Pipeline G, Pipeline H,
or the in-progress Layer 2 working-tree changes.

### 8.2 OCR recognition

PaddleOCR remains the detector. Its detected crops are passed to the selected
recognizer:

- `legacy`: current VietOCR/Paddle recognition path.
- `parseq`: a configured PARSeq Vietnamese checkpoint.
- `auto`: PARSeq when its checkpoint exists; otherwise a recorded legacy
  fallback.
- `compare`: run both and retain both candidates for evaluation.

The repository does not currently contain a Vietnamese PARSeq checkpoint.
`parseq.model_path` is therefore required for `parseq` mode. `auto` is the
only mode allowed to fall back, and the fallback is written to the manifest
and logs. The standard v2 OCR output retains the existing per-frame schema;
comparison candidates are written separately.

DeepSolo is not added in this implementation. The source plan requires an
eyeball miss-rate check before replacing Paddle detection, and no repository
evidence currently shows significant Paddle box misses.

## 9. Resume, failures, and provenance

Every stage writes through a temporary file and atomically replaces the final
file. Completion is recorded per video only after durable output is written.

`manifest.json` records:

- input video identity;
- resolved configuration and its hash;
- selected backend and model identifier per stage;
- whether a documented fallback occurred;
- start/end time, status, output path, and error per video;
- tool and schema versions.

By default, a single-video failure is recorded and processing continues with
the next video. `--fail-fast` stops at the first failure. Dependency errors,
invalid configuration, and unavailable required models always fail before
processing starts.

Resume refuses to mix outputs with an incompatible config hash unless
`--force` is supplied. `--force` replaces only the selected videos' v2 records
and never touches v1 output.

## 10. Comparison report

`compare_v1_v2.py` compares outputs without changing either pipeline. It
reports, per video and overall:

- shot count and boundary displacement statistics;
- ASR speech duration coverage, empty rate, and segment count;
- keyframe count and temporal coverage;
- OCR non-empty frame and token coverage;
- trajectory track survival, ID fragmentation, compensation success, event
  count, and unknown-event rate.

WER, OCR word accuracy, and event accuracy are emitted only when corresponding
ground truth is supplied. The report must not present output volume as an
accuracy metric.

## 11. Testing and acceptance

Tests are written before implementation for each behavior.

CPU unit tests cover:

- configuration validation and CLI override precedence;
- prefix routing and `--trajectory-all`;
- skip/only expansion and dependency resolution;
- v1-compatible shots and ASR schemas;
- atomic checkpoint and config-hash resume behavior;
- PARSeq missing-checkpoint behavior and explicit fallback recording;
- synthetic straight, left-turn, right-turn, acceleration, deceleration, and
  insufficient-quality tracks;
- synthetic camera pan compensation;
- conversion from optional OTVision records without importing OTVision.

An integration smoke test uses a generated tiny video and fake model backends,
so it runs without a GPU or network. GPU/model smoke tests are marked separately
and run only when the external checkpoints and hardware are available.

Acceptance requires:

1. Existing v1 tracked files have no diff.
2. CPU tests pass.
3. The fake-backend end-to-end smoke test passes.
4. CLI help documents all layer- and stage-level controls.
5. A one-video GPU smoke run produces all selected v2 outputs and a complete
   manifest when the required external models are present.
6. Output contracts and commands are documented in `preprocessing_v2/README.md`.

## 12. Rollout

Implementation proceeds in test-driven slices: configuration/CLI, manifest
and resume, shot adapter, ASR adapter, trajectory core, optional OTVision
adapter, Pipeline H adapter, OCR adapter, comparison report, then end-to-end
packaging and documentation. No whole-dataset run occurs until the small
validation runs in `preprocessing_layer_expansion_plan.md` have been reviewed.
