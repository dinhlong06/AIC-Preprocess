# OCR Module -- AIC2026

Extract Vietnamese text from video frames via a 2-stage hybrid pipeline: **PaddleOCR PP-OCRv6** (GPU)
reads the letters, **ising-calibration** (VLM, NVIDIA API) restores the diacritics. Measured 9/9 correct
banner sentences on a 20-frame validation set vs 5/9 for an earlier local Paddle+VietOCR pipeline
(retired -- see memory `ocr-hybrid-paddle-plus-vlm` for the full comparison history).

---

## Project Structure

```
OCR/
├── run_paddle.py         # CLI entry point, stage 1
├── run_paddle.sh          # build+run stage 1 in Docker, picks the freest GPU
├── run_correct.py        # CLI entry point, stage 2 (host, no Docker needed)
├── run_api.sh              # stage 2 by default (needs output_vietocr.json already there); --full runs stage 1 first
├── config.yaml            # All configurable settings, both stages
├── requirements.txt
├── Dockerfile              # GPU container, stage 1 only
├── .env                     # NVIDIA_API_KEY (gitignored, see Setup below)
├── ocr/
│   ├── __init__.py
│   ├── paddle_engine.py   # PaddleOCR wrapper (stage 1)
│   ├── corrector.py       # ising-calibration correction (stage 2)
│   ├── frame_skip.py      # blank/blur/pHash skip + CLAHE/news-band preprocessing
│   ├── loader.py          # frame folder scanner
│   ├── formatter.py       # JSON output + checkpoint I/O
│   └── pipeline.py        # orchestrators for both stages
├── experiments/            # abandoned approaches, kept for reference -- see memory
└── output/
```

**Why two stages, not one pipeline:** they run in different environments. Stage 1 needs the
GPU/Paddle Docker image; stage 2 only needs `requests`+`pyyaml` and a network path to NVIDIA's API,
so it runs on the host and is rate-limited (see Troubleshooting).

---

## Usage

### Stage 2 only (default) -- correction on an existing `output_vietocr.json`

```bash
export NVIDIA_API_KEY=nvapi-...
./run_api.sh                           # requires output/output_vietocr.json to already exist
```

### Both stages, one command (`--full`)

```bash
export NVIDIA_API_KEY=nvapi-...
./run_api.sh --full                          # full dataset, default pipeline_c
PIPELINE=pipeline_a ./run_api.sh --full --limit 60   # extra flags forward to run_paddle.sh only
```

### Stage 1 -- PaddleOCR (GPU, inside Docker)

```bash
./run_paddle.sh                       # builds the image, picks the freest GPU, runs
PIPELINE=pipeline_a ./run_paddle.sh   # different keyframe source pipeline
FRAMES_DIR=/other/path ./run_paddle.sh --limit 60   # arbitrary frames dir + extra flags forwarded
```

Input defaults to `../../layer_2/Keyframe_Extracting/benchmark/$PIPELINE` (default `pipeline_c`) --
`loader.py` scans it recursively so every video under that pipeline is picked up in one run (`frame_id`s
are globally unique, no collision risk). Output goes to `output/output_vietocr.json (plus output_paddle_origin.json, Paddle's own uncorrected recognizer, kept only for comparison)`. Model weights are
cached in `cache/paddlex/` across runs (persisted via volume mount, not re-downloaded each time).

### Stage 2 -- ising-calibration correction (host, no Docker)

```bash
export NVIDIA_API_KEY=nvapi-...
python run_correct.py   # reads correct.frames_dir/paddle_output/output_file/workers from config.yaml
```

Runs `correct.workers` (default 4) concurrent requests via a thread pool. An isolated 60-frame benchmark
measured 6 as the sweet spot (~2.62 req/s), but a real 714-frame overnight run stalled visibly at 6 with the
key already warmed up from a day of testing -- lowered to 4 based on that live observation; more workers is
also *slower* past the ceiling, not faster, due to 429 retry backoff (see Troubleshooting). Consecutive
frames with byte-identical raw text (a banner sitting unchanged for seconds)
are grouped so only the first frame of each run pays for an API call, then distributed across the pool --
result order in the output JSON always matches the input order regardless of completion order.

### Other CLI flags

| Flag | Stage | Description |
|---|---|---|
| `--config PATH` | both | Path to YAML config file (default `config.yaml`) |
| `--input DIR` / `--output FILE` / `--limit N` | 1 | Override `paddle.*` |
| `--frames DIR` / `--paddle-output FILE` / `--output FILE` / `--api-key KEY` | 2 | Override `correct.*` |

---

## Output Format

Per-frame JSON array:

```json
[
  { "frame_id": "K01_V001_000025_kf0001",
    "texts": [{ "text": "Vụ sạt lở nghiêm trọng tại Hà Giang", "confidence": 0.9823 }] }
]
```

---

## Pipeline Architecture

```
Frame on disk
     |
     v
[stage 1, GPU, docker]
frame_skip: blank / blur / pHash near-dup   <- reuse previous frame's result on a hit
     |
PaddleOCR PP-OCRv6 (marks often wrong/dropped, base letters usually right -- kept as-is, stage 2 restores them)
     |
output_vietocr.json  (+ output_paddle_origin.json, same schema, Paddle's own recognizer)  {frame_id, texts:[{text, confidence}]}
     |
     v
[stage 2, host, rate-limited API]
raw text == previous frame's raw text?  <- reuse previous correction on a hit
     |
ising-calibration VLM: substitution-only correction, never appends new lines
     |
output_hybrid.json
```

---

## Configuration (`config.yaml`)

```yaml
paddle:
  input_dir / output_file / output_file_paddle_origin / limit / checkpoint_every
  preprocess: false   # CLAHE+news-band crop -- TESTED HARMFUL on this dataset, see Troubleshooting
  skip:
    blank_brightness_threshold: 15
    blur_threshold: 50.0
    similarity_hamming_threshold: 8
  ocr:
    lang / ocr_version / unclip_ratio

correct:
  frames_dir / paddle_output / output_file / checkpoint_every / workers
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `NVIDIA_API_KEY` not set | `export NVIDIA_API_KEY=nvapi-...` or `--api-key` |
| Stage 2 hits 429 / slow | `correct()` retries 429s with backoff automatically (up to 8 attempts). Default `correct.workers` is 4 (lowered from an earlier 6 after a real overnight run stalled) -- try lowering further if still bursty, don't raise without re-measuring on a fresh key, see memory `ising-calibration-api-rate-limit` |
| Diacritics still wrong after stage 2 | Check `temperature: 0` in `ocr/corrector.py` -- non-zero made prompt tuning unreproducible |
| Frames losing the channel logo/timestamp | `paddle.preprocess: true`? Turn it off -- measured to corrupt 48/60 frames on this dataset (crop tuned for a different show's layout) |
| Concatenated words (`TINCHINH`) not splitting | Not a detector-tuning problem -- `paddle.ocr.unclip_ratio` measured to NOT fix this (words are genuinely touching in the source image); rely on stage 2's correction instead |
| `PaddleOCR not installed` | Build via `Dockerfile` (GPU) -- do not `pip install` on the bare shared host, see memory `shared-host-no-pip-install` |

---

## Notes

- `frame_id` is the file stem: `K01_V001_000025_kf0001.jpg` -> `"K01_V001_000025_kf0001"`.
- `texts: []` for blank/blurry/skipped frames or frames with no detected boxes.
- Both stages write output atomically (`.tmp` then `replace()`) and resume via `checkpoint_every`.
- See `../../` memory notes (`ocr-hybrid-paddle-plus-vlm`, `ocr-v2-nemotron-comparison`,
  `ising-calibration-api-rate-limit`) for the full experimentation history behind these choices --
  `experiments/` holds the abandoned code, not just the conclusions.
