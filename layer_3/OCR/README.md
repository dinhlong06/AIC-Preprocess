# OCR Module -- AIC2026

Reads Vietnamese text from keyframes on GPU with **PaddleOCR PP-OCRv6**,
correcting diacritics in-frame (`ocr/corrector.py`).

> The **production OCR is `../OCR_gemma/gemma_ocr.py`** — Gemma 4 over API
> (UIT + Google AI Studio). This module produces Paddle/VietOCR output for
> comparison.

---

## Project structure

```
OCR/
├── run_paddle.py                 # CLI entry point
├── run_paddle.sh                 # build+run in Docker, picks the freest GPU
├── run_paddle_batch1_shards.sh   # worker pool: add / status / release / merge
├── config.yaml                   # stage-1 params
├── requirements.txt
├── Dockerfile
├── ocr/
│   ├── paddle_engine.py   # PaddleOCR det + VietOCR/Paddle rec
│   ├── corrector.py       # Vietnamese diacritic fix, local CPU
│   ├── frame_skip.py      # blank/blur skip + CLAHE/news-band preprocessing
│   ├── loader.py          # frame folder scanner
│   ├── formatter.py       # JSON output + checkpoint I/O
│   └── pipeline.py        # orchestrator
└── output/                # run results (git-ignored)
```

---

## Usage

```bash
./run_paddle.sh                                  # 1 GPU, full container
FRAMES_DIR=/path ./run_paddle.sh                 # different keyframe dir
OUTPUT_DIR=/path ./run_paddle.sh                 # write JSON elsewhere
./run_paddle.sh --limit 60                       # extra flags forward to run_paddle.py

NSHARDS=4 ./run_paddle_batch1_shards.sh          # multiple background GPUs
./run_paddle_batch1_shards.sh status             # check progress
./run_paddle_batch1_shards.sh merge              # merge claims/done -> output
```

---

## Output

`output/` holds JSON, one record per frame:

```json
{"frame_id": "L21_V001_000000_kf0001", "boxes": [{"text": "HTV7", "confidence": 0.93, "bbox": [..]}]}
```

`frame_id` matches layer 2's `keyframe_id` so `layer_5` can join.
`output_paddle_origin.json` is Paddle's own recognizer output, for comparison.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Output already exists` | delete files in `output/` or pass `--overwrite` |
| `CUDA was requested ...` | drop the specific GPU, let the script pick a free one |
| Container dies mid-run | re-run the same command — checkpointed by `frame_id`, auto-resumes |
