# OCR Module -- AIC2026

Reads Vietnamese text from keyframes on GPU with **DeepSolo detection +
PARSeq-VN recognition**, correcting diacritics in-frame (`ocr/corrector.py`).

> The **production OCR is `../OCR_gemma/gemma_ocr.py`** — Gemma 4 over API
> (UIT + Google AI Studio). This module is the local-GPU path: no API quota,
> reproducible offline. Output schema (`frame_id` = layer 2 `keyframe_id`)
> matches what `layer_5` ingests via `--ocr`.

---

## Project structure

```
OCR/
├── run_ocr.py                # CLI entry point
├── run.sh                    # single run in Docker on the freest GPU
├── run_shards.sh             # worker pool: launch / add / status / release / merge
├── config.yaml               # detector + skip params
├── ocr/
│   ├── deepsolo_engine.py # DeepSolo det + PARSeq-VN rec (Vietnamese charset)
│   ├── corrector.py     # Vietnamese diacritic fix, local CPU
│   ├── frame_skip.py    # blank/blur skip + CLAHE/news-band preprocessing
│   ├── loader.py        # frame folder scanner
│   ├── formatter.py     # JSON output + checkpoint I/O
│   └── pipeline.py      # orchestrator (single + claims worker-pool modes)
└── output/              # run results (git-ignored)
```

Weights (git-ignored, `experiments/deepsolo_parseq/`):
`weights/ic15_... .pth` (DeepSolo R50 detector) +
`vn_scenetext/weights/rec/best-parseq.ckpt` (PARSeq-VN). The
`ocr-deepsolo-parseq` image (detectron2 + strhub) cannot be rebuilt from this
repo — keep the `ocr-deepsolo-parseq-backup.tar` backup.

---

## Usage

```bash
./run.sh                                     # 1 GPU, full container
FRAMES_DIR=/path OUTPUT_DIR=/path ./run.sh   # custom dirs
./run.sh --limit 60                          # extra flags forward to run_ocr.py

NSHARDS=4 ./run_shards.sh                    # background GPUs
./run_shards.sh status                       # progress
./run_shards.sh merge                        # merge claims/done -> output_ocr.json
```

---

## Output

`output_ocr.json`: one record per frame:

```json
{"frame_id": "L21_V001_000000_kf0001", "texts": [{"text": "HTV7", "confidence": 0.93, "box": [..]}]}
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Output already exists` | delete files in `output/` or pass `--overwrite` |
| `CUDA was requested ...` | drop the specific GPU, let the script pick a free one |
| Container dies mid-run | re-run the same command — checkpointed by `frame_id`, auto-resumes |
| Missing image | `docker load -i <backup>/ocr-deepsolo-parseq-backup.tar` |
