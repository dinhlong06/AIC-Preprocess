# OCR Module -- AIC2026

Reads Vietnamese text from keyframes on GPU with **DeepSolo detection +
PARSeq-VN recognition**, correcting diacritics in-frame (`ocr/corrector.py`).

> The API path is `../OCR_gemma/gemma_ocr.py` — Gemma 4 over API (UIT +
> Google AI Studio). This module is the local-GPU production path: no API
> quota, reproducible offline. Output schema (`frame_id` = layer 2
> `keyframe_id`) matches what `layer_5` ingests via `--ocr`.

Measured on 30 mixed frames vs Gemma: median char-similarity 0.95,
coverage 29/30 vs 24/30, ~2.9 fps (det ~260ms + rec ~50ms per frame).

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
`weights/ic15_res50_finetune_synth-tt-mlt-13-15-textocr.pth` (DeepSolo R50
detector) + `vn_scenetext/weights/rec/best-parseq.ckpt` (PARSeq-VN, Vietnamese
charset). The `ocr-deepsolo-parseq` image (detectron2 + strhub) cannot be
rebuilt from this repo — keep the `ocr-deepsolo-parseq-backup.tar` backup
at repo root.

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
| Missing image | `docker load -i $PROJECT_ROOT/ocr-deepsolo-parseq-backup.tar` (tarball at repo root) |
