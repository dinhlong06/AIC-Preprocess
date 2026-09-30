# Layer 3 — OCR

Reads text out of the keyframe images chosen by layer 2. Two paths:

```
keyframes ──► OCR_gemma (Gemma 4 over API) ──► gemma_ocr_batch1.jsonl      (API path)
         └──► OCR (DeepSolo+PARSeq, GPU) ───► output_ocr.json             (local production path)
```

- `OCR_gemma/gemma_ocr.py`: Gemma 4 27B/31B over HTTP (UIT endpoint + Google
  AI Studio), one row per frame `{"frame_id", "path", "text", "finish",
  "model"}` where `frame_id` = `keyframe_id`. Same script with `--task caption`
  produces captions. Retries 429/5xx with backoff.
- `OCR/`: DeepSolo detection + PARSeq-VN recognition on local GPU, diacritics
  corrected per frame ([details](OCR/README.md)). No API quota, reproducible
  offline; output schema matches what `layer_5` ingests via `--ocr`.

## Setup and run

```bash
cd OCR_gemma
python gemma_ocr.py --task ocr --frames <keyframes> --out gemma_ocr_batch1.jsonl
python gemma_ocr.py --task caption --frames <keyframes> --out captions.jsonl --skip "*.jsonl"
# keys: GEMINI_API_KEY(_2..4) for AI Studio, LLM_API_KEY for UIT
# UIT needs a private CA: UIT_CA_BUNDLE=/path/bundle.pem (unset = system trust store)

cd ../OCR
./run.sh                                     # 1 GPU
FRAMES_DIR=/path OUTPUT_DIR=/path ./run.sh   # custom dirs
NSHARDS=4 ./run_shards.sh                    # background GPUs (add/status/merge)
```

## Folder structure

```
layer_3/
├── OCR_gemma/
│   ├── gemma_ocr.py           # Gemma 4 OCR + caption over API (--task ocr|caption)
│   └── run_caption_batch1.sh  # orchestrates the caption shards
└── OCR/
    ├── run_ocr.py             # CLI entry point
    ├── run.sh                 # single run in Docker on the freest GPU
    ├── run_shards.sh          # worker pool: launch / add / status / release / merge
    ├── config.yaml            # detector + skip params
    ├── ocr/                   # deepsolo_engine, corrector, frame_skip, loader, formatter, pipeline
    └── output/                # run results (git-ignored)
```
