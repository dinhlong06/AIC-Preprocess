# Layer 3 — OCR

Reads text out of the keyframe images chosen by layer 2. Two paths:

```
keyframes ──► OCR_gemma (Gemma 4 over API) ──► gemma_ocr_batch1.jsonl      (production)
         └──► OCR (PaddleOCR, local GPU) ────► output_vietocr.json        (comparison)
```

- `OCR_gemma/gemma_ocr.py`: Gemma 4 27B/31B over HTTP (UIT endpoint + Google
  AI Studio), one row per frame `{"frame_id", "path", "text", "finish",
  "model"}` where `frame_id` = `keyframe_id`. Same script with `--task caption`
  produces captions. Retries 429/5xx with backoff.
- `OCR/`: PaddleOCR PP-OCRv6 detection + VietOCR/Paddle recognition, diacritics
  corrected locally per frame ([details](OCR/README.md)). Alternative engine:
  DeepSolo detection + PARSeq-VN recognition via `OCR/run_deepsolo.sh`
  (`--engine deepsolo_parseq`) — better on the 60-frame bottom-band GT
  (R 93.6% vs 75.2% with diacritics).

## Setup and run

```bash
cd OCR_gemma
python gemma_ocr.py --task ocr --frames <keyframes> --out gemma_ocr_batch1.jsonl
python gemma_ocr.py --task caption --frames <keyframes> --out captions.jsonl --skip "*.jsonl"
# keys: GEMINI_API_KEY(_2..4) for AI Studio, LLM_API_KEY for UIT
# UIT needs a private CA: UIT_CA_BUNDLE=/path/bundle.pem (unset = system trust store)

cd ../OCR
./run_paddle.sh                                        # 1 GPU
FRAMES_DIR=/path OUTPUT_DIR=/path ./run_paddle.sh      # custom dirs
NSHARDS=4 ./run_paddle_batch1_shards.sh                # background GPUs (add/status/merge)
```

## Folder structure

```
layer_3/
├── OCR_gemma/
│   ├── gemma_ocr.py           # Gemma 4 OCR + caption over API (--task ocr|caption)
│   └── run_caption_batch1.sh  # orchestrates the caption shards
└── OCR/
    ├── run_paddle.py          # CLI entry point
    ├── run_paddle.sh          # build + run in Docker on the freest GPU
    ├── run_paddle_batch1_shards.sh  # worker pool: add / status / release / merge
    ├── config.yaml            # params
    ├── ocr/                   # paddle_engine, corrector, frame_skip, loader, formatter, pipeline
    ├── Dockerfile
    └── requirements.txt
```
