# Layer 3 — OCR

Reads text out of the keyframe images chosen by layer 2.

## Production path: `OCR_gemma/` (API)

**Gemma 4 27B/31B over HTTP** — the only producer of the OCR files that
layer 5 ingests:

| Batch | Output | Consumed by |
|---|---|---|
| batch1 | `OCR_gemma/gemma_ocr_batch1.jsonl` (195,823 frames) | `run.sh ingest-batch1` |
| batch2 | `output_batch2/gemma_ocr*.jsonl` (~163k frames) | `run.sh ingest-batch2` |

Each row: `{"frame_id", "path", "text", "finish", "model"}` where `frame_id` =
`keyframe_id` of layer 2. Captions come from the same script with `--task caption`.

- Keys: `GEMINI_API_KEY(_2.._4)` for Google AI Studio (`ais*` models), `LLM_API_KEY`
  for the UIT-hosted endpoint. UIT needs a private CA: point `UIT_CA_BUNDLE` at the
  bundle file; unset falls back to the system trust store.
- Retries on 429/5xx with measured backoff (see module docstring).

Docs: [OCR_gemma/](OCR_gemma/) — script is self-documenting (docstring has the
production commands and the token-budget measurements that justify them).

## Comparison path: `OCR/` (local GPU)

**PaddleOCR PP-OCRv6 detection + VietOCR/Paddle recognition**, with Vietnamese
diacritics corrected locally per frame. Kept to regenerate baseline outputs and
for A/B comparison — superseded by Gemma on news-style frames.

Docs: [OCR/README.md](OCR/README.md)

## Retired approaches

Removed from the repo; rationale in git history:

- **Qwen3-VL as OCR** — ~4x slower, repetition loops, 72% line recall.
- **Host vLLM stage fixing merged-text boxes** — affected only ~0.55% of boxes.
- **DeepSolo + PARSeq** — better recall (93.6% vs 75.2%) but its output had no
  consumer and the Docker build context was never committable.

## Downstream

`layer_5/indexdb/ingest_batch1.py` joins OCR text onto frames by `keyframe_id`
and indexes it as `ocr_text` in Elasticsearch (plus `ocr_api` for any corrected
variant).
