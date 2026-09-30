# Layer 5 — Indexing + Search API

Loads batch1/batch2 preprocessing outputs into three stores and serves search
over HTTP:

```
layer_1 shots + ASR ──┐
layer_2 keyframes ────┼──► ingest-batch1/2 ──► MongoDB (source of truth)
layer_3 Gemma OCR ────┤                        ├──► Elasticsearch (BM25 text)
siglip embeddings ────┘                        └──► Milvus (vectors)
                                                          │
                                                          ▼
                                              api (FastAPI, :8000): /search/*, /frames, /shots, /transcript
```

Stores: `indexdb/mongo.py`, `elastic.py`, `milvus.py`. Write path: `writers.py`
→ `elastic_indexer.py` + `milvus_indexer.py`. Read path: `indexdb/read.py`
(`Reader`), wrapped by `api/main.py`.

## Setup and run

```bash
cd layer_5
./run.sh up              # start the stack, wait until healthy
./run.sh init            # create empty indexes/collections (once, after reset)
./run.sh ingest-batch1   # load dataset_batch1
./run.sh ingest-batch2   # load output_batch2
```

```bash
./run.sh ingest-batch1 --videos L21_V001   # one video
./run.sh ingest-batch1 --resume            # only videos not finished
./run.sh status                            # health of 3 DBs + ingest counts
./run.sh test                              # pytest (needs the stack running)
./run.sh down                              # stop, KEEP data
./run.sh reset                             # stop + WIPE all data (asks first)
```

Extra flags pass straight to the ingester
(`--keyframes-dir`, `--ocr`, `--captions`, `--siglip2-dir`, `--asr-segments`).

The `api` service authenticates with `X-API-Key`; export `API_KEY` before
`docker compose up`, otherwise every authenticated request gets a 401.
