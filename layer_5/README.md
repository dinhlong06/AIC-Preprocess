# Layer 5 — Indexing + Search API

## For the algorithm team (read the data, install nothing on the host)

Do **not** `pip install` anything on the host (it is a shared server). Mount the
`indexdb` code read-only straight into your own container and join the `milvus`
network — no need to rebuild the `layer5` image.

```bash
docker run --rm -it --gpus all \
  --network milvus \
  -v /workingspace_aiclub/WorkingSpace/Personal/vannk/Ai_challange_2026/layer_5:/opt/layer5:ro \
  -v /workingspace_aiclub/WorkingSpace/Personal/vannk/Ai_challange_2026:/data:ro \
  -e PYTHONPATH=/opt/layer5 \
  -e MONGO_URI=mongodb://root:rootpass@mongodb:27017 \
  -e ELASTICSEARCH_URI=http://elasticsearch:9200 \
  -e MILVUS_URI=http://standalone:19530 \
  -e DATA_ROOT=/data \
  <your-image> python
```

Your image needs Python >= 3.10 (the module uses `X | None` syntax) and 3 plain
Python clients (installing `layer_5` itself is not required):

```
pymongo==4.9.2
elasticsearch==9.0.1
pymilvus==2.6.1
```

Usage example:

```python
from indexdb.read import Reader
r = Reader()

# candidate generation — you encode text/images into vectors yourself, Reader does not
hits = r.search_vector("beit3", my_1024d_vector, top_k=100)
hits += r.search_ocr("chợ hoa tết", top_k=100)      # matches ocr_text + ocr_api
hits += r.search_asr("chợ hoa tết", top_k=100)      # matches transcript (ASR)
hits += r.search_object(["person", "car"], top_k=100)  # exact match on object_tags
hits += r.search_all("chợ hoa tết", top_k=100)      # matches content_all (all 3 text sources)
# search_vector (Milvus COSINE) and search_ocr/search_asr (Elasticsearch BM25) do not share
# a scale — concatenating and sorting them together as above is demo only; fusing the scores
# is the caller's job.
# each hit: {"keyframe_id", "video_id", "shot_id", "frame_idx", "timestamp_ms" (ms), "score"}
# keyframe_id is for INTERNAL hydration only (get_frames) — a submission must use frame_idx
# (the original frame number in the video, matching the BTC [s,e] range), not keyframe_id.
# video_ids=[] (empty/None) means DO NOT filter, not "matches nothing"

# hydrate — full metadata + absolute image paths for display/prediction
kfs = r.get_frames([h["keyframe_id"] for h in hits[:20]])
kfs[0]["image_path"]   # e.g. "/data/layer_2/Keyframe_Extracting/benchmark_batch1_v2/pipeline_g/L21_V001/..."

# shots of one video, sorted by start_ms
shots = r.get_shots("K01_V001")

# time context around a moment (VQA: "before/after leaving the shop")
segs = r.get_transcript("K01_V001", start_ms=118000, end_ms=130000)
```

**If your container cannot join `--network milvus`:** use `10.0.2.3` (the
rootless Docker gateway) instead of the container names, with the remapped host
ports: `MONGO_URI=mongodb://root:rootpass@10.0.2.3:27018`,
`ELASTICSEARCH_URI=http://10.0.2.3:19201`, `MILVUS_URI=http://10.0.2.3:19531`.
`DATA_ROOT` does not change with this approach — it only depends on where *you*
mount the dataset in *your* container, nothing to do with the network.

## Read API (HTTP)

If you would rather not install `pymongo`/`pymilvus`/`elasticsearch` or join the
Docker network, call it over HTTP:

- Base URL: `http://<server-ip>:8021` (needs VPN into the server network; export
  `API_KEY` in the shell or put it in `layer_5/.env` before `docker compose up` so
  the `api` service authenticates requests — without it every authenticated
  request gets a 401).
- Every request needs the header `X-API-Key: <API_KEY value>`, except
  `GET /health`.
- `/search/vector`, `/search/ocr`, `/search/asr`, `/search/object`, `/search/all`
  accept these optional fields in the JSON body: `top_k` (default 100),
  `video_ids` (filter by video; empty/None means DO NOT filter). `/search/ocr`
  matches `ocr_text`+`ocr_api`, `/search/asr` matches `transcript`, `/search/all`
  matches `content_all` (all 3 text sources combined), `/search/object` takes
  `labels: list[str]` and matches `object_tags` exactly (keyword, not fuzzy) with
  the score being the real detector confidence (max across objects sharing a
  label), not a term-match score.
- `image_path` in `/frames` responses is an absolute path resolved against the
  server's `DATA_ROOT` (default `/data`) — it cannot be resolved directly on the
  caller's machine; it is only usable for display/prediction on a machine that
  mounted the same dataset.

```bash
curl http://<server-ip>:8021/health

curl -X POST http://<server-ip>:8021/search/vector \
  -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '{"model": "beit3", "vector": [0.01, "..."], "top_k": 20}'

curl -X POST http://<server-ip>:8021/search/ocr \
  -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '{"query": "một khu chợ hoa", "top_k": 20}'

curl -X POST http://<server-ip>:8021/search/asr \
  -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '{"query": "một khu chợ hoa", "top_k": 20}'

curl -X POST http://<server-ip>:8021/search/object \
  -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '{"labels": ["person", "car"], "top_k": 20}'

curl -X POST http://<server-ip>:8021/search/all \
  -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '{"query": "một khu chợ hoa", "top_k": 20}'

curl -X POST http://<server-ip>:8021/frames \
  -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '{"ids": ["v001_f0001", "v001_f0002"]}'

curl "http://<server-ip>:8021/shots/v001" -H "X-API-Key: <key>"

curl "http://<server-ip>:8021/transcript/v001?start_ms=0&end_ms=5000" -H "X-API-Key: <key>"
```

Auto-generated docs (Swagger UI) at `http://<server-ip>:8021/docs`.
