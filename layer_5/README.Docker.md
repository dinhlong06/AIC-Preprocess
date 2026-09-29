### Building and running your application

When you're ready, start your application by running:
`docker compose up --build`.

Your application will be available at http://localhost:8000.

### Deploying your application to the cloud

First, build your image, e.g.: `docker build -t myapp .`.
If your cloud uses a different CPU architecture than your development
machine (e.g., you are on a Mac M1 and your cloud provider is amd64),
you'll want to build the image for that platform, e.g.:
`docker build --platform=linux/amd64 -t myapp .`.

Then, push it to your registry, e.g. `docker push myregistry.com/myapp`.

Consult Docker's [getting started](https://docs.docker.com/go/get-started-sharing/)
docs for more detail on building and pushing.

## Cho team thuật toán (đọc dữ liệu, không cài gì lên host)

Không `pip install` gì lên máy host (server dùng chung). Mount thẳng code
`indexdb` (read-only) vào container của bạn và join network `milvus` —
không cần build lại image `layer5`.

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
  <ten-image-cua-ban> python
```

Image của bạn cần Python >= 3.10 (module dùng cú pháp `X | None`) và 3 client
Python thuần (không cần cài `layer_5`):

```
pymongo==4.9.2
elasticsearch==9.0.1
pymilvus==2.6.1
```

Ví dụ dùng:

```python
from indexdb.read import Reader
r = Reader()

# candidate generation — bạn tự encode text/ảnh thành vector, Reader không encode hộ
hits = r.search_vector("beit3", my_1024d_vector, top_k=100)
hits += r.search_ocr("chợ hoa tết", top_k=100)      # match trên ocr_text + ocr_api
hits += r.search_asr("chợ hoa tết", top_k=100)      # match trên transcript (ASR)
hits += r.search_object(["person", "car"], top_k=100)  # match chính xác object_tags
hits += r.search_all("chợ hoa tết", top_k=100)      # match trên content_all (gộp cả 3 nguồn text)
# score của search_vector (Milvus COSINE) và search_ocr/search_asr (Elasticsearch BM25) không cùng thang đo —
# nối rồi sort/slice chung như trên chỉ để demo, việc fuse điểm là của caller
# mỗi hit: {"keyframe_id", "video_id", "shot_id", "frame_idx", "timestamp_ms" (mili giây), "score"}
# keyframe_id CHỈ dùng nội bộ để hydrate (get_frames) — khi nộp bài phải dùng frame_idx
# (số frame gốc trong video, khớp [s,e] của BTC), không phải keyframe_id.
# video_ids=[] (rỗng/None) nghĩa là KHÔNG lọc, không phải "không khớp gì"

# hydrate — lấy đủ metadata + đường dẫn ảnh tuyệt đối để hiển thị/predict
kfs = r.get_frames([h["keyframe_id"] for h in hits[:20]])
kfs[0]["image_path"]   # "/data/layer_2/Keyframe_Extracting/benchmark/pipeline_c/K01_V001/..."

# danh sách shot của 1 video, sắp theo start_ms
shots = r.get_shots("K01_V001")

# ngữ cảnh thời gian quanh một mốc (VQA: "trước/sau khi rời cửa hàng")
segs = r.get_transcript("K01_V001", start_ms=118000, end_ms=130000)
```

**Nếu container của bạn không join được `--network milvus`:** dùng
`10.0.2.3` (gateway của Docker rootless) thay cho tên container, với port
đã remap ở host: `MONGO_URI=mongodb://root:rootpass@10.0.2.3:27018`,
`ELASTICSEARCH_URI=http://10.0.2.3:19201`, `MILVUS_URI=http://10.0.2.3:19531`.
`DATA_ROOT` không đổi theo cách này — nó chỉ phụ thuộc vào việc bạn mount
dataset vào đâu trong container của chính bạn, không liên quan tới network.

## Read API (HTTP)

Nếu không muốn cài `pymongo`/`pymilvus`/`elasticsearch` hay join Docker network, gọi thẳng qua HTTP:

- Base URL: `http://<server-ip>:8021` (cần VPN vào mạng server; đặt `API_KEY` trong shell hoặc `layer_5/.env` trước khi `docker compose up` để service `api` auth request; nếu không đặt, mọi request có auth đều bị 401).
- Mọi request cần header `X-API-Key: <giá trị API_KEY>`, trừ `GET /health`.
- `/search/vector`, `/search/ocr`, `/search/asr`, `/search/object`, `/search/all` nhận thêm field tuỳ chọn trong body JSON: `top_k` (mặc định 100), `video_ids` (lọc theo video, rỗng/None nghĩa là KHÔNG lọc). `/search/ocr` match trên `ocr_text`+`ocr_api`, `/search/asr` match trên `transcript`, `/search/all` match trên `content_all` (gộp cả 3 nguồn text), `/search/object` nhận `labels: list[str]` và match chính xác `object_tags` (keyword, không fuzzy), score là confidence detector thật (max giữa các object cùng nhãn), không phải điểm term-match.
- `image_path` trong response của `/frames` là đường dẫn tuyệt đối tính theo `DATA_ROOT` của server (mặc định `/data`) — không resolve được trực tiếp trên máy caller, chỉ dùng để hiển thị/predict trên máy đã mount cùng dataset.

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

Docs tự sinh (Swagger UI) tại `http://<server-ip>:8021/docs`.

### References
* [Docker's Python guide](https://docs.docker.com/language/python/)