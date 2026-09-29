#!/usr/bin/env bash
# Thao tác với stack DB của layer_5: Milvus (vector) + Elasticsearch (BM25) + MongoDB (nguồn sự thật).
#
#   ./run.sh up                        # bật stack, chờ tới khi healthy
#   ./run.sh down                      # tắt, GIỮ data
#   ./run.sh reset                     # tắt + XOÁ sạch data (hỏi xác nhận)
#   ./run.sh status                    # health 3 DB + số liệu đã nạp
#   ./run.sh logs [service]            # xem log, mặc định elasticsearch
#   ./run.sh test                      # pytest (cần stack đang chạy)
#   ./run.sh init                      # tạo index/collection rỗng (1 lần, sau reset)
#   ./run.sh ingest-batch1 [cờ...]     # nạp dataset_batch1 (BTC) dùng keyframes.jsonl + SigLIP2 batch1,
#                                      # cờ thêm truyền thẳng cho indexdb.ingest_batch1 (--keyframes-dir/--ocr/--siglip2-dir ghi đè được):
#         ./run.sh ingest-batch1 --videos L21_V001      # 1 video
#         ./run.sh ingest-batch1                        # toàn bộ dataset_batch1
#   ./run.sh ingest-batch2 [cờ...]     # nạp output_batch2 (run_batch2.sh): Gemma OCR + SigLIP2, cùng code ingest_batch1
#   ./run.sh shell                     # python REPL đã nối sẵn 3 DB
#
# Mọi lệnh chạy code đều tự `docker build` trước, nên sửa code xong chạy được ngay.
# Port host đã đổi trong docker-compose.override.yaml vì máy này dùng chung.
# Service `api` (HTTP) chỉ auth đúng nếu bạn export API_KEY trước khi `docker compose up`;
# nếu không, mọi request bị 401 (fail-closed) chứ compose không báo lỗi.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

IMAGE=layer5
DB_ENV=(-e MONGO_URI=mongodb://root:rootpass@mongodb:27017
        -e ELASTICSEARCH_URI=http://elasticsearch:9200
        -e MILVUS_URI=http://standalone:19530
        -e DATA_ROOT=/data)

# Chạy python trong container: nối network compose + gắn thư mục dữ liệu các layer
# vào /data (chỉ đọc). Tên /data phải khớp `--root /data` bên dưới.
py() {
    docker build -q -t "$IMAGE" . > /dev/null
    # -i để `py - <<'PY'` đọc được script qua stdin
    docker run --rm -i --network milvus -v "$PWD/..:/data:ro" "${DB_ENV[@]}" "$IMAGE" python "$@"
}

case "${1:-}" in
up)
    docker compose up -d
    echo "== Chờ healthy =="
    for i in $(seq 1 30); do
        sleep 5
        es=$(curl -sf --max-time 3 http://127.0.0.1:19201/_cluster/health > /dev/null && echo ok || true)
        mv=$(curl -sf --max-time 3 http://127.0.0.1:19091/healthz > /dev/null && echo ok || true)
        [ "$es$mv" = "okok" ] && { echo "   sẵn sàng sau $((i*5))s"; break; }
        echo "   [$((i*5))s] ES=${es:-chưa} Milvus=${mv:-chưa}"
    done
    echo "Attu (Milvus UI): http://localhost:8010    Elasticvue (ES UI): http://localhost:8084"
    ;;
down)
    docker compose down
    ;;
reset)
    read -rp "XOÁ SẠCH toàn bộ data trong Milvus/Elastic/Mongo? [y/N] " ans
    [ "$ans" = "y" ] || { echo "huỷ."; exit 0; }
    docker compose down -v
    ;;
logs)
    docker compose logs -f --tail 50 "${2:-elasticsearch}"
    ;;
status)
    docker compose ps -a --format "table {{.Name}}\t{{.Status}}"
    py - <<'PY'
from indexdb.config import Config
from indexdb.mongo import MongoStore
from indexdb.elastic import ElasticStore
from indexdb.milvus import MilvusStore
cfg = Config.from_env()
s, es, mv = MongoStore(cfg), ElasticStore(cfg), MilvusStore(cfg)
print("\nMONGO   videos=%d shots=%d frames=%d segments=%d" % (
    s.videos.count_documents({}), s.shots.count_documents({}),
    s.frames.count_documents({}), s.transcript_segments.count_documents({})))
if es.client.indices.exists(index=es.index_name):
    es.client.indices.refresh(index=es.index_name)
    print("ELASTIC docs=%d" % es.client.count(index=es.index_name)["count"])
for name in mv.collection_names().values():
    if mv.client.has_collection(name):
        print("MILVUS  %-20s rows=%s" % (name, mv.client.get_collection_stats(name)["row_count"]))
done = [d["_id"] for d in s.ingest_status.find({"steps.elastic.status": "done"}, {"_id": 1})]
print("\nĐã nạp xong %d video: %s" % (len(done), ", ".join(sorted(done)[:5]) + (" ..." if len(done) > 5 else "")))
PY
    ;;
test)
    docker build -q -t "$IMAGE" . > /dev/null
    docker run --rm --network milvus "${DB_ENV[@]}" "$IMAGE" \
        python -m pytest -q -p no:cacheprovider --basetemp=/tmp/pt
    ;;
init)
    py -m indexdb.init_stores
    ;;
ingest-batch1)
    shift
    py -m indexdb.ingest_batch1 --root /data/dataset_batch1 \
        --keyframes-dir /data/layer_2/Keyframe_Extracting/benchmark_batch1_v2/pipeline_g \
        --ocr /data/layer_3/OCR_gemma/gemma_ocr_batch1.jsonl \
        --captions '/data/layer_3/OCR_gemma/*_caption_batch1.jsonl' \
        --siglip2-dir /data/siglip/artifacts/siglip_batch1_v2 "$@"
    ;;
ingest-batch2)
    shift
    py -m indexdb.ingest_batch1 --batch batch2 --root /data/output_batch2 \
        --keyframes-dir /data/output_batch2/keyframes/pipeline_h \
        --ocr '/data/output_batch2/gemma_ocr*.jsonl' \
        --shots /data/output_batch2/shots.jsonl \
        --transcripts /data/output_batch2/shot_transcripts.jsonl \
        --asr-segments /data/output_batch2/asr/whisper.jsonl \
        --siglip2-dir /data/output_batch2/siglip "$@"
    ;;
shell)
    docker build -q -t "$IMAGE" . > /dev/null
    docker run --rm -it --network milvus -v "$PWD/..:/data:ro" "${DB_ENV[@]}" "$IMAGE" python -i -c "
from indexdb.config import Config
from indexdb.mongo import MongoStore
from indexdb.elastic import ElasticStore
from indexdb.milvus import MilvusStore
from indexdb.read import Reader
cfg = Config.from_env(); s = MongoStore(cfg); es = ElasticStore(cfg); mv = MilvusStore(cfg)
r = Reader(cfg)
print('Sẵn: r (Reader), s (mongo), es (elastic), mv (milvus). VD: r.search_ocr(\"chợ hoa\")')"
    ;;
*)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac
