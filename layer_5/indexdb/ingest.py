"""Nạp output các layer vào Mongo + Milvus + Elastic.

    python -m indexdb.ingest --root /path/Ai_challange_2026 [--videos K01_V001] [--resume|--purge]

Đọc thẳng format gốc của từng layer, không qua bước staging trung gian.
"""
import argparse
import json
import os

from indexdb.config import Config
from indexdb.elastic import ElasticStore
from indexdb.elastic_indexer import ElasticIndexer
from indexdb.milvus import MilvusStore
from indexdb.milvus_indexer import MilvusIndexer
from indexdb.mongo import MongoStore
from indexdb.writers import MongoWriter

MIN_CONF = 0.3


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _group_by_video(path):
    out = {}
    for rec in _jsonl(path):
        out.setdefault(rec["video_id"], []).append(rec)
    return out


def _load_ocr(path):
    with open(path, encoding="utf-8") as f:
        return {r["frame_id"]: " ".join(t["text"] for t in r["texts"] if t["confidence"] >= MIN_CONF)
                for r in json.load(f)}


def _load_objects(path):
    # Layer 3b (object detection) đã bị bỏ, không còn detections.json nào được sinh ra.
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for r in data:
        objects = {}
        for o in r["objects"]:
            if o["confidence"] >= MIN_CONF:
                objects[o["label"]] = max(objects.get(o["label"], 0.0), o["confidence"])
        out[r["frame_id"]] = objects
    return out


def _load_recap(path):
    return {r["keyframe_id"]: r["caption"] for r in _jsonl(path)}


def _purge(video_id, store, es):
    store.frames.delete_many({"video_id": video_id})
    store.shots.delete_many({"video_id": video_id})
    store.transcript_segments.delete_many({"video_id": video_id})
    store.ingest_status.delete_one({"_id": video_id})
    es.client.delete_by_query(index=es.index_name, refresh=True,
                              body={"query": {"term": {"video_id": video_id}}})


def main():
    ap = argparse.ArgumentParser(prog="indexdb.ingest")
    ap.add_argument("--root", required=True, help="thư mục gốc Ai_challange_2026")
    ap.add_argument("--pipeline", default="pipeline_c")
    ap.add_argument("--videos", nargs="*")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="bỏ qua video đã nạp xong")
    mode.add_argument("--purge", action="store_true",
                      help="xoá sạch video trong Mongo/Elastic rồi nạp lại (khi layer_2 chạy lại)")
    args = ap.parse_args()

    kf_root = os.path.join(args.root, "layer_2/Keyframe_Extracting/benchmark", args.pipeline)
    siglip_dir = os.path.join(args.root, "recap_siglip/artifacts/siglip")
    recap_dir = os.path.join(args.root, "recap_siglip/artifacts/recap")

    shots = _group_by_video(os.path.join(args.root, "layer_1/shots.jsonl"))
    segs = _group_by_video(os.path.join(args.root, "layer_1/whisper.jsonl"))
    transcripts = {r["shot_id"]: r["text"]
                   for r in _jsonl(os.path.join(args.root, "layer_2/shot_transcript/shot_transcripts.jsonl"))}
    ocr = _load_ocr(os.path.join(args.root, "layer_3/OCR/output/output_vietocr.json"))
    ocr_api = _load_ocr(os.path.join(args.root, "layer_3/OCR/output/output_hybrid.json"))
    objects = _load_objects(os.path.join(args.root, "layer_3/ObjectDetection/output/detections.json"))

    cfg = Config.from_env()
    store = MongoStore(cfg)
    writer = MongoWriter(store)
    es_idx = ElasticIndexer(store, ElasticStore(cfg))
    mv_idx = MilvusIndexer(store, MilvusStore(cfg))

    for video_id in args.videos or sorted(os.listdir(kf_root)):
        if args.resume and store.ingest_status.find_one({"_id": video_id, "steps.elastic.status": "done"}):
            print(f"{video_id}: bỏ qua (đã xong)")
            continue
        if args.purge:
            _purge(video_id, store, es_idx.es)

        kf_dir = os.path.join(kf_root, video_id)
        keyframes = _jsonl(os.path.join(kf_dir, "keyframes.jsonl"))
        recap_path = os.path.join(recap_dir, f"{video_id}.jsonl")
        recap = _load_recap(recap_path) if os.path.exists(recap_path) else {}

        for rec in shots[video_id]:
            writer.upsert_shot(rec, transcript=transcripts.get(rec["shot_id"], ""))
        for rec in segs.get(video_id, []):
            writer.upsert_seg(rec)
        for rec in keyframes:
            kfid = rec["keyframe_id"]
            rec["batch"] = "batch2"
            # lưu tương đối so với gốc dataset: /data chỉ là tên mount của lần chạy
            # này, bên tiêu thụ (UI...) tự ghép gốc của nó vào.
            rec["image_path"] = os.path.relpath(os.path.join(kf_dir, rec["image_path"]), args.root)
            writer.upsert_frame(rec)
            writer.enrich_frame(kfid, objects=objects.get(kfid), ocr_text=ocr.get(kfid),
                                caption=recap.get(kfid), ocr_api=ocr_api.get(kfid))
        writer.upsert_video({"_id": video_id, "fps": shots[video_id][0]["fps"],
                             "num_shots": len(shots[video_id]), "num_keyframes": len(keyframes),
                             "batch": "batch2"})
        writer.mark_step(video_id, "mongo", "done", len(keyframes))

        n_vec = mv_idx.index_video(video_id, "beit3", kf_dir)
        if os.path.exists(os.path.join(siglip_dir, f"{video_id}.npy")):
            n_vec += mv_idx.index_video(video_id, "siglip", siglip_dir)
        writer.mark_step(video_id, "milvus", "done", n_vec)
        writer.mark_step(video_id, "elastic", "done", es_idx.index_video(video_id))
        print(f"{video_id}: {len(keyframes)} keyframe, {n_vec} vector")


if __name__ == "__main__":
    main()
