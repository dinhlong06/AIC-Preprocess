"""Nạp dataset_batch1 (BTC AIC2026) vào Mongo + Milvus + Elastic.

    python -m indexdb.ingest_batch1 --root /path/dataset_batch1 \\
        --keyframes-dir /path/pipeline_g \\
        [--ocr layer_3/OCR_gemma/gemma_ocr_batch1.jsonl] \\
        [--captions 'layer_3/OCR_gemma/*_caption_batch1.jsonl'] \\
        [--ocr-api layer_3/OCR/output_batch1_v2/output_vlm_corrected.json] \\
        --siglip2-dir recap_siglip/artifacts/siglip_batch1 \\
        [--videos L21_V001 ...]

Nguồn metadata: keyframes.jsonl từ pipeline_g (keyframe extraction output).
Nguồn embedding: SigLIP2 SO400M (1152d) từ recap_siglip/artifacts/siglip_batch1/.
frame_id = keyframe_id = SigLIP2 embedding ID (L21_V001_000000_kf0001).
"""
import argparse
import glob
import bisect
import json
import os
import unicodedata

import numpy as np

from indexdb.builders import build_milvus_entity
from indexdb.config import Config
from indexdb.elastic import ElasticStore
from indexdb.elastic_indexer import ElasticIndexer
from indexdb.milvus import MilvusStore
from indexdb.mongo import MongoStore
from indexdb.writers import MongoWriter

MIN_CONF = 0.5

_DATA_ROOT = os.getenv("DATA_ROOT", "/data")
DEFAULT_SHOTS_PATH = os.path.join(_DATA_ROOT, "layer_1", "batch1", "shots.jsonl")
DEFAULT_TRANSCRIPTS_PATH = os.path.join(
    _DATA_ROOT, "layer_2", "shot_transcript", "shot_transcripts_batch1.jsonl")


def _read_keyframes_jsonl(video_dir):
    path = os.path.join(video_dir, "keyframes.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _group_by_video(path):
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out.setdefault(rec["video_id"], []).append(rec)
    return out


def _derive_fps(keyframes, default=30.0):
    """keyframes.jsonl không ghi fps — suy ngược từ frame_idx/timestamp_ms
    của keyframe đầu tiên có cả hai giá trị dương."""
    for kf in keyframes:
        if kf["frame_idx"] > 0 and kf["timestamp_ms"] > 0:
            return kf["frame_idx"] / (kf["timestamp_ms"] / 1000)
    return default


def _build_shot_lookup(shots_by_video):
    lookup = {}
    for video_id, shots in shots_by_video.items():
        ordered = sorted(shots, key=lambda s: s["start_frame"])
        lookup[video_id] = ([s["start_frame"] for s in ordered], ordered)
    return lookup


def _find_shot_id(lookup, video_id, frame_idx):
    starts, ordered = lookup.get(video_id, ([], []))
    i = bisect.bisect_right(starts, frame_idx) - 1
    if i < 0:
        return ""
    shot = ordered[i]
    return shot["shot_id"] if frame_idx <= shot["end_frame"] else ""


def _load_objects(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    objects_by_frame = {}
    for r in rows:
        labels = {}
        for o in r["objects"]:
            if o["confidence"] >= MIN_CONF:
                labels[o["label"]] = max(labels.get(o["label"], 0.0), o["confidence"])
        if labels:
            objects_by_frame[r["frame_id"]] = labels
    return objects_by_frame


def _load_ocr(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["frame_id"]: " ".join(
                    "".join(c for c in unicodedata.normalize(
                        "NFD", t["text"].replace("Đ", "D").replace("đ", "d"))
                            if not unicodedata.combining(c))
                    for t in r["texts"] if t["confidence"] >= MIN_CONF)
                for r in json.load(f)}


def _load_gemma(path):
    # Giữ nguyên dấu (khác _load_ocr): Gemma đọc đúng dấu, và vi_analyzer đã asciifolding
    # ở cả index lẫn query nên tìm không dấu vẫn khớp.
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["frame_id"]: r["text"] for r in map(json.loads, f) if r["text"]}


def _purge(video_id, store, es):
    store.frames.delete_many({"video_id": video_id})
    store.shots.delete_many({"video_id": video_id})
    store.ingest_status.delete_one({"_id": video_id})
    es.client.delete_by_query(index=es.index_name, refresh=True,
                              body={"query": {"term": {"video_id": video_id}}})


def _index_vectors(store: MongoStore, mv: MilvusStore, video_id: str, model: str,
                   vecs: np.ndarray, frame_ids: list[str]) -> int:
    assert vecs.shape[0] == len(frame_ids), f"{video_id}: số vector {model} không khớp số frame"
    scalars = {d["_id"]: d for d in store.frames.find({"_id": {"$in": frame_ids}})}
    entities = [build_milvus_entity(fid, vecs[i].tolist(), scalars[fid]) for i, fid in enumerate(frame_ids)]
    name = mv.collection_names()[model]
    mv.client.delete(name, filter=f'video_id == "{video_id}"')
    mv.client.insert(collection_name=name, data=entities)
    store.frames.update_many({"_id": {"$in": frame_ids}},
                             {"$set": {"synced.milvus": True, f"embeddings.{model}": True}})
    return len(entities)


def main():
    ap = argparse.ArgumentParser(prog="indexdb.ingest_batch1")
    ap.add_argument("--root", required=True, help="thư mục dataset_batch1")
    ap.add_argument("--keyframes-dir", required=True,
                    help="thư mục pipeline_g chứa {video_id}/keyframes.jsonl")
    ap.add_argument("--ocr", nargs="*", default=[],
                    help="OCR Gemma, jsonl (nhận glob) của layer_3/OCR_gemma/gemma_ocr.py --task ocr")
    ap.add_argument("--captions", nargs="*", default=[],
                    help="caption, jsonl của layer_3/OCR_gemma/gemma_ocr.py --task caption (UIT + AI Studio chia nhau)")
    ap.add_argument("--ocr-api", help="đường dẫn output đã hiệu đính (vd output_vlm_corrected.json)")
    ap.add_argument("--objects", help="đường dẫn detections.json")
    ap.add_argument("--siglip2-dir", required=True,
                    help="thư mục SigLIP2 batch1 ({video_id}.npy + {video_id}_ids.json, 1152 chiều)")
    ap.add_argument("--shots", default=DEFAULT_SHOTS_PATH,
                    help="shots.jsonl batch1")
    ap.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS_PATH,
                    help="shot_transcripts_batch1.jsonl")
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--batch", default="batch1")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="bỏ qua video đã nạp xong")
    mode.add_argument("--purge", action="store_true",
                      help="xoá frame/shot cũ của video trong Mongo/Elastic trước khi nạp lại "
                           "(khi layer_2 chạy lại ra bộ keyframe khác, vd batch1_v2)")
    args = ap.parse_args()

    video_ids = args.videos or sorted(
        d for d in os.listdir(args.keyframes_dir)
        if os.path.isdir(os.path.join(args.keyframes_dir, d)) and not d.startswith("_")
    )
    # Giữ cả text rỗng: Gemma thấy frame không có chữ thì phải ghi đè OCR Paddle rác còn sót.
    # "_" -> cách: overlay camera N095 ghi "NGUYEN_VAN_CU_NGUYEN_TRAI", tokenizer giữ nguyên thành 1 từ.
    ocr = {r["frame_id"]: r["text"].replace("_", " ") for pat in args.ocr for path in sorted(glob.glob(pat))
           for r in map(json.loads, open(path, encoding="utf-8"))}
    captions = {k: v for pat in args.captions for path in sorted(glob.glob(pat)) for k, v in _load_gemma(path).items()}
    ocr_api = _load_ocr(args.ocr_api)
    objects_by_frame = _load_objects(args.objects)
    shots_by_video = _group_by_video(args.shots)
    shot_lookup = _build_shot_lookup(shots_by_video)
    transcripts = {r["shot_id"]: r["text"] for r in
                   (json.loads(l) for l in open(args.transcripts, encoding="utf-8"))} \
        if os.path.exists(args.transcripts) else {}

    cfg = Config.from_env()
    store = MongoStore(cfg)
    writer = MongoWriter(store)
    es_idx = ElasticIndexer(store, ElasticStore(cfg))
    mv = MilvusStore(cfg)

    for video_id in video_ids:
        if args.resume and store.ingest_status.find_one({"_id": video_id, "steps.elastic.status": "done"}):
            print(f"{video_id}: bỏ qua (đã xong)")
            continue
        if args.purge:
            _purge(video_id, store, es_idx.es)

        kf_dir = os.path.join(args.keyframes_dir, video_id)
        keyframes = _read_keyframes_jsonl(kf_dir)
        if not keyframes:
            print(f"{video_id}: bỏ qua (không có keyframes.jsonl)")
            continue

        frame_ids = []
        for kf in keyframes:
            keyframe_id = kf["keyframe_id"]
            frame_ids.append(keyframe_id)
            shot_id = _find_shot_id(shot_lookup, video_id, kf["frame_idx"])
            writer.upsert_frame({
                "keyframe_id": keyframe_id, "video_id": video_id, "shot_id": shot_id,
                "frame_idx": kf["frame_idx"], "timestamp_ms": kf["timestamp_ms"],
                # Tương đối với gốc repo (/data): backend thử gốc repo trước, nên batch2 (output_batch2/...)
                # tìm được ảnh mà không phải thêm gốc mới vào PIPELINE_G_ROOT.
                "image_path": os.path.relpath(os.path.join(kf_dir, kf["image_path"]), _DATA_ROOT),
                "batch": args.batch,
            })
            objects = objects_by_frame.get(keyframe_id)
            # Video N* (camera giao thông batch2) chỉ OCR frame đầu, lưu theo video_id: chép sang mọi keyframe.
            ocr_text = ocr.get(keyframe_id, ocr.get(video_id))
            ocr_api_text = ocr_api.get(keyframe_id)
            caption = captions.get(keyframe_id)
            if objects or ocr_text is not None or ocr_api_text or caption or shot_id:
                writer.enrich_frame(keyframe_id, objects=objects, ocr_text=ocr_text, caption=caption,
                                    ocr_api=ocr_api_text, shot_id=shot_id or None)

        for shot in shots_by_video.get(video_id, []):
            writer.upsert_shot(shot, transcript=transcripts.get(shot["shot_id"], ""))

        media_info_path = os.path.join(args.root, "media-info", f"{video_id}.json")
        fps = _derive_fps(keyframes)
        video_doc = {"_id": video_id, "fps": fps, "batch": args.batch}
        if os.path.exists(media_info_path):
            with open(media_info_path, encoding="utf-8") as f:
                video_doc["media_info"] = json.load(f)
        writer.upsert_video(video_doc)
        writer.mark_step(video_id, "mongo", "done", len(keyframes))

        n_vec = 0
        models = []
        npy_path = os.path.join(args.siglip2_dir, f"{video_id}.npy")
        ids_path = os.path.join(args.siglip2_dir, f"{video_id}_ids.json")
        if os.path.exists(npy_path) and os.path.exists(ids_path):
            with open(ids_path, encoding="utf-8") as f:
                siglip_ids = json.load(f)
            kf_id_set = set(frame_ids)
            valid = [(i, fid) for i, fid in enumerate(siglip_ids) if fid in kf_id_set]
            if valid:
                indices, valid_ids = zip(*valid)
                vecs = np.load(npy_path)[list(indices)]
                n_vec += _index_vectors(store, mv, video_id, "siglip2", vecs, list(valid_ids))
                models.append("siglip2")

        writer.mark_step(video_id, "milvus", "done", n_vec)

        n_es = es_idx.index_video(video_id)
        writer.mark_step(video_id, "elastic", "done", n_es)
        print(f"{video_id}: {len(keyframes)} frame, {n_vec} vector ({'+'.join(models)})")


if __name__ == "__main__":
    main()
