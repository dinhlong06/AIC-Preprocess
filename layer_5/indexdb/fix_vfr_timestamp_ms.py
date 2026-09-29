"""Sửa timestamp_ms sai (frame_idx/25) của video VFR N001-N100 trong Mongo+ES,
bằng pts_ms thật đã ghi vào keyframes.jsonl (layer_2/.../add_pts_ms.py).

BTC 2026-09-25: N001-N100 là VFR, frame_idx/fps không đúng thời gian thật.
"""
import glob
import json

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pymongo import MongoClient, UpdateOne

import sys
sys.path.insert(0, "../../retrieval_system/backend")
from config import settings  # noqa: E402

KF_GLOB = "../../output_batch2/keyframes/pipeline_h/N*/keyframes.jsonl"


def load_corrections():
    out = {}
    for path in glob.glob(KF_GLOB):
        for line in open(path):
            row = json.loads(line)
            if "pts_ms" in row:
                out[row["keyframe_id"]] = row["pts_ms"]
    return out


def main():
    corrections = load_corrections()
    print(f"{len(corrections)} keyframe cần sửa")

    mongo = MongoClient(settings.MONGO_URI)[settings.MONGO_DB]
    mongo_ops = [
        UpdateOne({"_id": kf_id}, {"$set": {"timestamp_ms": ms}})
        for kf_id, ms in corrections.items()
    ]
    res = mongo.frames.bulk_write(mongo_ops, ordered=False)
    print(f"Mongo: {res.modified_count} doc sửa")

    es = Elasticsearch(settings.ELASTIC_URI)
    actions = (
        {"_op_type": "update", "_index": settings.ELASTIC_INDEX, "_id": kf_id,
         "doc": {"timestamp_ms": ms}}
        for kf_id, ms in corrections.items()
    )
    ok, errors = bulk(es, actions, stats_only=False, raise_on_error=False)
    print(f"ES: {ok} doc sửa, {len(errors)} lỗi")
    if errors:
        print(errors[:5])


if __name__ == "__main__":
    main()
