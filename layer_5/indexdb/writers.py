from datetime import datetime, timezone
from indexdb.mongo import MongoStore
from indexdb.builders import build_shot_doc, build_frame_doc, build_seg_doc

def _now():
    return datetime.now(timezone.utc).isoformat()

class MongoWriter:
    def __init__(self, store: MongoStore):
        self.store = store

    def upsert_video(self, doc: dict):
        self.store.videos.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def upsert_shot(self, rec: dict, transcript=""):
        doc = build_shot_doc(rec, transcript)
        self.store.shots.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def upsert_seg(self, rec: dict):
        doc = build_seg_doc(rec)
        self.store.transcript_segments.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    def upsert_frame(self, rec: dict) -> str:
        # chỉ set field gốc nếu chưa tồn tại; giữ nguyên enrichment đã có
        doc = build_frame_doc(rec)
        self.store.frames.update_one(
            {"_id": doc["_id"]},
            {"$setOnInsert": doc},
            upsert=True,
        )
        return doc["_id"]

    def enrich_frame(self, frame_id: str, objects=None, ocr_text=None, caption=None,
                     ocr_api=None, shot_id=None):
        update = {}
        if objects is not None:  update["objects"] = objects
        if ocr_text is not None: update["ocr_text"] = ocr_text
        if ocr_api is not None:  update["ocr_api"] = ocr_api
        if caption is not None:  update["caption"] = caption
        if shot_id is not None:  update["shot_id"] = shot_id
        if update:
            self.store.frames.update_one({"_id": frame_id}, {"$set": update})

    def mark_step(self, video_id: str, step: str, status: str, count: int | None = None):
        entry = {"status": status, "updated_at": _now()}
        if count is not None:
            entry["count"] = count
        self.store.ingest_status.update_one(
            {"_id": video_id},
            {"$set": {f"steps.{step}": entry}},
            upsert=True,
        )

