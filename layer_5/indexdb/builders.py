def build_shot_doc(rec: dict, transcript: str = "") -> dict:
    return {
        "_id": rec["shot_id"],
        "video_id": rec["video_id"],
        "shot_id": rec["shot_id"],
        "start_frame": rec["start_frame"],
        "end_frame": rec["end_frame"],
        "fps": rec["fps"],
        "start_ms": round(rec["start_frame"] / rec["fps"] * 1000),
        "end_ms": round((rec["end_frame"] + 1) / rec["fps"] * 1000),
        "transcript": transcript,
    }

def build_frame_doc(rec: dict) -> dict:
    return {
        "_id": rec["keyframe_id"],
        "video_id": rec["video_id"],
        "shot_id": rec["shot_id"],
        "frame_idx": rec["frame_idx"],
        "timestamp_ms": rec["timestamp_ms"],
        "image_path": rec["image_path"],
        "batch": rec.get("batch", ""),
        "objects": {},
        "ocr_text": "",
        "ocr_api": "",
        "caption": "",
        "embeddings": {"beit3": False, "siglip2": False},
        "synced": {"elastic": False, "milvus": False},
    }

def build_seg_doc(rec: dict) -> dict:
    return {
        "_id": rec["seg_id"],
        "video_id": rec["video_id"],
        "seg_id": rec["seg_id"],
        "start_ms": rec["start_ms"],
        "end_ms": rec["end_ms"],
        "text": rec["text"],
    }

def build_es_doc(frame: dict, transcript: str = "") -> dict:
    return {
        "frame_id": frame["_id"],
        "video_id": frame["video_id"],
        "shot_id": frame["shot_id"],
        "frame_idx": frame["frame_idx"],
        "timestamp_ms": frame["timestamp_ms"],
        "ocr_text": frame.get("ocr_text", ""),
        "ocr_api": frame.get("ocr_api", ""),
        "caption": frame.get("caption", ""),
        "transcript": transcript,
        "object_tags": list(frame.get("objects", {}).keys()),
    }

def build_milvus_entity(frame_id: str, vector: list[float], scalars: dict) -> dict:
    return {
        "frame_id": frame_id,
        "video_id": scalars["video_id"],
        "frame_idx": scalars["frame_idx"],
        "shot_id": scalars["shot_id"],
        "timestamp_ms": scalars["timestamp_ms"],
        "embedding": vector,
    }
