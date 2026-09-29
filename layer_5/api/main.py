import os
from typing import Literal
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from indexdb.read import Reader

VIDEO_IDS_DESC = "Lọc theo video_id. Rỗng hoặc không truyền (None) nghĩa là KHÔNG lọc, không phải \"không khớp gì\"."
FRAME_IDS_DESC = (
    "Giới hạn search trong tập keyframe_id này (rerank trên candidate có sẵn, ví dụ từ "
    "/search/vector). Khác `video_ids`: không truyền (None) nghĩa là KHÔNG lọc, nhưng "
    "list rỗng `[]` nghĩa là KHÔNG khớp gì và trả về rỗng. Tối đa 65536 id."
)
MIN_SCORE_DESC = (
    "Chỉ giữ hit có `score` >= giá trị này. Lọc trước khi cắt `top_k`, nên top_k vẫn là "
    "trần an toàn (số hit tối đa trả về), không phải cỡ mẫu để lọc min_score từ đó. "
    "Không truyền (None) nghĩa là không lọc.\n\n"
    "Không có top_k=None (\"không giới hạn\") vì cả Milvus và Elasticsearch đều bắt buộc "
    "limit — muốn lấy hết hit đạt min_score, tự đặt top_k lớn (tối đa 16384 với "
    "/search/vector, 10000 với các endpoint text); vượt mức này server trả lỗi."
)

app = FastAPI(
    title="AIC2026 Read API",
    description=(
        "HTTP wrapper quanh Mongo/Milvus/Elasticsearch của layer_5 để search và đọc "
        "dữ liệu đã ingest (không cần cài pymongo/pymilvus/elasticsearch hay join "
        "Docker network). Mọi endpoint trừ /health cần header `X-API-Key`.\n\n"
        "Mọi kết quả /search/* trả cả `keyframe_id` (khoá nội bộ, chỉ dùng để gọi lại "
        "/frames) và `frame_idx` (số frame gốc trong video — **đây là giá trị phải "
        "nộp bài**, không phải `keyframe_id`)."
    ),
)

_reader: Reader | None = None


def get_reader() -> Reader:
    global _reader
    if _reader is None:
        _reader = Reader()
    return _reader


def require_api_key(x_api_key: str = Header(default="")) -> None:
    expected = os.getenv("API_KEY", "")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class SearchVectorRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {"model": "beit3", "vector": [0.01, -0.02, "... (1024 chiều)"], "top_k": 20},
            {"model": "siglip2", "vector": [0.01, -0.02, "... (1152 chiều)"], "top_k": 20},
        ]
    })
    model: Literal["beit3", "siglip2"]
    vector: list[float]
    top_k: int = 100
    video_ids: list[str] | None = Field(default=None, description=VIDEO_IDS_DESC)
    min_score: float | None = Field(default=None, description=MIN_SCORE_DESC)


class SearchQueryRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {"query": "một khu chợ hoa", "top_k": 20},
            {"query": "một khu chợ hoa", "top_k": 20, "frame_ids": ["v001_f0001", "v001_f0002"]},
        ]
    })
    query: str
    top_k: int = 100
    video_ids: list[str] | None = Field(default=None, description=VIDEO_IDS_DESC)
    frame_ids: list[str] | None = Field(default=None, max_length=65536, description=FRAME_IDS_DESC)


class SearchAsrRequest(SearchQueryRequest):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"query": "một khu chợ hoa", "top_k": 20, "min_score": 5.0}]
    })
    min_score: float | None = Field(default=None, description=MIN_SCORE_DESC)


class SearchObjectRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"labels": ["person", "car"], "top_k": 20, "min_score": 0.8}]
    })
    labels: list[str]
    top_k: int = 100
    video_ids: list[str] | None = Field(default=None, description=VIDEO_IDS_DESC)
    frame_ids: list[str] | None = Field(default=None, max_length=65536, description=FRAME_IDS_DESC)
    min_score: float | None = Field(default=None, description=MIN_SCORE_DESC)


class FramesRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"ids": ["v001_f0001", "v001_f0002"]}]
    })
    ids: list[str] = Field(description="keyframe_id lấy từ kết quả /search/* — dùng nội bộ, không phải frame_id để nộp bài.")


@app.get("/health", summary="Kiểm tra service còn sống")
def health():
    return {"status": "ok"}


@app.post(
    "/search/vector", dependencies=[Depends(require_api_key)],
    summary="Tìm bằng vector embedding (candidate generation)",
    description=(
        "Bạn tự encode text/ảnh thành vector, Reader không encode hộ. `model` là "
        "`beit3` (1024D) hoặc `siglip2` (1152D). Score là Milvus COSINE similarity "
        "(∈ [-1, 1]), `min_score` lọc trực tiếp trên thang này qua Milvus range search.\n\n"
        "Không hỗ trợ `frame_ids` (chỉ các endpoint text mới lọc được theo frame) — "
        "pipeline rerank là: lấy candidate rộng ở đây rồi truyền `frame_ids` vào `/search/*` text."
    ),
)
def search_vector(req: SearchVectorRequest, reader: Reader = Depends(get_reader)):
    return reader.search_vector(req.model, req.vector, top_k=req.top_k, video_ids=req.video_ids,
                                 min_score=req.min_score)


@app.post(
    "/search/ocr", dependencies=[Depends(require_api_key)],
    summary="Tìm theo chữ trong khung hình (OCR)",
    description="Match trên `ocr_text` (PaddleOCR gốc) + `ocr_api` (đã hiệu đính). Score là BM25.",
)
def search_ocr(req: SearchQueryRequest, reader: Reader = Depends(get_reader)):
    return reader.search_ocr(req.query, top_k=req.top_k, video_ids=req.video_ids, frame_ids=req.frame_ids)


@app.post(
    "/search/asr", dependencies=[Depends(require_api_key)],
    summary="Tìm theo lời thoại (ASR/transcript)",
    description=(
        "Match trên `transcript` của shot. Transcript được copy vào mọi keyframe thuộc shot "
        "đó, nên nhiều keyframe cùng shot có thể trả về cùng score. `min_score` cắt theo "
        "BM25 score — hữu ích để cắt gọn theo ranh giới shot thay vì cắt giữa cụm score bằng nhau, "
        "nhưng lưu ý BM25 không chuẩn hoá: ngưỡng hợp lý cho query này chưa chắc hợp lý cho query khác."
    ),
)
def search_asr(req: SearchAsrRequest, reader: Reader = Depends(get_reader)):
    return reader.search_asr(req.query, top_k=req.top_k, video_ids=req.video_ids, frame_ids=req.frame_ids,
                              min_score=req.min_score)


@app.post(
    "/search/all", dependencies=[Depends(require_api_key)],
    summary="Tìm gộp trên mọi nguồn text (OCR + ASR + caption)",
    description="Match trên `content_all`. Dùng khi không cần biết khớp từ nguồn nào, chỉ cần tìm nhanh.",
)
def search_all(req: SearchQueryRequest, reader: Reader = Depends(get_reader)):
    return reader.search_all(req.query, top_k=req.top_k, video_ids=req.video_ids, frame_ids=req.frame_ids)


@app.post(
    "/search/object", dependencies=[Depends(require_api_key)],
    summary="Tìm theo nhãn object detection",
    description=(
        "Match chính xác `labels` trong `object_tags` (không fuzzy). `score` trong kết quả "
        "là confidence thật của detector (lớn nhất giữa các object cùng nhãn trong 1 khung hình), "
        "không phải điểm term-match. `min_score` lọc trên confidence này — nhưng vì confidence chỉ "
        "biết được SAU khi lấy `top_k` hit từ Elasticsearch (ES không có confidence), hit confidence "
        "cao có thể bị bỏ sót nếu nằm ngoài top_k ban đầu. Đặt top_k dư ra khi dùng kèm min_score."
    ),
)
def search_object(req: SearchObjectRequest, reader: Reader = Depends(get_reader)):
    return reader.search_object(req.labels, top_k=req.top_k, video_ids=req.video_ids, frame_ids=req.frame_ids,
                                 min_score=req.min_score)


@app.post(
    "/frames", dependencies=[Depends(require_api_key)],
    summary="Hydrate metadata đầy đủ cho danh sách keyframe_id",
    description=(
        "Giữ nguyên thứ tự `ids` truyền vào, bỏ qua id không tồn tại. `image_path` trong "
        "response là đường dẫn tuyệt đối theo `DATA_ROOT` của server, chỉ dùng được trên "
        "máy đã mount cùng dataset. Response cũng có `frame_idx` — dùng giá trị này khi nộp bài."
    ),
)
def get_frames(req: FramesRequest, reader: Reader = Depends(get_reader)):
    return reader.get_frames(req.ids)


@app.get(
    "/shots/{video_id}", dependencies=[Depends(require_api_key)],
    summary="Danh sách shot của 1 video",
    description="Sắp theo `start_ms` tăng dần. Ví dụ: `/shots/K01_V001`.",
)
def get_shots(video_id: str, reader: Reader = Depends(get_reader)):
    return reader.get_shots(video_id)


@app.get(
    "/transcript/{video_id}", dependencies=[Depends(require_api_key)],
    summary="Đoạn transcript trong khoảng thời gian",
    description=(
        "Trả các đoạn transcript overlap `[start_ms, end_ms)`, dùng cho ngữ cảnh quanh 1 mốc "
        "(VQA kiểu \"trước/sau khi rời cửa hàng\"). Ví dụ: "
        "`/transcript/K01_V001?start_ms=118000&end_ms=130000`."
    ),
)
def get_transcript(video_id: str, start_ms: int, end_ms: int, reader: Reader = Depends(get_reader)):
    return reader.get_transcript(video_id, start_ms, end_ms)
