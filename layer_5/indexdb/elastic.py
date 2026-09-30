from elasticsearch import Elasticsearch
from indexdb.config import Config

MAPPING = {
    "settings": {
        "analysis": {
            "analyzer": {
                "vi_analyzer": {
                    "type": "custom",
                    "tokenizer": "vi_tokenizer",
                    # asciifolding: OCR residual errors are almost always a
                    # missing/wrong diacritic ("giay" for "giây"), never a
                    # wrong base letter. Folding both index and query side
                    # (same analyzer is used for search by default) makes
                    # those equivalent instead of needing perfect OCR text.
                    "filter": ["lowercase", "asciifolding"],
                },
                "vi_analyzer_strict": {
                    "type": "custom",
                    "tokenizer": "vi_tokenizer",
                    # KHÔNG asciifolding: khác OCR, ASR (Whisper) không có vấn đề
                    # mất dấu ở nguồn, nên giữ dấu để so khớp chính xác thay vì
                    # fold cả những câu gõ đúng chính tả về chung một dạng.
                    "filter": ["lowercase"],
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "frame_id":     {"type": "keyword"},
            "video_id":     {"type": "keyword"},
            "shot_id":      {"type": "keyword"},
            "frame_idx":    {"type": "long"},
            "timestamp_ms": {"type": "long"},
            # ocr_text.exact giữ dấu: OCR Gemma đọc đúng dấu, search_ocr cộng điểm frame khớp đúng dấu.
            "ocr_text":    {"type": "text", "analyzer": "vi_analyzer", "copy_to": "content_all",
                            "fields": {"exact": {"type": "text", "analyzer": "vi_analyzer_strict"}}},
            # OCR có 2 nguồn: PaddleOCR gốc (ocr_text) và bản đã hiệu đính
            # (ocr_api) — giữ riêng để so sánh chất lượng.
            "ocr_api":     {"type": "text", "analyzer": "vi_analyzer", "copy_to": "content_all"},
            "caption":     {"type": "text", "analyzer": "vi_analyzer", "copy_to": "content_all"},
            "transcript":  {"type": "text", "analyzer": "vi_analyzer_strict", "copy_to": "content_all"},
            "content_all": {"type": "text", "analyzer": "vi_analyzer"},
            "object_tags": {"type": "keyword"},
        }
    },
}

class ElasticStore:
    def __init__(self, cfg: Config, index_name: str = "frame_text"):
        self.client = Elasticsearch(cfg.elastic_uri, request_timeout=60,
                                    retry_on_timeout=True, max_retries=5)
        self.index_name = index_name

    def create_index(self):
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=MAPPING)

    def drop_index(self):
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
