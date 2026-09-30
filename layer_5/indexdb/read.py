import os
import unicodedata

from indexdb.config import Config
from indexdb.elastic import ElasticStore
from indexdb.milvus import MilvusStore
from indexdb.mongo import MongoStore


class Reader:
    def __init__(self, cfg: Config | None = None):
        cfg = cfg or Config.from_env()
        self.mongo = MongoStore(cfg)
        self.es = ElasticStore(cfg)
        self.mv = MilvusStore(cfg)
        self.data_root = cfg.data_root
        self._check_ready(self.es, self.mv)

    @staticmethod
    def _check_ready(es: ElasticStore, mv: MilvusStore):
        assert es.client.indices.exists(index=es.index_name), (
            f"thiếu index '{es.index_name}' — kiểm tra ELASTICSEARCH_URI, "
            f"hoặc chạy indexdb.init_stores nếu chưa init"
        )
        for name in mv.collection_names().values():
            assert mv.client.has_collection(name), (
                f"thiếu collection '{name}' — kiểm tra MILVUS_URI, "
                f"hoặc chạy indexdb.init_stores nếu chưa init"
            )

    def search_vector(self, model: str, vector: list[float], top_k: int = 100,
                       video_ids: list[str] | None = None, min_score: float | None = None) -> list[dict]:
        name = self.mv.collection_names()[model]
        flt = f'video_id in {video_ids}' if video_ids else None
        # range search: COSINE similarity nằm trong [radius, range_filter). range_filter=2.0
        # (max lý thuyết của COSINE là 1.0) để không phải lo min_score gần biên top do lỗi
        # làm tròn dấu phẩy động, và vẫn hợp lệ (radius < range_filter) với mọi min_score <= 1.
        search_params = {"params": {"radius": min_score, "range_filter": 2.0}} if min_score is not None else None
        results = self.mv.client.search(
            collection_name=name, data=[vector], limit=top_k, filter=flt, search_params=search_params,
            output_fields=["video_id", "shot_id", "frame_idx", "timestamp_ms"],
        )
        return [
            {
                "keyframe_id": hit["frame_id"],
                "video_id": hit["entity"]["video_id"],
                "shot_id": hit["entity"]["shot_id"],
                "frame_idx": hit["entity"]["frame_idx"],
                "timestamp_ms": hit["entity"]["timestamp_ms"],
                "score": hit["distance"],
            }
            for hit in results[0]
        ]

    def search_ocr(self, query: str, top_k: int = 100, video_ids: list[str] | None = None,
                   frame_ids: list[str] | None = None) -> list[dict]:
        # multi_match cả ocr_text (OCR Gemma gốc) và ocr_api (đã hiệu đính) vì
        # không biết trước nguồn nào khớp; fuzziness="AUTO" chịu lỗi chính tả OCR.
        # must=OR giữ nguyên recall (1 từ khớp là đủ vào candidate); should lặp lại
        # chính điều kiện đó với minimum_should_match="2<75%" + boost=5 làm điểm CỘNG
        # THÊM, không lọc bớt gì -- kéo candidate khớp PHẦN LỚN từ lên gần đầu, đỡ bị
        # BM25 tự chuẩn hoá độ dài chôn vùi dưới hàng trăm candidate ngắn chỉ khớp 1 từ
        # (đo thật, query 3 từ: không should thì 11/20 candidate khớp đủ từ rớt khỏi
        # top 500; should+boost=5 kéo 19/20 lên top 20).
        # ⚠️ Từng dùng operator="and" (đòi ĐỦ 100% từ) -- đổi vì gõ dư 1 từ không có
        # trong OCR của đúng frame (vd tìm "chùa một cột" gõ thêm "hà nội" mà OCR frame
        # chỉ có "chùa một cột") làm should KHÔNG kích hoạt cho frame đúng dù nó khớp
        # 3/4 từ, trong khi frame SAI khác chỉ cần tình cờ đủ 4/4 từ vẫn được thưởng
        # trọn boost -- đo được: frame đúng rớt xuống hạng 148-149/200.
        # ⚠️ Đổi tiếp sang "75%" trần rồi phát hiện SAI cho query NGẮN: query 3 từ,
        # ES làm tròn % kiểu không như ceil() tưởng -- mọi mức dưới 100% (34/50/67/75%)
        # đều rơi về "chỉ cần 1/3 từ" (test trực tiếp: 2/20 lọt top 20, gần như không
        # còn tác dụng gì so với operator="and" cũ). "75%" chỉ đúng ý cho query DÀI
        # (4 từ trở lên) chứ không phải mọi độ dài. Cú pháp combo "2<75%" (dưới 2 từ
        # optional thì bắt buộc đủ 2, từ 2 trở lên thì dùng 75%) giải quyết được cả
        # hai đầu cùng lúc -- đo lại: query 3 từ 19/20 lọt top 20 (khớp lại kỳ vọng),
        # query 4 từ gõ dư vẫn giữ hạng 51-84 (không hồi quy so với "75%" trần).
        # should thứ 2 (KHÔNG fuzziness, boost=3) tách doc khớp CHÍNH XÁC khỏi doc chỉ
        # khớp nhờ fuzzy sửa chính tả: should fuzzy đếm từ khớp mà không phân biệt hai
        # loại này nên cả hai ăn trọn boost=5 như nhau; thêm clause exact thì doc khớp
        # chính xác ăn 5+3, doc nhờ fuzzy vẫn ăn 5 (không bị loại -- đó là lý do có fuzzy).
        # boost=3 CHƯA ĐO, chỉnh theo bộ đo top-20 nếu query gõ sai chính tả bị chôn.
        # ⚠️ fuzziness AUTO -> 1 (2026-08-21, đồng bộ với backend/core/stores/elastic.py
        # cùng ngày): vi_tokenizer gộp 2 âm tiết thành MỘT token trước khi so khớp
        # ("châu đốc" -> token "chau doc", 8 ký tự); AUTO cho token >5 ký tự cho phép
        # sai lệch 2 ký tự, đủ để "chau doc" fuzzy-khớp "cham soc" ("chăm sóc" -- nghĩa
        # khác hẳn, không phải lỗi OCR). must=OR trần ở đây MẤT PHÒNG THỦ càng nặng hơn
        # elastic.py (chỉ cần 1 token khớp fuzzy nhầm là đã vào candidate). Xem docstring
        # search_ocr ở elastic.py cho bằng chứng đo trên ES sống.
        # ocr_text.exact giữ dấu (OCR Gemma đọc đúng dấu): câu gõ có dấu thì frame khớp
        # đúng dấu được cộng thêm, câu không dấu thì không thêm (khỏi thưởng nhầm frame
        # OCR thiếu dấu). Cùng ý với backend/core/stores/elastic.py::search_ocr.
        has_marks = any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", query)) \
            or "đ" in query.lower()
        es_query = {"bool": {
            "must": [{"multi_match": {"query": query, "fields": ["ocr_text", "ocr_api"], "fuzziness": 1, "max_expansions": 10}}],
            "should": [{"multi_match": {"query": query, "fields": ["ocr_text", "ocr_api"],
                                         "minimum_should_match": "2<75%", "fuzziness": 1, "max_expansions": 10, "boost": 5}},
                       {"multi_match": {"query": query, "fields": ["ocr_text", "ocr_api"],
                                         "minimum_should_match": "2<75%", "boost": 3}}]
                      + ([{"match": {"ocr_text.exact": {"query": query, "operator": "and", "boost": 3}}}]
                         if has_marks else []),
        }}
        return self._search(es_query, top_k, video_ids, frame_ids)

    def search_asr(self, query: str, top_k: int = 100, video_ids: list[str] | None = None,
                   frame_ids: list[str] | None = None, min_score: float | None = None) -> list[dict]:
        # fuzziness AUTO -> 1 (2026-08-21) -- cùng bug với search_ocr ở trên, transcript
        # dùng chung vi_tokenizer nên cùng bị gộp token và cùng fuzzy-khớp nhầm từ khác.
        es_query = {"bool": {
            "must": [{"match": {"transcript": {"query": query, "fuzziness": 1}}}],
            "should": [{"match": {"transcript": {"query": query, "minimum_should_match": "2<75%",
                                                  "fuzziness": 1, "boost": 5}}},
                       {"match": {"transcript": {"query": query, "minimum_should_match": "2<75%", "boost": 3}}}],
        }}
        return self._search(es_query, top_k, video_ids, frame_ids, min_score)

    def search_all(self, query: str, top_k: int = 100, video_ids: list[str] | None = None,
                   frame_ids: list[str] | None = None) -> list[dict]:
        # content_all gộp ocr_text/ocr_api/caption/transcript qua copy_to — dùng khi
        # không cần biết khớp từ nguồn nào, chỉ cần tìm nhanh trên toàn bộ text.
        # fuzziness AUTO -> 1 (2026-08-21), cùng lý do search_ocr/search_asr ở trên.
        es_query = {"bool": {
            "must": [{"match": {"content_all": {"query": query, "fuzziness": 1}}}],
            "should": [{"match": {"content_all": {"query": query, "minimum_should_match": "2<75%",
                                                   "fuzziness": 1, "boost": 5}}},
                       {"match": {"content_all": {"query": query, "minimum_should_match": "2<75%", "boost": 3}}}],
        }}
        return self._search(es_query, top_k, video_ids, frame_ids)

    def search_object(self, labels: list[str], top_k: int = 100, video_ids: list[str] | None = None,
                      frame_ids: list[str] | None = None, min_score: float | None = None) -> list[dict]:
        # object_tags là keyword (nhãn rời rạc từ detector) nên match chính xác qua terms, không fuzzy.
        # ES chỉ biết frame có/không có nhãn, không biết độ tự tin — lấy confidence
        # thật (max giữa các object cùng nhãn) từ Mongo rồi sort lại theo đó.
        # min_score lọc TRÊN confidence thật này, không phải _score của ES (không có ý nghĩa
        # với terms query) — nên lọc sau khi join Mongo, không truyền vào _search.
        hits = self._search({"terms": {"object_tags": labels}}, top_k, video_ids, frame_ids)
        objects_by_frame = {
            d["_id"]: d["objects"]
            for d in self.mongo.frames.find({"_id": {"$in": [h["keyframe_id"] for h in hits]}}, {"objects": 1})
        }
        for hit in hits:
            objects = objects_by_frame[hit["keyframe_id"]]
            hit["score"] = max(objects[label] for label in labels if label in objects)
        if min_score is not None:
            hits = [h for h in hits if h["score"] >= min_score]
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits

    def _search(self, es_query: dict, top_k: int, video_ids: list[str] | None,
                frame_ids: list[str] | None = None, min_score: float | None = None) -> list[dict]:
        # frame_ids=[] phải trả về rỗng (candidate list rỗng khi rerank), khác video_ids
        # nơi [] nghĩa là không lọc — nên check `is not None` thay vì truthiness.
        filters = ([{"terms": {"video_id": video_ids}}] if video_ids else []) \
                  + ([{"terms": {"frame_id": frame_ids}}] if frame_ids is not None else [])
        if filters:
            es_query = {"bool": {"must": es_query, "filter": filters}}
        kwargs = {"min_score": min_score} if min_score is not None else {}
        resp = self.es.client.search(index=self.es.index_name, query=es_query, size=top_k, **kwargs)
        return [
            {
                "keyframe_id": hit["_source"]["frame_id"],
                "video_id": hit["_source"]["video_id"],
                "shot_id": hit["_source"]["shot_id"],
                "frame_idx": hit["_source"]["frame_idx"],
                "timestamp_ms": hit["_source"]["timestamp_ms"],
                "score": hit["_score"],
            }
            for hit in resp["hits"]["hits"]
        ]

    def get_frames(self, ids: list[str]) -> list[dict]:
        docs = {d["_id"]: d for d in self.mongo.frames.find({"_id": {"$in": ids}})}
        out = []
        for i in ids:
            if i not in docs:
                continue
            doc = dict(docs[i])
            doc["keyframe_id"] = doc.pop("_id")
            doc["image_path"] = os.path.join(self.data_root, doc["image_path"])
            out.append(doc)
        return out

    def get_shots(self, video_id: str) -> list[dict]:
        return list(self.mongo.shots.find({"video_id": video_id}).sort("start_ms", 1))

    def get_transcript(self, video_id: str, start_ms: int, end_ms: int) -> list[dict]:
        query = {"video_id": video_id, "start_ms": {"$lt": end_ms}, "end_ms": {"$gt": start_ms}}
        return list(self.mongo.transcript_segments.find(query).sort("start_ms", 1))
