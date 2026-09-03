from pymilvus import MilvusClient, DataType
from indexdb.config import Config

COLLECTIONS = {"beit3": 1024, "siglip2": 1152}

class MilvusStore:
    def __init__(self, cfg: Config, suffix: str = ""):
        self.client = MilvusClient(cfg.milvus_uri)
        self.suffix = suffix

    def collection_names(self) -> dict:
        return {model: f"{model}_frames{self.suffix}" for model in COLLECTIONS}

    def _schema(self, dim: int):
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("frame_id",     DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("video_id",     DataType.VARCHAR, max_length=32)
        schema.add_field("frame_idx",    DataType.INT64)
        schema.add_field("shot_id",      DataType.VARCHAR, max_length=32)
        schema.add_field("timestamp_ms", DataType.INT64)
        schema.add_field("embedding",    DataType.FLOAT_VECTOR, dim=dim)
        return schema

    def create_collections(self):
        for model, dim in COLLECTIONS.items():
            name = self.collection_names()[model]
            if self.client.has_collection(name):
                continue
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding", index_type="HNSW", metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            self.client.create_collection(
                collection_name=name, schema=self._schema(dim), index_params=index_params
            )
            self.client.load_collection(name)

    def drop_all(self):
        for name in self.collection_names().values():
            if self.client.has_collection(name):
                self.client.drop_collection(name)
