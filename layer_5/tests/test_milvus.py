import pytest
from indexdb.milvus import MilvusStore, COLLECTIONS

@pytest.fixture
def mv(cfg):
    store = MilvusStore(cfg, suffix="_test")
    store.drop_all()
    yield store
    store.drop_all()

@pytest.mark.integration
def test_create_collections(mv):
    mv.create_collections()
    for name in mv.collection_names().values():
        assert mv.client.has_collection(name)

@pytest.mark.integration
def test_collection_dims():
    assert COLLECTIONS["beit3"] == 1024
    assert COLLECTIONS["siglip2"] == 1152
