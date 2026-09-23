import numpy as np
import gpu_shot_and_asr as m


class FakeModel:
    def __init__(self):
        self.calls = []

    def endless_decode(self, audio_path, **kwargs):
        self.calls.append(audio_path)
        return "xin chào"


def test_one_call_per_segment(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(m, "_load_chunkformer_model", lambda name: fake)
    recognize = m._build_chunkformer("fake-model", None)
    chunks = [
        {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
        {"array": np.zeros(8000, dtype=np.float32), "sampling_rate": 16000},
    ]
    results = recognize(chunks, batch_size=1)
    assert len(results) == 2
    assert len(fake.calls) == 2
    assert results[0]["text"] == "xin chào"


def test_temp_wav_is_removed(monkeypatch, tmp_path):
    fake = FakeModel()
    monkeypatch.setattr(m, "_load_chunkformer_model", lambda name: fake)
    recognize = m._build_chunkformer("fake-model", None)
    recognize([{"array": np.zeros(16000, dtype=np.float32),
                "sampling_rate": 16000}], batch_size=1)
    import os
    assert not os.path.exists(fake.calls[0])
