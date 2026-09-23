import numpy as np
import gpu_shot_and_asr as m


class FakeModel:
    def endless_decode(self, audio_path, **kwargs):
        return "greedy text"


def test_greedy_when_no_kenlm(monkeypatch, capsys):
    monkeypatch.setattr(m, "_load_chunkformer_model", lambda name: FakeModel())
    recognize = m._build_chunkformer("fake", None)
    out = recognize([{"array": np.zeros(16000, dtype=np.float32),
                      "sampling_rate": 16000}], batch_size=1)
    assert out[0]["text"] == "greedy text"
    assert "greedy" in capsys.readouterr().out


def test_kenlm_path_missing_fails_loudly(monkeypatch):
    monkeypatch.setattr(m, "_load_chunkformer_model", lambda name: FakeModel())
    try:
        m._build_chunkformer("fake", "/khong/ton/tai.arpa")
    except FileNotFoundError as exc:
        assert "/khong/ton/tai.arpa" in str(exc)
    else:
        raise AssertionError("phải báo lỗi khi file KenLM không tồn tại")
