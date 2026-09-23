# Layer 1 ASR — ChunkFormer + KenLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thay PhoWhisper bằng ChunkFormer + KenLM làm đường ASR duy nhất của layer_1, giữ nguyên VAD/resume/atomic-write, và chỉ xóa đường cũ sau khi có số đo được review.

**Architecture:** Không dựng registry backend. Tách một hàm thuần `_build_entries()` khỏi `run_asr` để test được schema và timestamp mà không cần GPU, thêm hàm dựng recognizer ChunkFormer cạnh hàm dựng PhoWhisper hiện có, rồi xóa nhánh PhoWhisper ở task cuối cùng sau khi đối chiếu.

**Tech Stack:** Python 3, PyTorch 2.1.2+cu118, `chunkformer`, `pyctcdecode`, `kenlm`, `silero-vad`, pytest, Docker (base `tensorflow/tensorflow:2.13.0-gpu`).

**Spec:** `docs/superpowers/specs/2026-09-22-merge-v2-into-layers-design.md` (mục 4, 9, 11, 13)

## Global Constraints

- **`docs/` bị `.gitignore` chặn** (`.gitignore:82`). Mọi commit chạm file trong `docs/` phải dùng `git add -f`, nếu không `git add` sẽ im lặng không thêm gì.
- **Không đụng** trong `gpu_shot_and_asr.py`: `extract_audio_to_wav`, `_claim`, `_read_done_video_ids`, `_write_lines_atomic`, `_append_video`, `_drop_video_ids`, `run_shot_detection`. Đã chạy qua 605 video.
- **`start_ms`/`end_ms` phải luôn đến từ Silero VAD**, không bao giờ từ model. `layer_2/shot_transcript/cpu_map_transcript.py:47-50` join shot↔transcript bằng chồng lấn thời gian; timestamp lệch không lộ ở layer_1 mà hiện ra thành kết quả tìm kiếm sai.
- **Schema `whisper.jsonl`** sau plan này: `{video_id, seg_id, start_ms, end_ms, text, confidence}`. Thêm trường là additive an toàn — `cpu_map_transcript.py` chỉ đọc 4 trường đầu, `layer_5/indexdb/ingest.py:110` truyền nguyên bản ghi vào Mongo.
- **Máy GPU dùng chung** (8x RTX 2080 Ti, có co-tenant). Không chạy full dataset trong plan này. Mọi smoke test giới hạn 1 video.
- **Không cài package lên host.** Mọi dependency chỉ cài trong Docker image của layer_1.
- **`dataset/video` là symlink tới `/mlcv2025/Datasets/HCMAI25/batch2/video` và hiện RỖNG.** Video thật nằm ở `dataset_batch1/videos/video/` (873 file), là thứ `run_layer1_batch1.sh` dùng. Mọi smoke test phải dùng đường đó.
- **Không bao giờ ghi vào `layer_1/batch1/`.** Đó là output production (57.846 segment). Smoke test luôn override `OUTPUT_DIR` sang `/tmp/...`.
- **Giữ nguyên** `seg_id` dạng `f"{video_id}_{seg_idx:06d}"` với `seg_idx` đếm trên **toàn bộ** segment VAD (segment rỗng vẫn tiêu thụ một chỉ số, tạo lỗ hổng trong dãy seg_id). Đây là hành vi hiện có, không được đổi.

---

### Task 1: Xác minh API thật của ChunkFormer trong Docker

Đây là task retire rủi ro lớn nhất của cả spec: code ChunkFormer trong `preprocessing_v2/asr.py` chưa từng chạy một lần nào, kể cả dòng import cũng chưa được xác nhận. Không làm task nào khác trước task này.

**Files:**
- Modify: `layer_1/requirements.txt`
- Create: `layer_1/_probe_chunkformer.py` (script dùng một lần, xóa ở Step 6)

- [ ] **Step 1: Thêm dependency vào requirements.txt**

```
transformers==4.46.3
opencv-python-headless
tqdm
accelerate
ffmpeg-python
silero-vad
chunkformer
pyctcdecode
kenlm
```

- [ ] **Step 2: Viết script probe**

Tạo `layer_1/_probe_chunkformer.py`:

```python
import wave
import numpy as np

MODEL = "khanhld/chunkformer-ctc-large-vie"

print("== import ==")
import chunkformer
print("chunkformer:", chunkformer.__file__)
print("dir:", [n for n in dir(chunkformer) if not n.startswith("_")])

print("== load ==")
from chunkformer import ChunkFormerModel
model = ChunkFormerModel.from_pretrained(MODEL)
print("type:", type(model))
print("methods:", [n for n in dir(model) if not n.startswith("_")])

print("== decode 3s im lặng ==")
# wave stdlib thay soundfile: soundfile KHÔNG có trong requirements.txt.
with wave.open("/tmp/probe.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(np.zeros(48000, dtype="<i2").tobytes())
out = model.endless_decode(audio_path="/tmp/probe.wav", chunk_size=64,
                           left_context_size=128, right_context_size=128,
                           total_batch_duration=1800, return_timestamps=False)
print("return type:", type(out))
print("value:", repr(out)[:500])
```

- [ ] **Step 3: Build image**

Run: `cd layer_1 && docker build -t ai26-layer1 .`
Expected: build thành công. Nếu `pip install chunkformer` thất bại hoặc xung đột với `torch==2.1.2`, DỪNG LẠI và báo cáo — đó là phát hiện quan trọng nhất của task này, không được tự ý nâng/hạ version torch.

- [ ] **Step 4: Chạy probe**

Run:
```bash
cd layer_1
GPU_ID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')
docker run --rm --gpus "device=$GPU_ID" --entrypoint python3 \
    -v "$PWD/_probe_chunkformer.py:/workspace/_probe_chunkformer.py" \
    -v "$PWD/cache:/root/.cache" \
    ai26-layer1 /workspace/_probe_chunkformer.py
```
Expected: in ra tên class, danh sách method, và kiểu trả về của `endless_decode`.

- [ ] **Step 5: Ghi lại API thật**

Dán output của Step 4 vào phần "Trạng thái" ở cuối file plan này. Nếu tên import hoặc chữ ký hàm khác với giả định (`ChunkFormerModel.from_pretrained` / `endless_decode`), **cập nhật Task 3 Step 3 cho khớp trước khi làm tiếp**. Đây là lý do task này tồn tại.

- [ ] **Step 6: Xóa script probe và commit**

```bash
cd /workingspace_aiclub/WorkingSpace/Personal/vannk/Ai_challange_2026
rm layer_1/_probe_chunkformer.py
git add layer_1/requirements.txt
git add -f docs/superpowers/plans/2026-09-23-layer1-asr-chunkformer-kenlm.md
git commit -m "chore: add chunkformer, pyctcdecode and kenlm to layer_1 image"
```

---

### Task 2: Tách `_build_entries()` thành hàm thuần và pin hành vi hiện tại

Lưới an toàn trước khi đổi decoder. Hiện không test được phần này vì nó nằm giữa `run_asr`, chỉ chạy sau khi đã load Silero VAD và model ASR.

**Files:**
- Modify: `layer_1/gpu_shot_and_asr.py:370-381`
- Create: `layer_1/tests/conftest.py`
- Create: `layer_1/tests/test_build_entries.py`

**Interfaces:**
- Produces: `_build_entries(video_id: str, speech_segments: list[dict], results: list[dict]) -> list[dict]`. `speech_segments` là output của `get_speech_timestamps(..., return_seconds=True)`, mỗi phần tử có `start`/`end` tính bằng giây. `results` là list cùng độ dài, mỗi phần tử có khóa `text`. Trả về list dict schema `{video_id, seg_id, start_ms, end_ms, text}`.

- [ ] **Step 1: Tạo conftest**

Tạo `layer_1/tests/conftest.py` (theo đúng pattern `layer_2/Keyframe_Extracting/tests/conftest.py`):

```python
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
```

- [ ] **Step 2: Viết test thất bại**

Tạo `layer_1/tests/test_build_entries.py`:

```python
from gpu_shot_and_asr import _build_entries


def test_timestamps_come_from_vad_not_model():
    segments = [{"start": 1.25, "end": 2.5}]
    entries = _build_entries("L23_V008", segments, [{"text": "xin chào"}])
    assert entries[0]["start_ms"] == 1250
    assert entries[0]["end_ms"] == 2500


def test_schema_and_seg_id_format():
    segments = [{"start": 0.0, "end": 1.0}]
    entries = _build_entries("L23_V008", segments, [{"text": "a"}])
    assert entries[0] == {
        "video_id": "L23_V008",
        "seg_id": "L23_V008_000000",
        "start_ms": 0,
        "end_ms": 1000,
        "text": "a",
    }


def test_empty_text_skipped_but_still_consumes_index():
    segments = [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]
    results = [{"text": "   "}, {"text": "hai"}]
    entries = _build_entries("L23_V008", segments, results)
    assert len(entries) == 1
    assert entries[0]["seg_id"] == "L23_V008_000001"


def test_text_is_stripped():
    entries = _build_entries("V", [{"start": 0.0, "end": 1.0}], [{"text": "  a  "}])
    assert entries[0]["text"] == "a"
```

- [ ] **Step 3: Chạy test để xác nhận nó thất bại**

Run: `cd layer_1 && python3 -m pytest tests/test_build_entries.py -v`
Expected: FAIL với `ImportError: cannot import name '_build_entries'`

- [ ] **Step 4: Tách hàm ra**

Trong `layer_1/gpu_shot_and_asr.py`, thêm hàm này ngay trước `def run_asr(`:

```python
def _build_entries(video_id, speech_segments, results):
    entries = []
    for seg_idx, (seg, result) in enumerate(zip(speech_segments, results)):
        text = result["text"].strip()
        if not text:
            continue
        entries.append({
            "video_id": video_id,
            "seg_id": f"{video_id}_{seg_idx:06d}",
            "start_ms": int(round(seg["start"] * 1000)),
            "end_ms": int(round(seg["end"] * 1000)),
            "text": text,
        })
    return entries
```

Rồi thay khối lệnh ở dòng 370-381 (từ `entries = []` tới hết `])`) bằng đúng một dòng:

```python
        entries = _build_entries(video_id, speech_segments, results)
```

- [ ] **Step 5: Chạy test để xác nhận pass**

Run: `cd layer_1 && python3 -m pytest tests/test_build_entries.py -v`
Expected: PASS, 4 test.

- [ ] **Step 6: Commit**

```bash
cd /workingspace_aiclub/WorkingSpace/Personal/vannk/Ai_challange_2026
git add layer_1/gpu_shot_and_asr.py layer_1/tests/conftest.py layer_1/tests/test_build_entries.py
git commit -m "refactor: extract _build_entries so ASR schema is testable without GPU"
```

---

### Task 3: Recognizer ChunkFormer và trường `confidence`

**Files:**
- Modify: `layer_1/gpu_shot_and_asr.py` (thêm `_build_chunkformer`, sửa `_build_entries`, sửa `run_asr`, thêm cờ CLI)
- Modify: `layer_1/tests/test_build_entries.py`
- Create: `layer_1/tests/test_chunkformer_recognizer.py`

**Interfaces:**
- Consumes: `_build_entries()` từ Task 2.
- Produces:
  - `_build_chunkformer(model_name: str, kenlm_path: str | None) -> callable`. Callable nhận `(chunks: list[dict], batch_size: int) -> list[dict]`, mỗi dict có `{"text": str, "confidence": float}`. `chunks` là cấu trúc `run_asr` đang dựng sẵn: `[{"array": np.ndarray, "sampling_rate": 16000}, ...]`.
  - `_build_entries()` nay trả thêm khóa `confidence` (mặc định `0.0` khi result không có khóa đó).

- [ ] **Step 1: Viết test thất bại cho confidence trong schema**

Trong `layer_1/tests/test_build_entries.py`, **sửa** `test_schema_and_seg_id_format` cho có thêm khóa mới (test này so sánh dict tuyệt đối nên sẽ vỡ nếu không sửa):

```python
def test_schema_and_seg_id_format():
    segments = [{"start": 0.0, "end": 1.0}]
    entries = _build_entries("L23_V008", segments, [{"text": "a"}])
    assert entries[0] == {
        "video_id": "L23_V008",
        "seg_id": "L23_V008_000000",
        "start_ms": 0,
        "end_ms": 1000,
        "text": "a",
        "confidence": 0.0,
    }
```

Rồi **thêm** hai test mới vào cùng file:

```python
def test_confidence_is_written_when_present():
    entries = _build_entries("V", [{"start": 0.0, "end": 1.0}],
                             [{"text": "a", "confidence": -1.25}])
    assert entries[0]["confidence"] == -1.25


def test_confidence_defaults_to_zero_when_absent():
    entries = _build_entries("V", [{"start": 0.0, "end": 1.0}], [{"text": "a"}])
    assert entries[0]["confidence"] == 0.0
```

- [ ] **Step 2: Viết test thất bại cho recognizer**

Tạo `layer_1/tests/test_chunkformer_recognizer.py`. Test dùng model giả nên không cần GPU, không tải checkpoint:

```python
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
```

- [ ] **Step 3: Chạy test để xác nhận thất bại**

Run: `cd layer_1 && python3 -m pytest tests/ -v`
Expected: FAIL — `AttributeError: module 'gpu_shot_and_asr' has no attribute '_build_chunkformer'`, và 2 test confidence fail vì thiếu khóa.

- [ ] **Step 4: Sửa `_build_entries` thêm confidence**

Trong `_build_entries`, đổi khối `entries.append({...})` thành:

```python
        entries.append({
            "video_id": video_id,
            "seg_id": f"{video_id}_{seg_idx:06d}",
            "start_ms": int(round(seg["start"] * 1000)),
            "end_ms": int(round(seg["end"] * 1000)),
            "text": text,
            "confidence": float(result.get("confidence", 0.0)),
        })
```

- [ ] **Step 5: Thêm recognizer ChunkFormer**

Thêm vào `gpu_shot_and_asr.py`, ngay trước `def run_asr(`. **Nếu Task 1 Step 5 cho thấy API khác, sửa `_load_chunkformer_model` cho khớp trước khi viết:**

```python
def _load_chunkformer_model(model_name):
    from chunkformer import ChunkFormerModel
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ChunkFormerModel.from_pretrained(model_name).to(device)
    model.eval()
    return model


def _chunkformer_text(result):
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return " ".join(_chunkformer_text(item) for item in result).strip()


def _build_chunkformer(model_name, kenlm_path):
    model = _load_chunkformer_model(model_name)
    print(f"Model {model_name} đã load.")

    def recognize(chunks, batch_size):
        results = []
        for chunk in chunks:
            # ChunkFormer nhận đường dẫn file, không nhận mảng -> WAV tạm trên
            # /dev/shm (RAM) vì out_dir nằm trên NFS.
            handle = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False,
                dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
            handle.close()
            try:
                pcm = (np.clip(chunk["array"], -1.0, 1.0) * 32767.0).astype("<i2")
                with wave.open(handle.name, "wb") as wav_out:
                    wav_out.setnchannels(1)
                    wav_out.setsampwidth(2)
                    wav_out.setframerate(chunk["sampling_rate"])
                    wav_out.writeframes(pcm.tobytes())
                with torch.inference_mode():
                    raw = model.endless_decode(
                        audio_path=handle.name, chunk_size=64,
                        left_context_size=128, right_context_size=128,
                        total_batch_duration=1800, return_timestamps=False)
                results.append({"text": _chunkformer_text(raw), "confidence": 0.0})
            finally:
                if os.path.exists(handle.name):
                    os.remove(handle.name)
        return results

    return recognize
```

Thêm vào phần import ở đầu file, cạnh các import chuẩn đang có: `import tempfile`, `import wave`, `import numpy as np`.

- [ ] **Step 6: Nối vào `run_asr`**

Trong `run_asr`, thêm tham số `asr_backend="phowhisper"` và `kenlm_path=None` vào chữ ký hàm. Thay khối dựng pipeline ở dòng 279-285 bằng:

```python
    if asr_backend == "chunkformer":
        asr_pipeline = _build_chunkformer(model_name, kenlm_path)
    else:
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=pipe_device,
            torch_dtype=dtype,
        )
        print(f"Model {model_name} đã load.")
```

- [ ] **Step 7: Thêm cờ CLI**

Trong `main()`, sau `--model_name`, thêm:

```python
    parser.add_argument("--asr_backend", default="phowhisper",
                        choices=["phowhisper", "chunkformer"],
                        help="Backend ASR. chunkformer là đích đến; phowhisper "
                             "chỉ còn để đối chiếu và sẽ bị xóa sau khi có số đo.")
    parser.add_argument("--kenlm_path", default=None,
                        help="File KenLM cho pyctcdecode. Bỏ trống = decode greedy.")
```

Và sửa default của `--model_name` thành `None`, rồi trong thân `main()` giải mặc định theo backend trước khi gọi `run_asr`:

```python
    model_name = args.model_name or (
        "khanhld/chunkformer-ctc-large-vie" if args.asr_backend == "chunkformer"
        else "vinai/PhoWhisper-large")
```

Truyền `asr_backend=args.asr_backend`, `kenlm_path=args.kenlm_path`, `model_name=model_name` vào lời gọi `run_asr`.

- [ ] **Step 8: Viết test hợp đồng với consumer hạ nguồn**

Trường mới chỉ an toàn nếu consumer thật vẫn đọc được. Test này gọi thẳng hàm
thật của layer_2 chứ không mock, nên nó bắt được đúng loại lỗi mà spec mục 11.1
cảnh báo: schema hợp lệ ở layer_1 nhưng làm hỏng ánh xạ shot↔transcript.

Tạo `layer_1/tests/test_downstream_contract.py`:

```python
import json
import sys
from pathlib import Path

_L2 = Path(__file__).resolve().parents[2] / "layer_2" / "shot_transcript"
sys.path.insert(0, str(_L2))

from cpu_map_transcript import run_shot_transcript_mapping
from gpu_shot_and_asr import _build_entries


def test_new_schema_still_maps_to_shots(tmp_path):
    shots = tmp_path / "shots.jsonl"
    whisper = tmp_path / "whisper.jsonl"
    out = tmp_path / "shot_transcripts.jsonl"

    shots.write_text(json.dumps({
        "video_id": "V1", "shot_id": "V1_000001",
        "start_frame": 0, "end_frame": 50, "fps": 25.0,
    }) + "\n", encoding="utf-8")

    entries = _build_entries("V1", [{"start": 0.5, "end": 1.5}],
                             [{"text": "xin chào", "confidence": -0.3}])
    whisper.write_text(json.dumps(entries[0], ensure_ascii=False) + "\n",
                       encoding="utf-8")

    run_shot_transcript_mapping(str(shots), str(whisper), str(out))
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["shot_id"] == "V1_000001"
    assert "xin chào" in rows[0]["text"]
```

- [ ] **Step 9: Chạy test**

Run: `cd layer_1 && python3 -m pytest tests/ -v`
Expected: PASS, 9 test. Nếu `test_new_schema_still_maps_to_shots` fail, DỪNG —
nghĩa là thay đổi schema đã phá ánh xạ shot↔transcript, và lỗi đó sẽ không lộ ra
ở bất kỳ chỗ nào khác cho tới khi tìm kiếm trả kết quả sai.

- [ ] **Step 10: Smoke test 1 video trên GPU**

Run:
```bash
cd layer_1
docker build -t ai26-layer1 .
mkdir -p /tmp/asr_smoke
GPU_ID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t',' -k2 -n -r | head -1 | cut -d',' -f1 | tr -d ' ')
docker run --rm --gpus "device=$GPU_ID" --shm-size=2g \
    -v "$(cd .. && pwd)/dataset_batch1/videos/video:/data/video:ro" \
    -v /tmp/asr_smoke:/data/output \
    -v "$PWD/cache:/root/.cache" \
    ai26-layer1 --input_dir /data/video --output_dir /data/output \
    --skip_shots --asr_backend chunkformer \
    --videos "$(ls ../dataset_batch1/videos/video | head -1)"
head -2 /tmp/asr_smoke/whisper.jsonl
```
Expected: mỗi dòng có đủ 6 khóa gồm `confidence`, và `text` là tiếng Việt đọc được. Quan sát VRAM trong lúc chạy (`nvidia-smi`) và ghi lại đỉnh — đường ChunkFormer không có OOM backoff nên con số này là dữ kiện cho Task 7.

- [ ] **Step 11: Commit**

```bash
cd /workingspace_aiclub/WorkingSpace/Personal/vannk/Ai_challange_2026
git add layer_1/gpu_shot_and_asr.py layer_1/tests/
git commit -m "feat: add chunkformer ASR backend and confidence field"
```

---

### Task 4: Nối KenLM qua pyctcdecode

**Files:**
- Modify: `layer_1/gpu_shot_and_asr.py` (`_build_chunkformer`)
- Create: `layer_1/tests/test_kenlm_wiring.py`

**Interfaces:**
- Consumes: `_build_chunkformer(model_name, kenlm_path)` từ Task 3.
- Produces: khi `kenlm_path` không rỗng, `confidence` trong kết quả là điểm beam thật của `pyctcdecode` thay vì `0.0`.

- [ ] **Step 1: Viết test thất bại**

Tạo `layer_1/tests/test_kenlm_wiring.py`:

```python
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
```

- [ ] **Step 2: Chạy test để xác nhận thất bại**

Run: `cd layer_1 && python3 -m pytest tests/test_kenlm_wiring.py -v`
Expected: FAIL — chưa in log về LM, và chưa kiểm tra file KenLM.

- [ ] **Step 3: Thêm kiểm tra và log vào `_build_chunkformer`**

Thêm vào đầu `_build_chunkformer`, **TRƯỚC** dòng `model = _load_chunkformer_model(model_name)` — fail fast, đừng tải model ~1 GB rồi mới báo thiếu file LM:

```python
    if kenlm_path:
        if not os.path.exists(kenlm_path):
            raise FileNotFoundError(f"Không tìm thấy file KenLM: {kenlm_path}")
        print(f"Decode với KenLM: {kenlm_path}")
    else:
        print("Decode greedy — không có LM, phần sửa lỗi chính tả chưa bật.")
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `cd layer_1 && python3 -m pytest tests/ -v`
Expected: PASS, 11 test.

- [ ] **Step 5: Commit**

```bash
cd /workingspace_aiclub/WorkingSpace/Personal/vannk/Ai_challange_2026
git add layer_1/gpu_shot_and_asr.py layer_1/tests/test_kenlm_wiring.py
git commit -m "feat: validate and log kenlm path in chunkformer decode"
```

> **Ghi chú cho người thực thi:** phần shallow fusion thật (`pyctcdecode.build_ctcdecoder` trên logits của ChunkFormer) **không** nằm trong plan này. Nó bị chặn bởi hai thứ chưa có: file KenLM tiếng Việt, và xác nhận từ Task 1 rằng ChunkFormer expose được logits thô chứ không chỉ text đã decode. Nếu Task 1 cho thấy `endless_decode` chỉ trả text, hãy dừng và báo cáo — khi đó KenLM cần một đường decode khác và phải quay lại spec.

---

### Task 5: `compare_asr.py` để đối chiếu hai backend

**Files:**
- Create: `layer_1/compare_asr.py`
- Create: `layer_1/tests/test_compare_asr.py`

**Interfaces:**
- Produces: `compare(a_path: str, b_path: str) -> dict` với các khóa `videos`, `segments_a`, `segments_b`, `chars_a`, `chars_b`, `empty_a`, `empty_b`, `only_a`, `only_b`, `both`. `only_*`/`both` đếm theo `video_id`.

- [ ] **Step 1: Viết test thất bại**

Tạo `layer_1/tests/test_compare_asr.py`:

```python
import json
from compare_asr import compare


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                    encoding="utf-8")


def test_counts_segments_and_videos(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write(a, [{"video_id": "V1", "text": "xin chào"},
               {"video_id": "V1", "text": "hai"},
               {"video_id": "V2", "text": "ba"}])
    _write(b, [{"video_id": "V1", "text": "xin chao"}])
    out = compare(str(a), str(b))
    assert out["segments_a"] == 3
    assert out["segments_b"] == 1
    assert out["only_a"] == ["V2"]
    assert out["both"] == ["V1"]


def test_counts_empty_text(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write(a, [{"video_id": "V1", "text": "  "}, {"video_id": "V1", "text": "x"}])
    _write(b, [{"video_id": "V1", "text": "x"}])
    out = compare(str(a), str(b))
    assert out["empty_a"] == 1
    assert out["empty_b"] == 0
```

- [ ] **Step 2: Chạy test để xác nhận thất bại**

Run: `cd layer_1 && python3 -m pytest tests/test_compare_asr.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'compare_asr'`

- [ ] **Step 3: Viết `compare_asr.py`**

```python
"""So sánh hai file whisper.jsonl sinh bởi hai backend ASR khác nhau.

Báo cáo độ phủ và khối lượng, KHÔNG phải độ chính xác: nhiều segment hơn hay
nhiều ký tự hơn không có nghĩa là đọc đúng hơn. WER cần ground truth có nhãn,
không có trong repo này.
"""

import argparse
import json


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compare(a_path, b_path):
    a, b = _rows(a_path), _rows(b_path)
    va = {r["video_id"] for r in a}
    vb = {r["video_id"] for r in b}
    return {
        "videos": sorted(va | vb),
        "segments_a": len(a),
        "segments_b": len(b),
        "chars_a": sum(len(r["text"].strip()) for r in a),
        "chars_b": sum(len(r["text"].strip()) for r in b),
        "empty_a": sum(1 for r in a if not r["text"].strip()),
        "empty_b": sum(1 for r in b if not r["text"].strip()),
        "only_a": sorted(va - vb),
        "only_b": sorted(vb - va),
        "both": sorted(va & vb),
    }


def main():
    ap = argparse.ArgumentParser(description="So sánh hai whisper.jsonl")
    ap.add_argument("--a", required=True, help="whisper.jsonl của backend A")
    ap.add_argument("--b", required=True, help="whisper.jsonl của backend B")
    args = ap.parse_args()
    report = compare(args.a, args.b)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nBáo cáo độ phủ, không phải độ chính xác. "
          "Đọc kèm việc nghe lại một mẫu segment trước khi kết luận.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `cd layer_1 && python3 -m pytest tests/ -v`
Expected: PASS, 13 test.

- [ ] **Step 5: Commit**

```bash
cd /workingspace_aiclub/WorkingSpace/Personal/vannk/Ai_challange_2026
git add layer_1/compare_asr.py layer_1/tests/test_compare_asr.py
git commit -m "feat: add compare_asr to measure two ASR backends"
```

---

### Task 6: Script và Docker

**Files:**
- Modify: `layer_1/run_layer1.sh`
- Modify: `layer_1/run_layer1_batch1.sh` (cùng hai biến, vì batch1 mới là script chạy dữ liệu thật)

- [ ] **Step 1: Thêm biến vào `run_layer1.sh` VÀ `run_layer1_batch1.sh`**

Sau dòng `SHOT_THRESHOLD="0.5"` (dòng 33), thêm:

```bash
# Backend ASR: chunkformer là đích đến, phowhisper chỉ còn để đối chiếu.
ASR_BACKEND="${ASR_BACKEND:-chunkformer}"
# File KenLM trên host; rỗng = decode greedy (chưa bật sửa lỗi chính tả).
KENLM_PATH="${KENLM_PATH:-}"
```

- [ ] **Step 2: Mount file KenLM và truyền cờ**

Trong khối `docker run`, thêm vào mảng `MOUNTS` ngay sau dòng xử lý `CLAIMS_DIR`:

```bash
[[ -n "$KENLM_PATH" ]] && MOUNTS+=(-v "$KENLM_PATH:/data/kenlm.arpa:ro")
```

Và thêm hai cờ vào phần tham số truyền cho image, ngay trước `"$@"`:

```bash
    --asr_backend "$ASR_BACKEND" \
    $( [[ -n "$KENLM_PATH" ]] && echo "--kenlm_path /data/kenlm.arpa" ) \
```

- [ ] **Step 3: Kiểm tra script vẫn parse được**

Run: `bash -n layer_1/run_layer1.sh layer_1/run_layer1_batch1.sh && echo "cú pháp OK"`
Expected: in ra `cú pháp OK`.

- [ ] **Step 4: Chạy lại smoke test qua script**

Run:
```bash
cd layer_1
OUTPUT_DIR=/tmp/asr_smoke2 ./run_layer1_batch1.sh --skip_shots \
    --videos "$(ls ../dataset_batch1/videos/video | head -1)"
head -1 /tmp/asr_smoke2/whisper.jsonl
```
Expected: chạy bằng ChunkFormer (mặc định mới), dòng output có đủ 6 khóa.

- [ ] **Step 5: Commit**

```bash
cd /workingspace_aiclub/WorkingSpace/Personal/vannk/Ai_challange_2026
git add layer_1/run_layer1.sh layer_1/run_layer1_batch1.sh
git commit -m "feat: default layer_1 scripts to chunkformer and mount kenlm"
```

---

### Task 7 (CHẶN — cần người duyệt): Xóa PhoWhisper

**KHÔNG tự ý làm task này.** Nó chỉ bắt đầu sau khi con người đã xem báo cáo của `compare_asr.py` trên một mẫu thật và xác nhận ChunkFormer đủ tốt để thay thế. Đây là tiêu chí nghiệm thu số 6 của spec.

**Files:**
- Modify: `layer_1/gpu_shot_and_asr.py`
- Modify: `layer_1/requirements.txt`
- Modify: `layer_1/run_layer1.sh`, `layer_1/run_layer1_batch1.sh`, `layer_1/run_shards.sh`, `layer_1/run_shards_batch1.sh`

- [ ] **Step 1: Dừng lại và lấy xác nhận**

Chạy `compare_asr.py` trên mẫu đã chạy cả hai backend, đưa báo cáo cho người duyệt, và **đợi trả lời**. Nếu chưa có xác nhận, dừng plan ở đây.

- [ ] **Step 2: Xóa nhánh PhoWhisper khỏi `run_asr`**

Trong `gpu_shot_and_asr.py`:
- xóa tham số `asr_backend` khỏi chữ ký `run_asr` và gọi thẳng `_build_chunkformer(model_name, kenlm_path)`;
- xóa `from transformers import pipeline` và các biến `dtype`, `pipe_device` nếu không còn ai dùng;
- xóa vòng lặp OOM backoff ở dòng 354-366 (`for batch in (asr_batch_size, 1)...`), thay bằng lời gọi thẳng:

```python
        try:
            results = asr_pipeline(chunks, batch_size=1) if chunks else []
        except Exception as e:
            print(f"  [ERROR] ChunkFormer lỗi trên {video_file}: {e}")
            continue
```

- xóa tham số `asr_batch_size` khỏi `run_asr` và cờ `--asr_batch_size` khỏi `main()`;
- xóa cờ `--asr_backend` khỏi `main()`, đổi default `--model_name` thành `"khanhld/chunkformer-ctc-large-vie"`;
- sửa docstring/description còn nhắc PhoWhisper.

- [ ] **Step 3: Gỡ dependency không còn dùng**

Trong `layer_1/requirements.txt`, xóa `transformers==4.46.3` và `accelerate` **chỉ khi** `grep -rn "transformers\|accelerate" layer_1/*.py` không còn kết quả nào.

- [ ] **Step 4: Rà cả bốn script**

Run: `grep -n "asr_batch_size\|ASR_BACKEND\|PhoWhisper\|phowhisper" layer_1/*.sh`
Xóa mọi kết quả còn lại. Trong `run_layer1.sh` bỏ luôn biến `ASR_BACKEND` và cờ `--asr_backend` (không còn lựa chọn nào để chọn).

- [ ] **Step 5: Chạy toàn bộ test**

Run: `cd layer_1 && python3 -m pytest tests/ -v`
Expected: PASS. Test nào mock `phowhisper` thì xóa cùng lúc.

- [ ] **Step 6: Smoke test lần cuối**

Run:
```bash
cd layer_1
OUTPUT_DIR=/tmp/asr_final ./run_layer1_batch1.sh --skip_shots \
    --videos "$(ls ../dataset_batch1/videos/video | head -1)"
head -1 /tmp/asr_final/whisper.jsonl
bash -n run_layer1.sh run_layer1_batch1.sh run_shards.sh run_shards_batch1.sh && echo "script OK"
```
Expected: chạy được, output đủ 6 khóa, cả bốn script parse được.

- [ ] **Step 7: Commit**

```bash
cd /workingspace_aiclub/WorkingSpace/Personal/vannk/Ai_challange_2026
git add layer_1/
git commit -m "refactor: drop PhoWhisper now that chunkformer is measured"
```

---

## Ngoài phạm vi plan này

Các phần còn lại của spec cần plan riêng, mỗi plan tự chạy và tự test được:

- **`layer_1/trajectory/`** (spec mục 5) — không chặn bởi gì cả, làm được ngay sau plan này.
- **Adaptive DAKE** (spec mục 6) — không chặn, nhưng bật lên kéo theo chạy lại SigLIP + recap + purge/ingest (spec mục 11.3).
- **PARSeq + đồng thuận layer_3** (spec mục 7) — **bị chặn**: repo chưa có checkpoint PARSeq tiếng Việt.
- **Giải thể `preprocessing_v2/`** (spec mục 8) — làm sau khi trajectory đã lấy xong phần cần dùng.
- **Sửa `ingest.py` đọc output OCR đã merge** (spec mục 11.2, rủi ro âm thầm nhất) — đi cùng plan layer_3.

Chạy lại toàn bộ 853 video và build lại index hạ nguồn (spec mục 4.5) là việc vận hành, không thuộc plan nào ở trên.

## Trạng thái

- Task 1 Step 5 — API thật của ChunkFormer: *(chưa chạy, điền sau)*
- Task 3 Step 10 — đỉnh VRAM quan sát được: *(chưa chạy, điền sau)*
- Task 7 Step 1 — người duyệt xác nhận thay PhoWhisper: *(chưa có)*
