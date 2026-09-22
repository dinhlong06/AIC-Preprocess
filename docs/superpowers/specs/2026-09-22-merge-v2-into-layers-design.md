# Gộp preprocessing v2 vào layer_1/2/3

**Ngày:** 2026-09-22
**Trạng thái:** Chờ review
**Phạm vi:** layer_1, layer_2, layer_3 trong repo gốc `Ai_challange_2026`

## 1. Mục tiêu

Giải thể package `preprocessing_v2/` và đưa phần mới của nó vào đúng layer tương
ứng, theo hướng **bổ sung lựa chọn model** chứ không viết lại pipeline. Backend
mặc định của mỗi layer giữ nguyên như hiện tại, nên mọi script, output và
benchmark cũ không đổi.

Thiết kế này thay thế hướng cách ly của
[2026-09-22-preprocessing-v2-design.md](2026-09-22-preprocessing-v2-design.md),
vốn đặt v2 thành một package độc lập chạy song song.

Hai phần **không** đến từ v2 nhưng nằm trong phạm vi này vì cùng một mục tiêu
"xử lý khác nhau theo dữ liệu thay vì một cấu hình cho tất cả": adaptive DAKE ở
mục 6 và sửa lỗi chính tả ASR ở mục 4b.

## 2. Bối cảnh

Ba điều đã xác minh trong repo, chi phối toàn bộ thiết kế:

1. **layer_2 đã có sẵn pattern cần nhân rộng.** `cli.py` dùng registry
   `_PIPELINE_BUILDERS` ánh xạ tên → (hàm dựng, YAML riêng). Thêm `pipeline_h`
   chỉ là thêm một dòng registry + một file config, không đụng `pipeline_g`.
   Đây là hình mẫu cho layer_1 và layer_3.

2. **Code v2 chưa từng chạy.** README của v2 ghi rõ phiên triển khai không được
   dùng CPU/GPU chung, nên kể cả unit test cũng chưa chạy lần nào. Mọi đoạn code
   mang từ v2 sang đều phải coi là chưa kiểm chứng.

3. **Code layer 1/2/3 đang tồn tại song song ở hai repo.**
   `AIC2026_Artiz_System/offline/01_av_preprocessing/gpu_shot_and_asr.py` giống
   hệt từng byte với `layer_1/gpu_shot_and_asr.py`, và `AIC2026_Artiz_System` là
   git repo riêng (remote `hohoangluan/AIC2026_Artiz_System`). Thiết kế này chỉ
   áp dụng cho `layer_*`; đồng bộ sang Artiz là việc riêng, một chiều, sau khi
   v2 chạy ổn.

## 3. Nguyên tắc chung

Mỗi layer có một registry nhỏ `tên backend → (hàm dựng, model mặc định)`. Backend
mặc định là backend hiện tại. Không có lớp orchestrator nào bắc ngang ba layer:
mỗi layer giữ Docker image, `run.sh` và cơ chế resume riêng như hiện nay, và
"chọn stage để chạy" chính là chọn layer để chạy.

## 4. Layer 1 — ASR đa backend

Chỗ nối nằm ở đúng một điểm: `gpu_shot_and_asr.py:279` dựng
`pipeline("automatic-speech-recognition")`. Mọi thứ còn lại trong `run_asr` đều
độc lập với model: tách audio bằng ffmpeg, Silero VAD, claims/resume, ghi atomic,
OOM backoff batch→1, tmp trên `/dev/shm`. Phần đó đã chạy qua 605 video và
**không được đụng vào**.

**File mới `layer_1/asr_backends.py`:**

```python
ASR_BACKENDS = {
    "phowhisper":  (build_phowhisper,  "vinai/PhoWhisper-large"),
    "chunkformer": (build_chunkformer, "khanhld/chunkformer-ctc-large-vie"),
}
```

Hợp đồng: `build(model_name, device, dtype)` trả về một callable
`(chunks, batch_size) -> list[str]`, trong đó `chunks` đúng là cấu trúc
`run_asr` đã dựng sẵn ở dòng 342-350 (`[{"array": np.ndarray, "sampling_rate": 16000}, ...]`).

- `phowhisper`: giữ nguyên hành vi hiện tại, gồm cả gọi theo batch.
- `chunkformer`: lặp từng segment và gọi `endless_decode()`, vì API của nó nhận
  đường dẫn file chứ không nhận mảng. Logic lấy từ
  `preprocessing_v2/asr.py:25-58`, nhưng **không** mang `PhoWhisperRecognizer`
  của v2 sang: bản đó gọi từng segment một, làm mất batching và OOM backoff mà
  bản layer_1 đang có.

**Sửa trong `gpu_shot_and_asr.py` (~5 dòng):** thay lời gọi `pipeline(...)` bằng
tra cứu registry; thêm `--asr_backend` (mặc định `phowhisper`); để `--model_name`
mặc định theo backend đã chọn.

**Rủi ro cần kiểm chứng sớm:** đoạn ChunkFormer của v2 chưa từng chạy, kể cả
dòng `from chunkformer import ChunkFormerModel` cũng chưa được xác nhận là đúng
API của `khanhld/chunkformer-ctc-large-vie`. Đây là chỗ khả năng sai cao nhất
trong toàn bộ kế hoạch, nên phải thử trước khi làm phần còn lại.

**So sánh backend:** rút gọn phần so sánh shot/ASR của `preprocessing_v2/compare.py`
thành `layer_1/compare_asr.py`, chạy trên CPU trên hai file `whisper.jsonl` đã có
để chọn giữa PhoWhisper và ChunkFormer. Báo cáo coverage/độ dài/tỉ lệ rỗng; WER
chỉ xuất khi có ground truth.

## 4b. Layer 1 — sửa lỗi chính tả ASR, không dùng LLM

Registry ở mục 4 mới chỉ cho **đổi** model, không sửa lỗi chính tả. Ba bước dưới
đây làm theo đúng thứ tự, mỗi bước đo được trước khi sang bước sau.

Quy mô để cân nhắc chi phí: batch1 có **57.846 segment / 853 video**.

### 4b.1 Thêm confidence vào schema (tiền đề)

Schema hiện tại là `{video_id, seg_id, start_ms, end_ms, text}` — không có tín
hiệu nào để biết segment nào đáng ngờ, nên không cơ chế chấm điểm nào ở 4b.3
hoạt động được. Thêm trường `confidence`.

Lấy confidence không đối xứng giữa hai backend, cần biết trước khi ước lượng
công: ChunkFormer là CTC nên `pyctcdecode` trả sẵn điểm beam; PhoWhisper là
seq2seq nên phải lấy qua `output_scores` / `return_dict_in_generate` của
`generate()`, không có trong lời gọi `pipeline()` đơn giản hiện nay.

File `whisper.jsonl` cũ không được retro-fit: muốn dùng cổng ở 4b.3 thì phải
chạy lại ASR để sinh confidence.

### 4b.2 KenLM n-gram rescoring (nền)

`khanhld/chunkformer-ctc-large-vie` là model **CTC**, nên decode được bằng
`pyctcdecode` với shallow fusion của một n-gram LM tiếng Việt (KenLM). Đây là
cách chuẩn để sửa lỗi chính tả, ranh giới từ và dấu bằng thống kê corpus, không
cần LLM.

Chi phí: **không thêm GPU pass nào** — chỉ là đổi bước decode, chạy CPU, cộng
một file LM.

Giới hạn phải nói rõ: chỉ áp dụng được cho ChunkFormer. PhoWhisper là
encoder-decoder, không có CTC logits, nên `pyctcdecode` không dùng được cho nó.
Backend `phowhisper` giữ nguyên đường decode hiện tại.

Config: khối `kenlm: {model_path, alpha, beta}` trong cấu hình ASR, mặc định
`model_path: null` → decode greedy như hiện nay.

### 4b.3 Đối chứng nhiều model, có cổng chi phí

Chạy cả PhoWhisper lẫn ChunkFormer trên toàn bộ 57.846 segment là gấp đôi GPU
time ASR trên cụm dùng chung. Nên áp đúng khuôn cổng chi phí của layer_3 (mục
7.3): model thứ hai **chỉ chạy trên segment đáng ngờ** — confidence dưới ngưỡng
cấu hình được.

Với segment đã chạy hai model, chọn theo điểm (điểm CTC + điểm LM ở 4b.2) chứ
không gọi LLM. Đây là dạng rút gọn của ROVER: bỏ phiếu theo từ giữa các hệ
thống, lấy confidence làm trọng số.

Chỉ làm bước này sau khi 4b.1 và 4b.2 đã chạy và đo được tỉ lệ segment
confidence thấp thực tế — con số đó quyết định cổng có đáng làm không.

## 5. Layer 1 — trajectory (folder mới)

`layer_1/trajectory/` là capability hoàn toàn mới, không có sẵn trong layer nào.
Đặt ở layer_1 vì nó đọc thẳng video gốc với sampling cố định.

Chuyển từ v2: `trajectory.py`, `trajectory_math.py`, `trajectory_types.py`,
`trajectory_ultralytics.py`, `trajectory_otvision.py`, `configs/trajectory.yaml`
và các test tương ứng. Viết mới: `Dockerfile`, `requirements.txt`,
`run_trajectory.sh` theo khuôn `run_layer1.sh` (tự chọn GPU rảnh nhất qua
`nvidia-smi`, mount cache, `--shm-size`).

Giữ nguyên theo spec v2: backend Ultralytics + BoT-SORT mặc định, bù chuyển động
camera, và bộ nhãn sự kiện hạn chế (`turn_left`, `turn_right`, `accelerating`,
`decelerating`, `steady_motion`, `unknown` — không có `hard_brake`). Track thô
ghi tách khỏi event để chỉnh ngưỡng event mà không phải chạy lại detection.

**Resume:** viết lại theo convention layer_1 (`.done` + `claims_dir`) thay cho
`manifest.json` của v2, để `run_shards.sh` dùng lại được.

**OTVision:** giữ làm backend thứ hai theo spec v2, cài trong Dockerfile của
`layer_1/trajectory/` chứ không phải của `layer_1/`. Lý do: OTVision kéo theo
torch/ultralytics bản riêng, trong khi `layer_1/Dockerfile:32` pin cứng
`torch==2.1.2+cu118` để khớp TensorFlow 2.13 — xung đột ở đó sẽ làm hỏng image
layer_1 đang chạy tốt. Để riêng thì hỏng cũng chỉ hỏng trajectory. OTVision là
GPL-3.0, nên nếu image hoặc repo được phát hành kèm bài nộp thì cần ghi nhận
trong `THIRD_PARTY_NOTICES`.

## 6. Layer 2 — adaptive DAKE

Việc chọn pipeline đã pluggable sẵn, nên phần registry không phải làm gì. Nhưng
có một chỗ **không** thích nghi theo video: DAKE.

### 6.1 Hiện trạng

`dake.py:104` tính `k = max(1, int(len(frames) * self.candidate_ratio))`.
`candidate_ratio` là hằng số từ YAML, áp cho mọi shot của mọi video. Hệ quả:

- shot 1000 frame tĩnh (người dẫn ngồi yên), ratio 0,05 → vẫn lấy 50 candidate,
  đốt BEiT-3 vô ích;
- shot 40 frame chuyển động dữ dội → chỉ 2 candidate, sót nội dung.

Cơ chế thích nghi *có* tồn tại nhưng nằm **sau** DAKE, trong
`semantic_filter.py:162-163`: gap decay nới threshold khi gap vượt
`gap_decay_start_frames` (tối đa 0,05) và force-pick khi vượt `max_gap_frames`.
Đó là cứu vãn ở hạ nguồn — DAKE không đưa candidate vào thì semantic filter
không có gì để chọn. DAKE là trần trên của toàn pipeline.

### 6.2 Thiết kế

Thay top-k-theo-ratio bằng ngưỡng thống kê trên chính mảng `aggregated` mà
`select_with_scores()` đã tính sẵn — không thêm một phép tính nào:

- giữ frame có `aggregated` vượt percentile P của **chính shot đó**;
- clamp số lượng trong `[min_per_shot, len(frames) * candidate_ratio]`, tức
  `candidate_ratio` đổi vai từ "tỉ lệ cố định" thành "trần an toàn";
- vẫn ép giữ frame đầu shot như hiện nay.

Shot tĩnh có phân phối score phẳng nên ít frame vượt ngưỡng; shot động nhiều
peak nên nhiều frame vượt. Thích nghi theo **từng shot**, mịn hơn theo loại
video, và không cần gán nhãn loại video.

Repo đã có tiền lệ đúng họ thống kê này: `transition_selector` dùng
`peak_percentile: 90.0` cùng prominence window cho mục đích khác.

Config: thêm khối `dake.adaptive: {enabled: false, percentile, min_per_shot}`.
Mặc định tắt → `pipeline_g`/`pipeline_h` chạy y hệt hiện nay; bật lên mới là
hành vi mới. Không đụng `DAKE_THREADS` (mặc định 2, đang giới hạn cho máy dùng
chung).

### 6.3 Việc còn lại của layer 2

Commit phần `pipeline_h` đang dở dang trong working tree của `main`
(`cli.py`, `src/extractors/pipeline_h.py`, `configs/pipeline_h.yaml`,
`src/components/text_prescan.py`, `src/components/semantic_filter.py` và test đi
kèm). Chưa commit thì trên một checkout sạch sẽ không có `pipeline_h`.

Bỏ `preprocessing_v2/keyframes.py` — nó chỉ là wrapper gọi
`layer_2/Keyframe_Extracting/run.sh`, không còn tác dụng khi không có orchestrator.

## 7. Layer 3 — thêm PARSeq, đồng thuận có trọng tài

### 7.1 Hiện trạng

Paddle đóng vai **detector**. Text của recognizer Paddle đi kèm miễn phí trong
cùng lời gọi `predict()` (`paddle_engine.py:105-107` lấy cả `rec_boxes` lẫn
`rec_texts`), nên giữ nó làm ý kiến tham chiếu không tốn thêm GPU.

`merge_recognizers.py` là hệ thống trọng tài đã đo đạc kỹ: VietOCR thắng mặc
định, trừ các trường hợp có bằng chứng ngược — từ số/giờ và từ ASCII thuần thì
Paddle thắng (VietOCR bị đo là bịa số, `"9 TRIEU"` → `"99 TRIEU"`), dòng ≤2 từ
in hoa thì Paddle thắng (đúng 17/28 so với VietOCR 6/28 trên các bất đồng thật).

### 7.2 Thiết kế mới

Thêm PARSeq thành recognizer thứ hai đọc crop của detector. Luồng mới:

```text
frame
  └─ PaddleOCR.predict()  →  boxes + rec_texts (Paddle, miễn phí)
       └─ crops  ─┬→ VietOCR  → output_vietocr.json
                  └→ PARSeq   → output_parseq.json        [MỚI]

merge_recognizers.py  [MỞ RỘNG]
  ├─ VietOCR == PARSeq                      → chốt, đi tiếp
  ├─ chỉ một bên có text                    → lấy bên đó
  ├─ khớp luật đã đo (số / ASCII thuần /
  │  dòng in hoa ngắn, tham chiếu Paddle)   → theo luật
  └─ còn lại                                → đánh dấu unresolved

run_vlm_correct.py (Qwen3-VL-4B)  →  chỉ chạy trên box unresolved
```

"Đồng thuận" định nghĩa là khớp chuỗi tuyệt đối sau `strip()`. Không chuẩn hóa
dấu trước khi so: đo trên 4.342 box cho thấy bỏ dấu gần như không đổi tỉ lệ đồng
thuận (68,2% → 68,8%), tức bất đồng đến từ đọc khác nhau thật chứ không phải
khác biệt dấu, nên chuẩn hóa chỉ thêm phức tạp mà không giảm khối lượng.

### 7.3 Vì sao phải có trọng tài trước VLM

Đo trên batch1 thật: **557.567 box / 163.211 frame**. Với ~0,45 s/box của
Qwen3-VL-4B:

| Trigger escalation | Số box | GPU time |
|---|---|---|
| Hiện tại (blob dính liền, ~0,55%) | ~3.000 | ~23 phút |
| Bất đồng 10% | ~56.000 | ~7 giờ |
| Bất đồng ~30% | ~167.000 | ~21 giờ |

30% là tỉ lệ đo được giữa VietOCR và recognizer Paddle — chỉ dùng làm proxy, vì
chưa có checkpoint PARSeq để đo tỉ lệ thật VietOCR↔PARSeq. Kể cả ở kịch bản lạc
quan nhất, đẩy thẳng mọi bất đồng sang VLM vẫn là hàng giờ GPU trên cụm dùng
chung. Trọng tài CPU chạy trước giải quyết miễn phí các nhóm token đã có bằng
chứng và chặn khối lượng VLM.

### 7.4 Thay đổi cụ thể

- `paddle_engine.py`: `run()` trả thêm phần tử thứ ba cho PARSeq; đưa ba hằng số
  VietOCR đang hardcode ở dòng 76-79 (`vgg_seq2seq`, `cuda:0`, `beamsearch`) lên
  `config.yaml`.
- `config.yaml`: thêm khối `parseq` (`model_path`, `enabled`) và
  `output_file_parseq`. Mặc định `enabled: false` vì repo chưa có checkpoint
  PARSeq tiếng Việt; bật lên khi có checkpoint. Khi tắt, luồng chạy y hệt hiện nay.
- `merge_recognizers.py`: thêm nhánh đồng thuận VietOCR↔PARSeq và cờ
  `unresolved` cho mỗi box.
- `run_vlm_correct.py`: đổi điều kiện chọn box từ heuristic "blob dính liền"
  sang cờ `unresolved`.

Bỏ `preprocessing_v2/ocr.py` và phần preflight PARSeq của nó: nó bọc layer_3 như
một tiến trình con, không còn ý nghĩa sau khi giải thể.

## 8. Phần v2 bị bỏ

`orchestrator.py`, `manifest.py`, `config.py`, `io.py`, `preflight.py`,
`shot.py`, `keyframes.py`, `ocr.py`, `run_v2.py`, `run_v2.sh`, `Dockerfile`,
`pyproject.toml`. Lý do: layer_* đã có resume riêng (`.done` + `claims_dir`),
mỗi layer có Docker image và `run.sh` riêng, và chọn stage chính là chọn layer.
Giữ lại một lớp điều phối thứ hai sẽ tạo đúng kiểu nhân đôi mà mục 2.3 muốn tránh.

`shot.py` bỏ được vì v2 không đổi model shot detection, chỉ đổi ngưỡng — mà
`--shot_threshold` đã có sẵn trong `run_layer1.sh:33`.

`compare.py` và `compare_v1_v2.py` chỉ giữ phần so sánh shot/ASR, thành
`layer_1/compare_asr.py` (mục 4); phần keyframe/OCR/trajectory của chúng bỏ.
Test của v2 đi theo module tương ứng: test trajectory và test OTVision chuyển
sang `layer_1/trajectory/`, phần còn lại (`test_config`, `test_orchestrator`,
`test_io_manifest`, `test_preflight`, `test_cli_help`, `test_stage_adapters`) bỏ
cùng module mà chúng kiểm thử.

## 9. Script và chi phí build lại image

`run_layer1.sh` đã có `"$@"` passthrough nên cờ mới dùng được ngay; chỉ thêm
biến `ASR_BACKEND` ở đầu file cho đồng bộ style với `SHOT_THRESHOLD`/`VAD_*`.

`run_layer1.sh:45` luôn gọi `docker build` mỗi lần chạy, nhưng nhờ layer cache
chi phí phụ thuộc vào file bị sửa:

| Sửa gì | Layer rebuild | Thời gian |
|---|---|---|
| `gpu_shot_and_asr.py` (COPY, layer cuối) | 1 layer | vài giây |
| `requirements.txt` (thêm chunkformer) | từ `Dockerfile:35` xuống | vài phút |
| base image / torch (`Dockerfile:3,32`) | ~5,5 GB | rất lâu |

Thêm package vào `requirements.txt` không phải dựng lại toàn bộ image: base
TensorFlow 2.13, apt, ffmpeg static và torch cu118 vẫn giữ cache.

## 10. Trọng số model

Không commit file `.pt`/`.ckpt`/`.bin` vào git. Theo convention đã có trong repo
(`AIC2026_Artiz_System/offline/04_embedding_extraction/run.sh:42-43`): mount một
thư mục cache cố định của host vào container và set `HF_HOME`.

| Model | Nguồn | Nơi lưu |
|---|---|---|
| ChunkFormer, PhoWhisper | tự tải từ HF Hub | `layer_1/cache` đã mount sẵn vào `/root/.cache` |
| YOLO `yolov8x-oiv7.pt` | ultralytics tự tải | cache riêng của `layer_1/trajectory/`, mount tương tự |
| PARSeq tiếng Việt | **chưa có, phải tự cung cấp** | ngoài git, trỏ bằng `parseq.model_path` trong `config.yaml` |
| Qwen3-VL-4B | đã có sẵn | `layer_3/OCR_v2/models/qwen3-vl-4b-vi-ocr` |

## 11. Kiểm thử và nghiệm thu

Test viết trước khi implement từng phần. Test CPU không cần GPU/mạng:

- registry ASR: chọn đúng backend, `--model_name` mặc định theo backend, backend
  lạ báo lỗi rõ ràng;
- `chunks` mà backend nhận đúng cấu trúc `run_asr` đang dựng;
- định tuyến prefix trajectory và `--trajectory-all`;
- track tổng hợp: đi thẳng, rẽ trái, rẽ phải, tăng tốc, giảm tốc, chất lượng kém;
- bù chuyển động camera trên pan tổng hợp;
- chuyển đổi bản ghi OTVision mà không import OTVision;
- đồng thuận VietOCR↔PARSeq, thứ tự ưu tiên các luật trọng tài, và cờ
  `unresolved` được đặt đúng;
- adaptive DAKE: shot có score phẳng cho ít candidate hơn shot nhiều peak, clamp
  `min_per_shot` và trần `candidate_ratio` đều có hiệu lực, và `enabled: false`
  cho ra đúng tập candidate như thuật toán top-k hiện nay;
- ASR ghi trường `confidence`, và `kenlm.model_path: null` cho ra đúng kết quả
  decode greedy như hiện nay.

Nghiệm thu:

1. Output v1 đang tracked không có diff khi chạy với backend mặc định.
2. Test CPU pass.
3. `layer_3` với `parseq.enabled: false` cho kết quả giống hệt hiện nay.
4. `layer_2` với `dake.adaptive.enabled: false` cho keyframe giống hệt hiện nay.
5. Một lần chạy GPU smoke trên 1 video cho mỗi phần mới (ChunkFormer, trajectory).
6. `compare_asr.py` chạy được trên hai file `whisper.jsonl`.

## 12. Rủi ro còn mở

- **ChunkFormer chưa kiểm chứng** (mục 4). Thử trước tiên.
- **Không có checkpoint PARSeq tiếng Việt.** Nhánh đồng thuận ở mục 7.2 chỉ chạy
  được khi có checkpoint; trước đó `parseq.enabled: false` và luồng giữ nguyên
  như hiện nay.
- **Tỉ lệ bất đồng VietOCR↔PARSeq chưa đo được.** Khi có checkpoint, đo trên mẫu
  ~1.000 box trước khi chạy toàn bộ batch1, để biết khối lượng VLM thực tế.
- **Chưa có KenLM tiếng Việt trong repo** (mục 4b.2). Phải train từ corpus tin
  tức hoặc lấy bản có sẵn; `alpha`/`beta` cần chỉnh trên tập nhỏ có ground truth
  trước khi tin.
- **Confidence của PhoWhisper không lấy được từ `pipeline()`** (mục 4b.1), phải
  đổi sang `generate()` với `return_dict_in_generate`. Đây là thay đổi trong
  đường chạy mặc định đang ổn định, nên làm riêng và đối chiếu output trước/sau.
- **Percentile của adaptive DAKE chưa có giá trị đã đo** (mục 6.2). Chọn P bằng
  cách chạy trên vài shot tĩnh và vài shot động rồi đối chiếu số candidate, chứ
  không lấy 90,0 của `transition_selector` làm mặc định vì đó là thống kê cho
  mục đích khác.
- **Phân kỳ với `AIC2026_Artiz_System`** (mục 2.3) sẽ rộng thêm sau thay đổi này.
