# Gộp preprocessing v2 vào layer_1/2/3

**Ngày:** 2026-09-22
**Trạng thái:** Chờ review
**Phạm vi:** layer_1, layer_2, layer_3 trong repo gốc `Ai_challange_2026`

## 1. Mục tiêu

Giải thể package `preprocessing_v2/` và đưa phần mới của nó vào đúng layer tương
ứng, theo hướng bổ sung chứ không viết lại pipeline.

Layer 2 và layer 3 giữ nguyên hành vi mặc định: tính năng mới nằm sau cờ config
mặc định tắt, nên script, output và benchmark cũ không đổi.

Layer 1 là ngoại lệ có chủ ý: ASR **thay hẳn** PhoWhisper bằng ChunkFormer +
KenLM (mục 4), nên output ASR sẽ khác và phải chạy lại. Đây là thay đổi duy nhất
trong spec này phá vỡ tính tương thích ngược, và mục 4.5 quy định trình tự đo
trước khi bỏ đường cũ.

Thiết kế này thay thế hướng cách ly của
[2026-09-22-preprocessing-v2-design.md](2026-09-22-preprocessing-v2-design.md),
vốn đặt v2 thành một package độc lập chạy song song.

Hai phần **không** đến từ v2 nhưng nằm trong phạm vi này vì cùng một mục tiêu
"xử lý khác nhau theo dữ liệu thay vì một cấu hình cho tất cả": adaptive DAKE ở
mục 6 và sửa lỗi chính tả ASR bằng KenLM ở mục 4.

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

Không có lớp orchestrator nào bắc ngang ba layer: mỗi layer giữ Docker image,
`run.sh` và cơ chế resume riêng như hiện nay, và "chọn stage để chạy" chính là
chọn layer để chạy.

Nơi nào thật sự có nhiều lựa chọn cùng tồn tại thì dùng registry như
`_PIPELINE_BUILDERS` của layer 2 đang làm. Nơi nào chỉ có một đường chạy — ASR
sau mục 4 — thì gọi thẳng, không dựng registry cho một phần tử.

## 4. Layer 1 — ASR chuyển sang ChunkFormer + KenLM

ASR đi về **một đường duy nhất**: ChunkFormer decode bằng `pyctcdecode` với
shallow fusion của KenLM tiếng Việt. PhoWhisper bị bỏ. Không có registry backend
— một backend thì registry là lớp trừu tượng không kiếm được chỗ đứng.

### 4.1 Vì sao ChunkFormer + KenLM

`khanhld/chunkformer-ctc-large-vie` là model **CTC**, nên decode được bằng
`pyctcdecode` + KenLM. Đó là cách chuẩn để sửa lỗi chính tả, ranh giới từ và dấu
bằng thống kê corpus, không cần LLM, và **không thêm GPU pass nào** — chỉ đổi
bước decode, phần LM chạy CPU.

PhoWhisper là encoder-decoder, không có CTC logits, nên không dùng được
`pyctcdecode`. Đây là lý do kỹ thuật khiến không thể giữ cả hai mà vẫn hưởng
KenLM: cơ chế sửa lỗi chỉ tồn tại trên nhánh CTC.

### 4.2 Thay đổi trong `gpu_shot_and_asr.py`

Giữ nguyên, **không đụng vào**: tách audio bằng ffmpeg, Silero VAD, claims/resume,
ghi atomic, tmp trên `/dev/shm`. Phần đó đã chạy qua 605 video.

Thay đổi, nhiều hơn một chỗ nối — cần lường trước:

- dòng 279: thay `pipeline("automatic-speech-recognition")` bằng khởi tạo
  ChunkFormer + `pyctcdecode` decoder;
- dòng 342-350: `chunks` hiện dựng dạng `{"array", "sampling_rate"}` cho HF
  pipeline; ChunkFormer nhận đường dẫn file nên chuyển sang ghi WAV tạm từng
  segment (logic ở `preprocessing_v2/asr.py:39-58`);
- dòng 354-366: vòng lặp batch với OOM backoff batch→1 **thành mã chết**, vì
  ChunkFormer tự gom theo `total_batch_duration` chứ không nhận `batch_size`.
  Gỡ bỏ cùng cờ `--asr_batch_size`.

Nói cách khác đây không phải sửa 5 dòng như bản trước của spec này ước lượng:
mất luôn cơ chế OOM backoff đã được viết riêng cho GPU dùng chung, nên phải xác
nhận ChunkFormer có hành vi VRAM chấp nhận được trước khi tin.

### 4.3 Schema và confidence

Thêm trường `confidence` vào mỗi dòng `whisper.jsonl`:
`{video_id, seg_id, start_ms, end_ms, text, confidence}`.

`pyctcdecode` trả sẵn điểm beam nên trường này gần như miễn phí. Nó cần thiết vì
khi chỉ còn một model, **không còn ý kiến thứ hai nào** để phát hiện segment
hỏng — điểm của chính nó là tín hiệu chất lượng duy nhất còn lại.

File `whisper.jsonl` cũ không được retro-fit.

### 4.4 Config

Khối `kenlm: {model_path, alpha, beta}`. `model_path: null` → decode greedy,
tức ChunkFormer trần không có sửa lỗi. Đây là chế độ suy giảm để chạy được khi
chưa có file LM, không phải mặc định mong muốn.

### 4.5 Trình tự bỏ PhoWhisper

PhoWhisper là đường đang chạy production, đã sinh 57.846 segment / 853 video.
ChunkFormer là code chưa từng chạy một lần nào (kể cả dòng
`from chunkformer import ChunkFormerModel` cũng chưa được xác nhận đúng API), và
KenLM tiếng Việt thì repo chưa có. Bỏ đường cũ trước khi đường mới được đo là
đánh đổi không lấy lại được.

Nên trình tự là:

1. dựng đường ChunkFormer + KenLM song song, chạy trên một mẫu nhỏ;
2. `layer_1/compare_asr.py` (rút gọn từ `preprocessing_v2/compare.py`) đối chiếu
   hai file `whisper.jsonl` trên cùng mẫu đó — coverage, độ dài, tỉ lệ rỗng; WER
   chỉ xuất khi có ground truth;
3. đạt kết quả thì **xóa hẳn** nhánh PhoWhisper khỏi `gpu_shot_and_asr.py`,
   `requirements.txt` và `run_layer1.sh`, rồi chạy lại toàn bộ 853 video.

Bước 3 là xóa thật, không để lại cờ hay nhánh chết. Trước bước 3, sự tồn tại của
PhoWhisper là để đo, không phải để phòng hờ.

Chi phí cần biết trước: sau khi đổi, toàn bộ `whisper.jsonl` hiện có trở thành
output của model cũ, nên phải chạy lại ASR cho cả 853 video, và
`layer_2/shot_transcript` cùng index BM25 hạ nguồn phải build lại theo.

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

`run_layer1.sh` đã có `"$@"` passthrough nên cờ mới dùng được ngay. Bỏ
`--asr_batch_size` khỏi script cùng lúc với việc gỡ vòng lặp batch ở mục 4.2;
thêm biến `KENLM_PATH` ở đầu file cho đồng bộ style với `SHOT_THRESHOLD`/`VAD_*`,
và mount file LM vào container như một volume read-only.

`run_layer1.sh:45` luôn gọi `docker build` mỗi lần chạy, nhưng nhờ layer cache
chi phí phụ thuộc vào file bị sửa:

| Sửa gì | Layer rebuild | Thời gian |
|---|---|---|
| `gpu_shot_and_asr.py` (COPY, layer cuối) | 1 layer | vài giây |
| `requirements.txt` (thêm chunkformer, pyctcdecode, kenlm; bỏ transformers nếu không còn ai dùng) | từ `Dockerfile:35` xuống | vài phút |
| base image / torch (`Dockerfile:3,32`) | ~5,5 GB | rất lâu |

Thêm package vào `requirements.txt` không phải dựng lại toàn bộ image: base
TensorFlow 2.13, apt, ffmpeg static và torch cu118 vẫn giữ cache.

## 10. Trọng số model

Không commit file `.pt`/`.ckpt`/`.bin` vào git. Theo convention đã có trong repo
(`AIC2026_Artiz_System/offline/04_embedding_extraction/run.sh:42-43`): mount một
thư mục cache cố định của host vào container và set `HF_HOME`.

| Model | Nguồn | Nơi lưu |
|---|---|---|
| ChunkFormer | tự tải từ HF Hub | `layer_1/cache` đã mount sẵn vào `/root/.cache` |
| KenLM tiếng Việt | **chưa có, phải train hoặc tải** | ngoài git, trỏ bằng `kenlm.model_path` |
| YOLO `yolov8x-oiv7.pt` | ultralytics tự tải | cache riêng của `layer_1/trajectory/`, mount tương tự |
| PARSeq tiếng Việt | **chưa có, phải tự cung cấp** | ngoài git, trỏ bằng `parseq.model_path` trong `config.yaml` |
| Qwen3-VL-4B | đã có sẵn | `layer_3/OCR_v2/models/qwen3-vl-4b-vi-ocr` |

## 11. Kiểm thử và nghiệm thu

Test viết trước khi implement từng phần. Test CPU không cần GPU/mạng:

- ASR ghi đúng schema có `confidence`, và timestamp segment vẫn lấy từ VAD chứ
  không từ model (đây là bất biến dễ vỡ nhất khi đổi decoder);
- `kenlm.model_path: null` chạy được ở chế độ greedy và ghi rõ trong log rằng
  đang chạy không có LM;
- định tuyến prefix trajectory và `--trajectory-all`;
- track tổng hợp: đi thẳng, rẽ trái, rẽ phải, tăng tốc, giảm tốc, chất lượng kém;
- bù chuyển động camera trên pan tổng hợp;
- chuyển đổi bản ghi OTVision mà không import OTVision;
- đồng thuận VietOCR↔PARSeq, thứ tự ưu tiên các luật trọng tài, và cờ
  `unresolved` được đặt đúng;
- adaptive DAKE: shot có score phẳng cho ít candidate hơn shot nhiều peak, clamp
  `min_per_shot` và trần `candidate_ratio` đều có hiệu lực, và `enabled: false`
  cho ra đúng tập candidate như thuật toán top-k hiện nay.

Nghiệm thu:

1. Test CPU pass.
2. `layer_3` với `parseq.enabled: false` cho kết quả giống hệt hiện nay.
3. `layer_2` với `dake.adaptive.enabled: false` cho keyframe giống hệt hiện nay.
4. Output shot detection không có diff — đổi ASR không được đụng nhánh TransNetV2.
5. Một lần chạy GPU smoke trên 1 video cho ChunkFormer + KenLM và cho trajectory.
6. `compare_asr.py` cho ra báo cáo đối chiếu PhoWhisper↔ChunkFormer trên mẫu
   đã chọn, và kết quả đó được review **trước** khi xóa nhánh PhoWhisper.

ASR không có tiêu chí "không diff": đổi model thì output đổi theo định nghĩa.
Thay vào đó tiêu chí là số 6 — bằng chứng đo được, không phải diff trống.

## 12. Rủi ro còn mở

- **Toàn bộ ASR dồn vào một đường chưa kiểm chứng** (mục 4). ChunkFormer chưa
  chạy lần nào và KenLM thì chưa có, trong khi PhoWhisper đang chạy được. Đây là
  rủi ro tập trung lớn nhất của spec; mục 4.5 giữ nó lại bằng trình tự đo trước,
  xóa sau. Thử ChunkFormer trước mọi việc khác.
- **Mất OOM backoff khi bỏ đường PhoWhisper** (mục 4.2). Vòng lặp batch→1 được
  viết riêng cho GPU dùng chung có co-tenant; ChunkFormer tự gom theo
  `total_batch_duration` nên cơ chế đó không còn. Phải quan sát VRAM thực tế
  trong lần smoke đầu tiên.
- **Không có checkpoint PARSeq tiếng Việt.** Nhánh đồng thuận ở mục 7.2 chỉ chạy
  được khi có checkpoint; trước đó `parseq.enabled: false` và luồng giữ nguyên
  như hiện nay.
- **Tỉ lệ bất đồng VietOCR↔PARSeq chưa đo được.** Khi có checkpoint, đo trên mẫu
  ~1.000 box trước khi chạy toàn bộ batch1, để biết khối lượng VLM thực tế.
- **Chưa có KenLM tiếng Việt trong repo** (mục 4.4). Phải train từ corpus tin
  tức hoặc lấy bản có sẵn; `alpha`/`beta` cần chỉnh trên tập nhỏ có ground truth
  trước khi tin. Đây là phụ thuộc chặn: không có nó thì phần sửa lỗi chính tả —
  lý do chính để đổi sang ChunkFormer — chưa hoạt động.
- **Chi phí chạy lại toàn bộ ASR** (mục 4.5): 853 video, kéo theo build lại
  `layer_2/shot_transcript` và index BM25 hạ nguồn.
- **Percentile của adaptive DAKE chưa có giá trị đã đo** (mục 6.2). Chọn P bằng
  cách chạy trên vài shot tĩnh và vài shot động rồi đối chiếu số candidate, chứ
  không lấy 90,0 của `transition_selector` làm mặc định vì đó là thống kê cho
  mục đích khác.
- **Phân kỳ với `AIC2026_Artiz_System`** (mục 2.3) sẽ rộng thêm sau thay đổi này.
