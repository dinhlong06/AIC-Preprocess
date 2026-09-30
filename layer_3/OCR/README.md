# OCR Module -- AIC2026

Trích text tiếng Việt từ keyframe bằng **PaddleOCR PP-OCRv6** trên GPU, tự
hiệu đính dấu ngay trong từng frame (`ocr/corrector.py`).

> Đường chạy **production hiện tại là `../OCR_gemma/gemma_ocr.py`** — Gemma 4
> qua API (UIT + Google AI Studio), chất lượng tốt hơn Paddle trên ảnh bản tin.
> Module này giữ lại để sinh lại output Paddle/VietOCR cũ và làm đối chiếu.

Các hướng đã bị loại (không còn code ở đây, lý do trong git history):

- **Qwen3-VL làm OCR** (`OCR_v2`): chậm hơn ~4x, hay lặp, recall 72%.
- **VLM sửa box gộp chữ** (vLLM host, stage 3 `vlm_correct`): chỉ ~0.55% box
  bị ảnh hưởng, không đáng một model 8GB + pipeline 3 tầng.
- **DeepSolo + PARSeq**: recall 93.6% > 75.2% của Paddle, nhưng output không
  ai dùng và build context không track được nên clone mới không dựng lại nổi.

---

## Project Structure

```
OCR/
├── run_paddle.py                 # CLI entry point
├── run_paddle.sh                 # build+run trong Docker, tự chọn GPU rảnh nhất
├── run_paddle_batch1_shards.sh   # worker pool: add / status / release / merge
├── config.yaml                   # tham số stage 1
├── requirements.txt
├── Dockerfile
├── ocr/
│   ├── paddle_engine.py   # PaddleOCR det + VietOCR/Paddle rec
│   ├── corrector.py       # hiệu đính dấu tiếng Việt, chạy local CPU
│   ├── frame_skip.py      # blank/blur skip + CLAHE/news-band preprocessing
│   ├── loader.py          # frame folder scanner
│   ├── formatter.py       # JSON output + checkpoint I/O
│   └── pipeline.py        # orchestrator
└── output/                # kết quả chạy (git-ignored)
```

---

## Usage

```bash
./run_paddle.sh                                  # 1 GPU, container đầy đủ
FRAMES_DIR=/path ./run_paddle.sh                 # thư mục keyframe khác
OUTPUT_DIR=/path ./run_paddle.sh                 # ghi JSON ra chỗ khác
./run_paddle.sh --limit 60                       # flag thừa forward cho run_paddle.py

NSHARDS=4 ./run_paddle_batch1_shards.sh          # nhiều GPU nền
./run_paddle_batch1_shards.sh status             # xem tiến độ
./run_paddle_batch1_shards.sh merge              # gộp claims/done -> output
```

---

## Output

`output/` chứa JSON, mỗi frame một record:

```json
{"frame_id": "L21_V001_000000_kf0001", "boxes": [{"text": "HTV7", "confidence": 0.93, "bbox": [..]}]}
```

`frame_id` khớp với `keyframe_id` của layer 2 để `layer_5` join.
`output_paddle_origin.json` là kết quả recognizer gốc của Paddle, để so sánh.

---

## Troubleshooting

| Lỗi | Cách xử lý |
|---|---|
| `Output already exists` | xoá file trong `output/` hoặc truyền `--overwrite` |
| `CUDA was requested ...` | bỏ chọn GPU cụ thể, để script tự chọn GPU rảnh |
| Container chết giữa chừng | chạy lại cùng lệnh — checkpoint theo `frame_id`, tự resume |
