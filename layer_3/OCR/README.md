# OCR Module -- AIC2026

Trích text tiếng Việt từ keyframe bằng **PaddleOCR PP-OCRv6** trên GPU, tự
hiệu đính dấu ngay trong từng frame (`ocr/corrector.py`). Stage 3
(`run_vlm_correct.py`) sửa nốt những box bị gộp chữ sau khi merge 2 recognizer.

> Lưu ý: đường chạy **production hiện tại là `../OCR_gemma/gemma_ocr.py`** (Gemma 4
> qua API UIT/AI Studio, cho chất lượng tốt hơn Paddle trên ảnh bản tin). Module
> này giữ lại để sinh lại output Paddle/VietOCR cũ và làm đối chiếu.

---

## Project Structure

```
OCR/
├── run_paddle.py         # CLI entry point, stage 1
├── run_paddle.sh          # build+run stage 1 in Docker, picks the freest GPU
├── run_paddle_batch1.sh         # batch1, 1 GPU
├── run_paddle_batch1_shards.sh  # batch1, nhiều shard/GPU
├── run_vlm_correct.py    # CLI entry point, stage 3
├── run_vlm_correct.sh          # build+run stage 3 in Docker (image OCR_v2/Dockerfile.vllm)
├── run_vlm_correct_resilient.sh # stage 3, retry dài hạn
├── merge_recognizers.py  # CPU: gộp output 2 recognizer, re-decide box gộp chữ
├── config.yaml           # settings của stage 1 và stage 3
├── requirements.txt
├── Dockerfile             # GPU container, stage 1
├── ocr/
│   ├── __init__.py
│   ├── paddle_engine.py   # PaddleOCR det + VietOCR/Paddle rec
│   ├── corrector.py       # hiệu đính dấu tiếng Việt, chạy local CPU
│   ├── frame_skip.py      # blank/blur skip + CLAHE/news-band preprocessing
│   ├── loader.py          # frame folder scanner
│   ├── formatter.py       # JSON output + checkpoint I/O
│   ├── vlm_correct.py     # stage 3: sửa box gộp chữ bằng Qwen3-VL
│   └── pipeline.py        # orchestrator
└── output/                # kết quả chạy (git-ignored)
```

---

## Usage

### Stage 1 -- PaddleOCR (GPU, inside Docker)

```bash
./run_paddle.sh                       # build image, chọn GPU rảnh nhất, chạy
./run_paddle.sh --input /path/frames # chạy thư mục keyframe khác
NSHARDS=4 ./run_paddle_batch1_shards.sh shots   # chạy nhiều GPU
```

Output: `output/output_vietocr.json`, `output/output_paddle_origin.json` (tuỳ
`config.yaml`).

### Merge 2 recognizer (CPU, không cần GPU)

```bash
python merge_recognizers.py --config config.yaml
```

### Stage 3 -- sửa box gộp chữ (GPU, container vLLM riêng)

```bash
./run_vlm_correct.sh                  # dựng image từ ../OCR_v2/Dockerfile.vllm
./run_vlm_correct_resilient.sh        # chạy lâu, tự retry
```

Input là output của `merge_recognizers.py`, không phải output thô của Paddle.

---

## Output

`output/` chứa JSON, mỗi frame một record:

```json
{"frame_id": "L21_V001_000000_kf0001", "boxes": [{"text": "HTV7", "confidence": 0.93, "bbox": [..]}]}
```

`frame_id` khớp với `keyframe_id` của layer 2 để `layer_5` join.

---

## Troubleshooting

| Lỗi | Cách xử lý |
|---|---|
| `Output already exists` | xoá file trong `output/` hoặc dùng `--overwrite` nếu script có cờ đó |
| `CUDA was requested ...` | bỏ chọn GPU cụ thể, để script tự chọn GPU rảnh |
| Container stage 3 không build được | image đó dựng từ `../OCR_v2/Dockerfile.vllm`, cần `models/` của OCR_v2 |
