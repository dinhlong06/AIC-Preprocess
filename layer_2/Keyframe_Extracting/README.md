# Keyframe Extractor — AI Challenge 2026

Module trích xuất Keyframe từ video dựa trên kết quả Shot Detector.
Hỗ trợ **2 pipeline**: `pipeline_g` (mặc định, đang chạy production) và
`pipeline_h` (G + nhận biết vùng chữ, ứng viên A/B).

---

## Tổng quan hệ thống

```
Shot Detector
     ↓
shot.jsonl / shots.json
     ↓
Keyframe Extraction   ← module này
     ↓
keyframes.jsonl + ảnh .jpg
     ↓
Embedding → Milvus
```

---

## 2 Pipeline

| Pipeline | DAKE | Encoder | Mục tiêu |
|----------|------|---------|-----------|
| **G** | ✅ | BEiT-3 Large (1024-dim) | **Mặc định** — 7 giai đoạn, tham số hiệu chỉnh v3.1 |
| **H** | ✅ | BEiT-3 Large | G + `text_prescan`: ưu tiên khung hình có thay đổi vùng chữ |

Cả hai dùng chung bộ tham số G; H bổ sung khối `text_prescan` trong config.

### DAKE là gì?

DAKE là **Coarse Temporal Filter** — không phải semantic model.  
Hoạt động dựa trên kích thước JPEG file (không dùng CNN, không dùng GPU).  
Nhiệm vụ: giảm số frame cần encode (giữ `candidate_ratio × total_frames`).

---

## Cấu trúc thư mục

```
Keyframe_Extractor/
├── checkpoint/
│   └── beit-3/
│       ├── beit3_large_patch16_224.pth   ← BEiT-3 weights
│       └── beit3.spm                      ← SentencePiece tokenizer
│
├── configs/
│   ├── pipeline_g.yaml    ← config Pipeline G (mặc định)
│   └── pipeline_h.yaml    ← config Pipeline H (G + text_prescan)
│
├── src/
│   ├── core/
│   │   ├── models.py      ← Dataclasses: ShotRecord, Keyframe, ShotKeyframes, PipelineStatistics
│   │   ├── interfaces.py  ← BaseKeyframeExtractor ABC
│   │   ├── runner.py      ← KeyframeBenchmarkRunner (điều phối toàn bộ)
│   │   └── metrics.py     ← Diversity, Coverage, Precision/Recall vs GT
│   │
│   ├── components/
│   │   ├── frame_loader.py      ← Đọc frame từ video (Mode 1) hoặc ảnh dir (Mode 2)
│   │   ├── dake.py              ← DAKE: JPEG steepness → sliding window → top-k
│   │   ├── beit3_encoder.py     ← BEiT-3 Large visual encoder (1024-dim CLS)
│   │   ├── mobilenet_encoder.py ← MobileNetV3-Large, chỉ dùng cho đánh giá vs GT
│   │   ├── semantic_filter.py   ← Cosine similarity sequential filter
│   │   ├── text_prescan.py      ← Nhận biết vùng chữ thay đổi (chỉ H)
│   │   ├── diversity_filter.py  ← Giãn cách keyframe trong shot
│   │   ├── transition_selector.py, blank_veto.py, sharpness_selector.py
│   │
│   └── extractors/
│       ├── pipeline_g.py  ← 7 giai đoạn, mặc định
│       └── pipeline_h.py  ← G + text_prescan
│
├── unilm/                 ← BEiT-3 repo (đã clone từ microsoft/unilm)
│   └── beit3/
│
├── dataset/
│   ├── raw_video/         ← Video .mp4 đầu vào
│   └── shots/             ← shots.json từ Shot Detector (1 folder/video)
│
├── benchmark*/            ← Output tự động sinh khi chạy (git-ignored)
│   ├── pipeline_g/
│   ├── pipeline_h/
│   └── benchmark_summary.csv
│
├── cli.py                 ← Entry point chính
└── requirements.txt
```

## Input & Output của hệ thống (Giao tiếp giữa các Module)

### 1. INPUT (Đầu vào)
*   **Video gốc (`.mp4`)**: Đường dẫn đến file video cần trích xuất.
*   **`shots.json`**: File chứa danh sách các phân cảnh (shot) sinh ra từ module Shot Detector. Mỗi shot bao gồm:
    *   `video_id`: Tên video.
    *   `shot_id`: Mã phân cảnh (vd: `S0001`).
    *   `start_frame` & `end_frame`: Khung hình bắt đầu và kết thúc.
    *   *Nếu không có file này, hệ thống sẽ tự động băm video thành các chunk nhỏ (300 frames/chunk) để tránh tràn RAM.*

### 2. OUTPUT (Đầu ra)
Hệ thống sẽ tạo ra một thư mục cho mỗi video chứa:
*   **Các ảnh Keyframe (`.jpg`)**: Được lưu theo cấu trúc `<video_id>/<shot_id>/kf0001.jpg`. Các module phía sau (như SigCLIP) sẽ đọc các ảnh này để đẩy vào Vector Database.
*   **File `keyframes.jsonl`**: Chứa siêu dữ liệu (metadata) của toàn bộ keyframe trong video. Mỗi dòng là 1 JSON Object:
    ```json
    {"video_id": "L30_V001", "shot_id": "S0001", "keyframe_id": "S0001_kf0001", "frame_idx": 42, "timestamp_ms": 1680, "image_path": "S0001/kf0001.jpg"}
    ```
    *Ý nghĩa các trường:*
    *   `keyframe_id`: ID định danh duy nhất.
    *   `frame_idx`: Vị trí khung hình thực tế trong video gốc.
    *   `timestamp_ms`: Thời gian xuất hiện tính bằng mili-giây (Dùng cho Video Player).
    *   `image_path`: Đường dẫn tương đối tới ảnh `.jpg` đã cắt.

---

## Cách chạy

### 1. Chuẩn bị dataset

```
dataset/
├── raw_video/
│   ├── video1.mp4
│   └── video2.mp4
└── shots/
    ├── video1/
    │   └── shots.json    ← từ Shot Detector
    └── video2/
        └── shots.json
```

> **Lưu ý**: Nếu không có `shots.json`, module sẽ tự động coi toàn bộ video là 1 shot.

### 2. Chạy Pipeline G (mặc định)

```bash
cd Keyframe_Extractor
python cli.py --pipeline pipeline_g --video_dir dataset/raw_video
```

### 3. Chạy cả G và H

```bash
python cli.py --pipeline all --video_dir dataset/raw_video
```

### 4. Tùy chỉnh tham số qua CLI

```bash
python cli.py \
  --pipeline pipeline_g \
  --video_dir dataset/raw_video \
  --shots_dir dataset/shots \
  --output_dir benchmark \
  --threshold 0.85 \
  --candidate_ratio 0.05 \
  --device cuda \
  --batch_size 16
```

### 5. Dùng file config YAML

```bash
python cli.py --config configs/pipeline_h.yaml --video_dir dataset/raw_video
```

---

## Output

### Cấu trúc thư mục output

```
benchmark/
├── pipeline_g/
│   ├── video1/
│   │   ├── keyframes.jsonl     ← metadata từng keyframe (1 dòng/KF)
│   │   ├── statistics.json     ← thống kê video này
│   │   ├── S0001/
│   │   │   ├── kf0001.jpg
│   │   │   └── kf0002.jpg
│   │   └── S0002/
│   │       └── kf0001.jpg
│   └── video2/
│       └── ...
└── benchmark_summary.csv       ← tổng hợp tất cả pipeline × video
```

### keyframes.jsonl (mỗi dòng = 1 keyframe)

```json
{"video_id": "video1", "shot_id": "S0001", "keyframe_id": "S0001_kf0001", "frame_idx": 42, "timestamp_ms": 1680, "image_path": "S0001/kf0001.jpg"}
{"video_id": "video1", "shot_id": "S0001", "keyframe_id": "S0001_kf0002", "frame_idx": 120, "timestamp_ms": 4800, "image_path": "S0001/kf0002.jpg"}
```

### benchmark_summary.csv
Ví dụ:
| Pipeline | Video | #KF | KF/Shot | Time(s) | FPS | VRAM(GB) | Storage(MB) | Diversity | Coverage |
|----------|-------|-----|---------|---------|-----|----------|-------------|-----------|----------|
| pipeline_g | video1 | 820 | 2.5 | 145 | 62 | 4.2 | 312 | 0.0 | 0.91 |
| pipeline_h | video1 | 650 | 2.0 | 52 | 170 | 2.1 | 248 | 0.0 | 0.94 |



---

## Benchmark Metrics

### 1. Keyframe Quality (vs Ground Truth)

| Metric | Mô tả | Cần GT? |
|--------|-------|---------|
| Recall | % GT keyframe được tìm thấy | ✅ |
| Precision | % predicted keyframe đúng | ✅ |
| F1-score | Harmonic mean | ✅ |
| Diversity Score | 1 - avg cosine sim giữa các KF (cao = tốt) | ❌ |
| Coverage Score | % thời gian shot được đại diện | ❌ |

> GT matching dùng Nearest Neighbor (cosine similarity ≥ 0.80).

### 2. Computational Efficiency

| Metric | Mô tả |
|--------|-------|
| Processing Time (s) | Thời gian xử lý toàn video |
| Processing FPS | Frame/giây |
| Peak VRAM (GB) | GPU memory peak |

### 3. Storage Cost

| Metric | Mô tả |
|--------|-------|
| #Keyframes | Tổng số keyframe |
| KF/Shot | Trung bình keyframe/shot |
| Storage (MB) | Dung lượng ảnh .jpg |

### 4. Downstream Retrieval Impact

Đánh giá ở giai đoạn sau khi tích hợp Embedding + Milvus:
- mAP@K, R@1/R@5/R@10 trên tập query thử nghiệm.
---

## Cấu hình DAKE (candidate_ratio benchmark)

Thử các giá trị sau để tìm điểm cân bằng tốt nhất:

| candidate_ratio | Ý nghĩa | Tốc độ |
|----------------|---------|--------|
| 0.01 | Giữ 1% frame | Nhanh nhất |
| 0.02 | Giữ 2% frame | **Mặc định** |
| 0.05 | Giữ 5% frame | Trung bình |
| 0.10 | Giữ 10% frame | Chậm hơn |
| 0.20 | Giữ 20% frame | Chất lượng cao nhất |

Đây là một số gợi ý thôi có thể chạy nhiều kiểu.
---
