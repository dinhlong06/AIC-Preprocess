"""
models.py — Data models (dataclasses) dùng chung toàn bộ module.

Các class ở đây chỉ chứa dữ liệu, không chứa business logic.
Được dùng để truyền dữ liệu giữa các thành phần (FrameLoader → DAKE → Encoder → Filter → Exporter).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ShotRecord:
    """
    Thông tin một shot đọc từ shot.jsonl (output của Shot Detector).

    Fields:
        video_id    : ID của video (ví dụ "V001")
        shot_id     : ID của shot (ví dụ "S0003")
        start_frame : Frame bắt đầu (inclusive)
        end_frame   : Frame kết thúc (inclusive)
        fps         : Frame-per-second của video gốc
    """
    video_id: str
    shot_id: str
    start_frame: int
    end_frame: int
    fps: float = 25.0

    @property
    def duration_sec(self) -> float:
        """Thời lượng của shot tính bằng giây."""
        return (self.end_frame - self.start_frame) / max(self.fps, 1.0)

    @property
    def total_frames(self) -> int:
        """Tổng số frame trong shot."""
        return self.end_frame - self.start_frame + 1


@dataclass
class Keyframe:
    """
    Thông tin của một keyframe đã được chọn.

    Fields:
        keyframe_id  : ID duy nhất, ví dụ "kf0001"
        frame_idx    : Chỉ số frame trong video gốc
        timestamp_ms : Thời điểm tương ứng (millisecond)
        image_path   : Đường dẫn đến ảnh .jpg đã lưu
        embedding    : Vector embedding (đã tính sẵn lúc Semantic Filter), L2-normalized.
                        Không ghi vào keyframes.jsonl — chỉ dùng để lưu .npy riêng.
        image_jpeg   : Ảnh đã mã hoá JPEG, giữ lại từ lúc trích xuất khi frame còn trong
                        RAM. Có giá trị này thì Runner ghi thẳng ra đĩa, khỏi tua lại
                        video (mỗi lần tua ~118 ms vì phải nhảy về I-frame).
    """
    keyframe_id: str
    frame_idx: int
    timestamp_ms: int
    image_path: str
    embedding: Optional[np.ndarray] = None
    image_jpeg: Optional[bytes] = None

    def to_dict(self) -> dict:
        """Chuyển sang dict để ghi vào keyframes.jsonl."""
        return {
            "keyframe_id": self.keyframe_id,
            "frame_idx": self.frame_idx,
            "timestamp_ms": self.timestamp_ms,
            "image_path": self.image_path,
        }


@dataclass
class ShotKeyframes:
    """
    Kết quả keyframe extraction cho một shot cụ thể.

    Fields:
        video_id  : ID video
        shot_id   : ID shot
        keyframes : Danh sách Keyframe đã chọn
    """
    video_id: str
    shot_id: str
    keyframes: List[Keyframe] = field(default_factory=list)

    def to_records(self) -> List[dict]:
        """
        Chuyển thành list dict để ghi từng dòng vào keyframes.jsonl.
        Mỗi dict tương ứng một dòng JSON trong file output.
        """
        records = []
        for kf in self.keyframes:
            record = {
                "video_id": self.video_id,
                "shot_id": self.shot_id,
            }
            record.update(kf.to_dict())
            records.append(record)
        return records


@dataclass
class PipelineStatistics:
    """
    Thống kê hiệu năng của một pipeline trên một video.
    Được dùng để tạo benchmark_summary.csv.

    Fields:
        pipeline            : Tên pipeline (ví dụ "pipeline_c")
        video_id            : ID video
        num_shots           : Số shot xử lý
        total_keyframes     : Tổng keyframe sinh ra
        avg_keyframes_per_shot : Trung bình KF/shot
        processing_time_sec : Thời gian xử lý toàn bộ video (giây)
        processing_fps      : Tốc độ xử lý (frame/giây)
        peak_vram_gb        : VRAM peak (GB), 0.0 nếu CPU
        total_storage_mb    : Dung lượng ảnh keyframe (MB)
        diversity_score     : 1 - avg cosine similarity giữa các KF (cao = đa dạng)
        coverage_score      : % thời gian shot được cover bởi keyframe
    """
    pipeline: str
    video_id: str
    num_shots: int
    total_keyframes: int
    avg_keyframes_per_shot: float
    processing_time_sec: float
    processing_fps: float
    peak_vram_gb: float
    total_storage_mb: float
    diversity_score: float = 0.0
    coverage_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    def to_dict(self) -> dict:
        """Chuyển sang dict để ghi vào CSV."""
        return {
            "Pipeline": self.pipeline,
            "Video": self.video_id,
            "#KF": self.total_keyframes,
            "KF/Shot": round(self.avg_keyframes_per_shot, 2),
            "Time(s)": round(self.processing_time_sec, 2),
            "FPS": round(self.processing_fps, 1),
            "VRAM(GB)": round(self.peak_vram_gb, 2),
            "Storage(MB)": round(self.total_storage_mb, 2),
            "Diversity": round(self.diversity_score, 3),
            "Coverage": round(self.coverage_score, 3),
            "Precision": round(self.precision, 3),
            "Recall": round(self.recall, 3),
            "F1": round(self.f1_score, 3),
        }
