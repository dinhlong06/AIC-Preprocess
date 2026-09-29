"""
interfaces.py — Abstract Base Class cho tất cả pipeline Keyframe Extraction.

Thiết kế theo Interface Pattern:
  - Mọi pipeline (A, B, C, D) đều phải kế thừa BaseKeyframeExtractor.
  - Runner và CLI chỉ giao tiếp qua interface này, không biết chi tiết pipeline.
  - Thêm pipeline mới chỉ cần tạo subclass, không sửa code cũ.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from src.core.models import ShotRecord, ShotKeyframes


class BaseKeyframeExtractor(ABC):
    """
    Interface chung cho tất cả Keyframe Extraction pipeline.

    Mỗi subclass phải implement:
      - name    : tên pipeline (dùng làm tên thư mục output)
      - extract : logic trích xuất keyframe cho 1 video
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Trả về tên định danh của pipeline.
        Ví dụ: 'pipeline_g', 'pipeline_h'.
        Tên này sẽ được dùng làm tên thư mục trong benchmark/.
        """
        pass

    @abstractmethod
    def extract(
        self,
        video_path: Path,
        shots: List[ShotRecord],
    ) -> List[ShotKeyframes]:
        """
        Trích xuất keyframe từ video dựa trên danh sách shots.

        Args:
            video_path : Đường dẫn đến file video (.mp4).
            shots      : Danh sách ShotRecord (đọc từ shot.jsonl).

        Returns:
            Danh sách ShotKeyframes — mỗi phần tử chứa keyframes của một shot.
        """
        pass

    def setup(self) -> None:
        """
        (Tùy chọn) Load model, khởi tạo tài nguyên trước khi xử lý.
        Được gọi một lần trước vòng lặp video.
        Mặc định không làm gì.
        """
        pass

    def teardown(self) -> None:
        """
        (Tùy chọn) Giải phóng tài nguyên (VRAM, file handles) sau khi xử lý xong.
        Được gọi một lần sau khi hoàn thành tất cả video.
        Mặc định không làm gì.
        """
        pass
