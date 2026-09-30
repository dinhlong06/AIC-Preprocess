"""
semantic_filter.py — Lọc Keyframe dựa trên ngữ nghĩa (Semantic Filtering).

Nhận đầu vào là embedding của Candidate Frames (từ DAKE hoặc toàn bộ frame).
Loại bỏ các frame quá giống nhau dựa trên cosine similarity.
Trả về list frame_idx là Keyframes cuối cùng.

Thuật toán:
  1. Sắp xếp candidate frames theo thứ tự thời gian (frame_idx tăng dần).
  2. Kiểm tra khoảng cách tối thiểu min_frame_distance so với keyframe đã chọn gần nhất.
     Nếu frame_idx mới cách keyframe trước < min_frame_distance => Bỏ qua ngay lập tức.
  3. Frame tiếp theo được giữ lại nếu cosine similarity với keyframe
     đã chọn gần nhất < threshold (đủ khác biệt).

Upgrade v2 + min_frame_distance:
  - min_frame_distance: Chỉ xem xét frame nếu nó cách keyframe trước đó ít nhất min_frame_distance frames.
  - Bounded video-wide history via ring buffer (max_history_size).
  - Batched matrix similarity check.
  - Adaptive threshold relaxation.

Upgrade v3 — Coverage Ceiling:
  - Adaptive threshold decay dựa trên khoảng cách frame_idx thực tế kể từ keyframe
    trước (gap_decay_start_frames), thay vì số candidate liên tiếp bị lọc.
  - Force-pick: nếu gap vượt max_gap_frames, ép chọn candidate khác biệt nhất
    (max_sim thấp nhất) trong số đã bị lọc kể từ keyframe trước, đảm bảo không
    có khoảng trống quá dài (shot tĩnh) hoàn toàn thiếu keyframe.

Tham khảo: KEYFRAME_EXTRACTING.md, Section 8.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import List, Tuple, Optional


class SemanticFilter:
    """
    Lọc frame dựa trên ngữ nghĩa: chỉ giữ lại frame đủ khác biệt.

    Args:
        similarity_threshold     : Cosine similarity tối đa để giữ frame.
            Frame mới được chọn nếu sim < threshold so với KF trước (hoặc tất cả KF đã chọn).
            Mặc định 0.90 theo spec KEYFRAME_EXTRACTING.md.
        min_frame_distance       : Khoảng cách frame tối thiểu so với keyframe kế trước.
                                   Chỉ xem xét frame nếu cách keyframe trước ít nhất N frames. Mặc định 5.
        max_history_size         : Số lượng embedding tối đa trong ring buffer lịch sử video.
                                   Ngăn memory growth O(N) trong video dài. Mặc định 512.
        gap_decay_start_frames    : Nếu khoảng cách frame_idx kể từ keyframe trước vượt
                                   ngưỡng này, nới ngưỡng threshold tạm thời. Mặc định 150.
        max_gap_frames            : Nếu khoảng cách frame_idx kể từ keyframe trước vượt
                                   ngưỡng này, ép chọn (force-pick) candidate khác biệt
                                   nhất trong số đã bị lọc, đảm bảo không có khoảng trống
                                   quá dài thiếu keyframe. Mặc định 300.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.90,
        min_frame_distance: int = 5,
        max_history_size: int = 512,
        gap_decay_start_frames: int = 150,
        max_gap_frames: int = 300,
    ):
        self.threshold = similarity_threshold
        self.min_frame_distance = max(1, int(min_frame_distance))
        self.max_history_size = max_history_size
        self.gap_decay_start_frames = gap_decay_start_frames
        self.max_gap_frames = max_gap_frames

    def filter(
        self,
        frame_indices: List[int],
        embeddings: np.ndarray,
        history_embeddings: Optional[List[np.ndarray]] = None,
        last_keyframe_idx: Optional[int] = None,
        protected: Optional[set] = None,
    ) -> Tuple[List[int], np.ndarray, set]:
        """
        Duyệt tuần tự và loại bỏ frame quá giống với keyframe trước / keyframe đã chọn.

        Args:
            frame_indices       : Danh sách frame_idx (tuyệt đối), đã sort tăng dần.
            embeddings          : numpy array shape (N, D), L2-normalized.
            history_embeddings  : Danh sách embedding của các keyframe đã chọn từ các shot trước đó trong video.
            last_keyframe_idx   : Frame_idx của keyframe được chọn gần đây nhất (từ shot trước).
            protected           : Frame_idx được miễn similarity check (vẫn chịu
                                  min_frame_distance). Dùng cho frame mà vùng text đổi:
                                  whole-frame cosine của chúng ~0.98 nên similarity
                                  check luôn loại, dù nội dung chữ đã khác hẳn.

        Returns:
            Tuple (selected_indices, selected_embeddings, forced_indices).
            forced_indices: tập frame_idx được force-pick bởi coverage ceiling — các
            frame này về bản chất vẫn RẤT giống nhau (cùng cảnh tĩnh), nên caller
            KHÔNG được đưa qua DiversityRedundancyFilter (threshold 0.88 sẽ coi
            chúng là trùng lặp và xoá gần hết, phá vỡ coverage ceiling).
        """
        if len(frame_indices) == 0:
            return [], embeddings, set()

        selected_indices: List[int] = []
        selected_embeddings: List[np.ndarray] = []
        forced_indices: set = set()

        # Build bounded ring buffer from provided history
        ring: deque[np.ndarray] = deque(maxlen=self.max_history_size)
        if history_embeddings:
            for h in history_embeddings[-self.max_history_size :]:
                ring.append(h)

        last_selected = last_keyframe_idx
        # Candidate bị lọc kể từ last_selected: (frame_idx, embedding, max_sim).
        # Cần giữ lại để force-pick (coverage ceiling) biết ứng viên nào khác biệt nhất.
        pending: List[Tuple[int, np.ndarray, float]] = []

        for i in range(len(frame_indices)):
            curr_idx = frame_indices[i]

            # Coverage ceiling: gap kể từ keyframe trước đã vượt max_gap_frames
            # -> ép chọn ứng viên khác biệt nhất (max_sim thấp nhất) trong pending.
            if (
                last_selected is not None
                and pending
                and (curr_idx - last_selected) >= self.max_gap_frames
            ):
                # Ưu tiên max_sim thấp nhất (khác biệt nhất); hoà thì chọn frame
                # muộn nhất (đẩy last_selected xa hơn, giảm rủi ro gap kế tiếp
                # lại vượt ngưỡng ngay lập tức).
                pick_idx, pick_emb, _ = min(pending, key=lambda p: (p[2], -p[0]))
                selected_indices.append(pick_idx)
                selected_embeddings.append(pick_emb)
                forced_indices.add(pick_idx)
                ring.append(pick_emb)
                last_selected = pick_idx
                pending = []

            # Bước 1: Enforce min_frame_distance — chỉ xem xét nếu cách keyframe trước ít nhất min_frame_distance
            if last_selected is not None and (curr_idx - last_selected) < self.min_frame_distance:
                continue

            emb = embeddings[i]
            check_pool = list(ring) + selected_embeddings

            if protected and curr_idx in protected:
                max_sim = -1.0
                is_different = True
            elif check_pool:
                # Batched matmul: (K, D) @ (D,) → (K,)
                pool_mat = np.stack(check_pool)  # (K, D)
                sims = pool_mat @ emb            # (K,)
                max_sim = float(np.max(sims))

                # Adaptive threshold: nới nếu gap kể từ keyframe trước đã lớn
                effective_thresh = self.threshold
                gap = curr_idx - last_selected if last_selected is not None else 0
                if gap > self.gap_decay_start_frames:
                    relax = min(0.05, (gap - self.gap_decay_start_frames) * 0.005)
                    effective_thresh = min(0.99, self.threshold + relax)

                is_different = max_sim < effective_thresh
            else:
                max_sim = -1.0
                is_different = True

            if is_different:
                selected_indices.append(curr_idx)
                selected_embeddings.append(emb)
                ring.append(emb)
                last_selected = curr_idx
                pending = []
            else:
                pending.append((curr_idx, emb, max_sim))

        if not selected_embeddings:
            return [], np.empty((0, embeddings.shape[1]), dtype=np.float32), forced_indices

        return selected_indices, np.stack(selected_embeddings), forced_indices
