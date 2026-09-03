"""
pipeline_g.py — Pipeline G: Upgraded Multi-Stage Keyframe Extraction Pipeline.

Flow:
  ALL FRAMES (shot/video)
    → Stage 1a: DAKE Coarse Candidate Selection (high candidate_ratio)
    → Stage 1b: Transition Peak Detection (Prominence Window + NMS distance)
    → Stage 1c: Blank/Black/White Transition Cut Hard-Veto
    → Stage 2:  Visual Encoder (BEiT-3 Large) — shot.start_frame luôn được encode
    → Stage 3:  Global Semantic Filter (min_frame_distance, bounded ring-buffer history,
                gap-based adaptive threshold + coverage-ceiling force-pick)
    → Stage 4:  Diversity / Redundancy Filter (score-aware NMS, temporal min_frame_distance gap dedup)
    → Stage 5:  Local Sharpness Reselection (±window_radius, cùng cảnh)
    → Keyframes

Mục tiêu:
  - Bắt buộc khoảng cách tối thiểu min_frame_distance (mặc định 5 frame) giữa 2 keyframe liên tiếp.
  - Khắc phục tình trạng mất keyframe quan trọng bằng cách tăng candidate recall.
  - Loại bỏ triệt để các keyframe trùng lặp (duplicate) bằng Global Semantic + Pairwise Diversity NMS.
  - Bảo vệ các keyframe ở vị trí chuyển cảnh (Transition Aware) — prominence-based, không tạo cluster.
  - DAKE steepness scores được truyền vào DiversityFilter để ưu tiên frame sắc nét hơn.
  - Video-wide history dùng bounded ring buffer, tránh OOM trên video dài.
  - Loại frame blank (đen/trắng phẳng) trước khi encode; ép frame đầu mỗi shot luôn
    được encode làm candidate (vẫn phải qua Semantic/Diversity Filter như các
    candidate khác — không tự động thành keyframe); ép coverage-ceiling khi 1 shot
    tĩnh quá lâu không có keyframe; tráo ảnh sang frame nét hơn trong ±window_radius
    lân cận (không đổi frame_idx).
"""

from __future__ import annotations

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

_beit3_src = Path(__file__).resolve().parents[2] / "unilm" / "beit3"
if str(_beit3_src) not in sys.path:
    sys.path.insert(0, str(_beit3_src))

from src.core.interfaces import BaseKeyframeExtractor
from src.core.models import ShotRecord, ShotKeyframes, Keyframe
from src.components.frame_loader import VideoFrameLoader
from src.components.dake import DAKESelector
from src.components.beit3_encoder import BEiT3Encoder
from src.components.semantic_filter import SemanticFilter
from src.components.diversity_filter import DiversityRedundancyFilter
from src.components.transition_selector import TransitionAwareSelector
from src.components.blank_veto import BlankTransitionVetoFilter
from src.components.sharpness_selector import SharpnessReselector


# Shot dài nhất của batch1 là 29279 frame; đọc nguyên shot ở 720p là 79 GB RAM.
# Cắt thành cửa sổ để trần RAM phụ thuộc hằng số này thay vì độ dài shot.
_MAX_WINDOW_FRAMES = 1200


class PipelineG(BaseKeyframeExtractor):
    """
    Pipeline G: Upgraded Multi-Stage Pipeline.

    Args:
        checkpoint_path         : Path đến beit3_large_patch16_224.pth.
        spm_path                : Path đến beit3.spm.
        candidate_ratio         : Tỉ lệ frame DAKE chọn (mặc định 0.05 để tăng recall).
        window_size             : Sliding window size của DAKE.
        min_frame_distance      : Khoảng cách tối thiểu (frame count) giữa 2 keyframe liên tiếp (default 5).
        similarity_threshold    : Ngưỡng cosine similarity cho Global Semantic Filter.
        redundancy_threshold    : Ngưỡng cosine similarity cho Diversity Filter.
        peak_prominence_window  : Bán kính cửa sổ prominence cho Transition peak detection.
        peak_percentile         : Ngưỡng percentile để một frame được coi là steepness peak.
        min_keyframes_per_shot  : Sàn mềm của Diversity Filter (chỉ có tác dụng khi >= 2).
        enable_transition       : Bật bảo vệ transition frames.
        max_history_size        : Giới hạn ring buffer lịch sử embedding cho Semantic Filter.
        gap_decay_start_frames  : Bắt đầu nới similarity threshold sau N frame không có keyframe.
        max_gap_frames          : Ép chọn keyframe nếu khoảng cách vượt N frame (coverage ceiling).
        enable_veto             : Bật blank/black/white transition cut hard-veto.
        veto_min_brightness     : Dưới ngưỡng này (mean pixel) coi là tối.
        veto_max_brightness     : Trên ngưỡng này (mean pixel) coi là sáng/trắng.
        veto_min_variance       : Dưới ngưỡng này (Laplacian variance) coi là phẳng.
        enable_sharpness        : Bật ±N frame local sharpness reselection.
        sharpness_window_radius : Bán kính cửa sổ frame lân cận để xét reselection.
        device                  : "cuda" hoặc "cpu".
        batch_size              : Batch size cho BEiT-3 encoder.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        spm_path: str | Path,
        candidate_ratio: float = 0.05,
        window_size: int = 3,
        min_frame_distance: int = 5,
        similarity_threshold: float = 0.90,
        redundancy_threshold: float = 0.88,
        peak_prominence_window: int = 3,
        peak_percentile: float = 90.0,
        min_keyframes_per_shot: int = 1,
        enable_transition: bool = True,
        max_history_size: int = 512,
        gap_decay_start_frames: int = 150,
        max_gap_frames: int = 300,
        enable_veto: bool = True,
        veto_min_brightness: float = 15.0,
        veto_max_brightness: float = 240.0,
        veto_min_variance: float = 15.0,
        enable_sharpness: bool = True,
        sharpness_window_radius: int = 4,
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.spm_path = Path(spm_path)
        self.candidate_ratio = candidate_ratio
        self.window_size = window_size
        self.min_frame_distance = min_frame_distance
        self.similarity_threshold = similarity_threshold
        self.redundancy_threshold = redundancy_threshold
        self.peak_prominence_window = peak_prominence_window
        self.enable_transition = enable_transition
        self.max_history_size = max_history_size
        self.max_gap_frames = max_gap_frames
        self.device = device
        self.batch_size = batch_size

        self._dake = DAKESelector(candidate_ratio=candidate_ratio, window_size=window_size)
        self._encoder: BEiT3Encoder | None = None
        self._semantic_filter = SemanticFilter(
            similarity_threshold=similarity_threshold,
            min_frame_distance=min_frame_distance,
            global_check=True,
            max_history_size=max_history_size,
            gap_decay_start_frames=gap_decay_start_frames,
            max_gap_frames=max_gap_frames,
        )
        self._diversity_filter = DiversityRedundancyFilter(
            redundancy_threshold=redundancy_threshold,
            min_temporal_gap_frames=min_frame_distance,
            min_keyframes_per_shot=min_keyframes_per_shot,
        )
        self._transition_selector = TransitionAwareSelector(
            peak_percentile=peak_percentile,
            prominence_window=peak_prominence_window,
            min_peak_distance=min_frame_distance,
        )
        self._veto = (
            BlankTransitionVetoFilter(
                min_brightness=veto_min_brightness,
                max_brightness=veto_max_brightness,
                min_variance=veto_min_variance,
            )
            if enable_veto
            else None
        )
        self._sharpness = (
            SharpnessReselector(window_radius=sharpness_window_radius)
            if enable_sharpness
            else None
        )

    @property
    def name(self) -> str:
        return "pipeline_g"

    def setup(self) -> None:
        self._encoder = BEiT3Encoder(
            checkpoint_path=self.checkpoint_path,
            spm_path=self.spm_path,
            device=self.device,
            batch_size=self.batch_size,
        )
        self._encoder.load()

    def teardown(self) -> None:
        if self._encoder:
            self._encoder.unload()
            self._encoder = None

    def extract(
        self,
        video_path: Path,
        shots: List[ShotRecord],
    ) -> List[ShotKeyframes]:
        results: List[ShotKeyframes] = []
        video_history: List[np.ndarray] = []
        last_keyframe_idx: Optional[int] = None

        with VideoFrameLoader(video_path) as loader:
            fps = loader.fps

            for shot in shots:
                shot_indices: List[int] = []
                shot_embeddings: List[np.ndarray] = []
                shot_jpegs: List[bytes] = []
                # Fallback nếu Semantic/Diversity Filter lọc sạch cả shot: đảm bảo
                # mỗi shot luôn có >=1 keyframe (mặc định là frame đầu shot).
                fallback: Optional[Tuple[int, np.ndarray, bytes]] = None

                for w_start in range(shot.start_frame, shot.end_frame + 1, _MAX_WINDOW_FRAMES):
                    w_end = min(w_start + _MAX_WINDOW_FRAMES - 1, shot.end_frame)
                    all_frames = loader.read_range(w_start, w_end)
                    if not all_frames:
                        continue

                    frame_idx_list = [idx for idx, _ in all_frames]

                    # Stage 1a: DAKE Candidate Selection — top-k by steepness, kèm luôn
                    # steepness/aggregated cho stage 1b và stage 4 (tránh encode JPEG lần hai)
                    candidate_indices, steepness, aggregated_scores = self._dake.select_with_scores(all_frames)
                    candidate_map = {idx: img for idx, img in all_frames}
                    steepness_map = {
                        idx: score for idx, score in zip(frame_idx_list, aggregated_scores)
                    }

                    # Stage 1b: Transition Peak Detection (Prominence Window + min_frame_distance NMS)
                    if self.enable_transition:
                        trans_peaks = self._transition_selector.find_transition_peaks(
                            frame_idx_list, steepness
                        )
                        combined_indices = sorted(set(candidate_indices) | trans_peaks)
                    else:
                        combined_indices = candidate_indices

                    # DAKE chọn top-k theo steepness GLOBAL trong window, không đảm bảo
                    # trải đều theo thời gian — cảnh tĩnh kéo dài có thể để lại khoảng
                    # trống hoàn toàn không có candidate nào, khiến force-pick coverage
                    # ceiling ở Stage 3 không có gì để chọn (không có candidate thì
                    # không force-pick được). Chèn thêm "anchor" mỗi max_gap_frames vào
                    # chỗ trống để đảm bảo luôn có candidate cho Stage 3 xét tới.
                    combined_indices = _fill_candidate_gaps(
                        combined_indices, frame_idx_list, w_start, self.max_gap_frames
                    )

                    candidate_frames = [
                        (idx, candidate_map[idx])
                        for idx in combined_indices
                        if idx in candidate_map
                    ]

                    # Stage 1c: Blank/Black/White Transition Cut Hard-Veto
                    if self._veto is not None:
                        candidate_frames = self._veto.filter(candidate_frames)

                    if not candidate_frames:
                        candidate_frames = [all_frames[0]]

                    # Ép shot.start_frame luôn được encode (bất kể veto/DAKE/transition),
                    # chỉ ở window đầu tiên của shot — không đổi logic global dedup cho
                    # các candidate khác.
                    is_first_window = w_start == shot.start_frame
                    if (
                        is_first_window
                        and shot.start_frame in candidate_map
                        and not any(idx == shot.start_frame for idx, _ in candidate_frames)
                    ):
                        candidate_frames = [
                            (shot.start_frame, candidate_map[shot.start_frame])
                        ] + candidate_frames

                    # Stage 2: Visual Encoder (BEiT-3 Large)
                    enc_indices, embeddings = self._encoder.encode_batch(candidate_frames)
                    if len(enc_indices) == 0:
                        continue

                    if is_first_window and fallback is None and shot.start_frame in enc_indices:
                        pos = enc_indices.index(shot.start_frame)
                        fallback = (
                            shot.start_frame,
                            embeddings[pos],
                            candidate_map[shot.start_frame],
                        )

                    # Stage 3: Global Semantic Filter (with min_frame_distance & video-wide history)
                    sem_indices, sem_embeddings, forced_set = self._semantic_filter.filter(
                        enc_indices,
                        embeddings,
                        history_embeddings=video_history,
                        last_keyframe_idx=last_keyframe_idx,
                    )

                    # Frame force-pick bởi coverage ceiling (forced_set) đi thẳng ra output,
                    # KHÔNG qua Diversity Filter: các frame này vốn rất giống nhau (cùng 1
                    # cảnh tĩnh) — nếu để redundancy_threshold=0.88 của Diversity Filter xét,
                    # nó sẽ coi chúng là trùng lặp và xoá gần hết, phá vỡ coverage ceiling
                    # vừa đảm bảo ở Stage 3.
                    forced_pairs = [
                        (idx, emb) for idx, emb in zip(sem_indices, sem_embeddings) if idx in forced_set
                    ]
                    normal_pairs = [
                        (idx, emb) for idx, emb in zip(sem_indices, sem_embeddings) if idx not in forced_set
                    ]
                    normal_indices = [idx for idx, _ in normal_pairs]
                    normal_embeddings = (
                        np.stack([emb for _, emb in normal_pairs])
                        if normal_pairs
                        else np.empty((0, embeddings.shape[1]), dtype=embeddings.dtype)
                    )
                    normal_scores = [steepness_map.get(idx, 0.0) for idx in normal_indices]

                    # Stage 4: Pairwise Matrix Diversity & Redundancy NMS Filter
                    div_indices, div_embeddings = self._diversity_filter.filter(
                        normal_indices, normal_embeddings, scores=normal_scores
                    )

                    merged = sorted(
                        forced_pairs + list(zip(div_indices, div_embeddings)),
                        key=lambda p: p[0],
                    )

                    if merged:
                        last_keyframe_idx = merged[-1][0]

                    final_indices = [idx for idx, _ in merged]
                    final_embeddings = [emb for _, emb in merged]

                    if not final_indices:
                        continue

                    # Stage 5: Local Sharpness Reselection (±window_radius, cùng cảnh)
                    if self._sharpness is not None:
                        images = self._sharpness.reselect(
                            final_indices, candidate_map, shot.start_frame, shot.end_frame
                        )
                    else:
                        images = [candidate_map[idx] for idx in final_indices]

                    shot_indices.extend(final_indices)
                    shot_embeddings.extend(final_embeddings)
                    # Frame còn trong RAM ngay lúc này; mã hoá luôn để Runner khỏi
                    # phải tua lại video cho từng keyframe.
                    shot_jpegs.extend(
                        cv2.imencode(".jpg", img)[1].tobytes() for img in images
                    )

                    for _, emb in merged:
                        video_history.append(emb)
                    if len(video_history) > self.max_history_size:
                        video_history = video_history[-self.max_history_size :]

                if not shot_indices and fallback is not None:
                    idx, emb, img = fallback
                    shot_indices = [idx]
                    shot_embeddings = [emb]
                    shot_jpegs = [cv2.imencode(".jpg", img)[1].tobytes()]
                    video_history.append(emb)
                    if len(video_history) > self.max_history_size:
                        video_history = video_history[-self.max_history_size :]
                    last_keyframe_idx = idx

                keyframes = _make_keyframes(shot_indices, shot_embeddings, shot_jpegs, fps, shot.shot_id)
                results.append(ShotKeyframes(shot.video_id, shot.shot_id, keyframes))

        return results


def _fill_candidate_gaps(
    combined_indices: List[int],
    frame_idx_list: List[int],
    w_start: int,
    max_gap: int,
) -> List[int]:
    """
    Đảm bảo không có khoảng trống > max_gap frame nào hoàn toàn thiếu candidate
    trong combined_indices, bằng cách chèn thêm frame từ frame_idx_list (đã có
    sẵn trong RAM, không cần đọc thêm) làm "anchor" candidate mỗi max_gap frame.
    """
    if not frame_idx_list:
        return combined_indices

    combined_set = set(combined_indices)
    filled: List[int] = []
    prev = w_start - 1
    for idx in frame_idx_list:
        if idx in combined_set:
            filled.append(idx)
            prev = idx
        elif idx - prev >= max_gap:
            filled.append(idx)
            prev = idx

    return sorted(filled)


def _make_keyframes(
    frame_indices: List[int],
    embeddings,
    jpegs: List[bytes],
    fps: float,
    shot_id: str,
) -> List[Keyframe]:
    keyframes = []
    for i, (idx, emb, jpg) in enumerate(zip(frame_indices, embeddings, jpegs)):
        ts_ms = int(idx / max(fps, 1.0) * 1000)
        kf = Keyframe(
            keyframe_id=f"{shot_id}_kf{i + 1:04d}",
            frame_idx=idx,
            timestamp_ms=ts_ms,
            image_path="",
            embedding=emb,
            image_jpeg=jpg,
        )
        keyframes.append(kf)
    return keyframes
