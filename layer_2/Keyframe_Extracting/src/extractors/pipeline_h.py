"""
pipeline_h.py — Pipeline H: Pipeline G + text-awareness.

Giống hệt pipeline_g, thêm một việc: đánh dấu frame mà VÙNG TEXT thay đổi
(Stage 1d) và miễn cho chúng Stage 3 + Stage 4.

Lý do: DAKE chấm điểm theo steepness của kích thước frame nén — proxy cho "chuyển
động", không liên quan gì tới "có chữ", nên pipeline_g bỏ sót keyframe chứa text.
Union chúng vào candidate KHÔNG đủ: frame chỉ khác frame trước ở chỗ có thêm banner
chữ thì BEiT-3 whole-frame cosine ~0.98, bị Semantic Filter (0.93) loại trước cả khi
tới Diversity Filter. Vì vậy chúng đi theo đúng đường bypass của forced_set.

Flow:
  ALL FRAMES (shot/video)
    → Stage 1a: DAKE Coarse Candidate Selection
    → Stage 1b: Transition Peak Detection
    → Stage 1d: Text Region Change Pre-scan  ← MỚI so với pipeline_g
    → Stage 1c: Blank/Black/White Transition Cut Hard-Veto
    → Stage 2:  Visual Encoder (BEiT-3 Large)
    → Stage 3:  Global Semantic Filter (frame text-protected được miễn)
    → Stage 4:  Diversity / Redundancy Filter (frame text-protected đi vòng)
    → Stage 5:  Local Sharpness Reselection
    → Keyframes
"""

from __future__ import annotations

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

_beit3_src = Path(__file__).resolve().parents[2] / "beit3_src"
if str(_beit3_src) not in sys.path:
    sys.path.insert(0, str(_beit3_src))

from src.core.interfaces import BaseKeyframeExtractor
from src.core.models import ShotRecord, ShotKeyframes
from src.components.frame_loader import KeyframeDirLoader, VideoFrameLoader
from src.components.dake import DAKESelector
from src.components.beit3_encoder import BEiT3Encoder
from src.components.semantic_filter import SemanticFilter
from src.components.diversity_filter import DiversityRedundancyFilter
from src.components.transition_selector import TransitionAwareSelector
from src.components.blank_veto import BlankTransitionVetoFilter
from src.components.sharpness_selector import SharpnessReselector
from src.components.text_prescan import TextRegionScanner
from src.extractors.pipeline_g import (
    _MAX_WINDOW_FRAMES,
    _fill_candidate_gaps,
    _make_keyframes,
)


class PipelineH(BaseKeyframeExtractor):
    """
    Pipeline H: pipeline_g + text-aware keyframe protection.

    Args:
        (giống PipelineG, thêm:)
        enable_text_prescan   : Bật Stage 1d.
        text_stride_seconds   : Lấy mẫu dò text mỗi N giây (quy đổi sang frame bằng fps).
        text_band_ratio       : Ngưỡng edge-energy để coi dải top/bottom là có text.
        text_change_threshold : Hamming distance tối thiểu (0-64) để coi chữ đã đổi.
        text_min_gap_seconds  : Khoảng cách tối thiểu giữa 2 frame text-protected.
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
        enable_text_prescan: bool = True,
        text_stride_seconds: float = 1.0,
        text_band_ratio: float = 2.0,
        text_change_threshold: int = 8,
        text_min_gap_seconds: float = 2.0,
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.spm_path = Path(spm_path)
        self.enable_transition = enable_transition
        self.max_history_size = max_history_size
        self.max_gap_frames = max_gap_frames
        self.enable_text_prescan = enable_text_prescan
        self.text_stride_seconds = text_stride_seconds
        self.text_band_ratio = text_band_ratio
        self.text_change_threshold = text_change_threshold
        self.text_min_gap_seconds = text_min_gap_seconds
        self.device = device
        self.batch_size = batch_size

        self._dake = DAKESelector(candidate_ratio=candidate_ratio, window_size=window_size)
        self._encoder: BEiT3Encoder | None = None
        self._semantic_filter = SemanticFilter(
            similarity_threshold=similarity_threshold,
            min_frame_distance=min_frame_distance,
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
        return "pipeline_h"

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

        loader_cls = KeyframeDirLoader if video_path.is_dir() else VideoFrameLoader
        with loader_cls(video_path) as loader:
            fps = loader.fps
            # Knob text tính bằng giây, quy đổi ở đây: batch1 lẫn nhiều FPS, 25 vs
            # 60fps lệch 2.4x nếu để nguyên đơn vị frame. Stride đếm theo số mẫu đã đọc
            # nên với keyframe BTC (mỗi mẫu cách >= 5 frame) phải chia frame_step.
            text_scanner = (
                TextRegionScanner(
                    stride=max(1, round(self.text_stride_seconds * fps / loader.frame_step)),
                    band_ratio=self.text_band_ratio,
                    change_threshold=self.text_change_threshold,
                    min_gap_frames=max(1, round(self.text_min_gap_seconds * fps)),
                )
                if self.enable_text_prescan
                else None
            )

            for shot in shots:
                shot_indices: List[int] = []
                shot_embeddings: List[np.ndarray] = []
                shot_jpegs: List[bytes] = []
                fallback: Optional[Tuple[int, np.ndarray, bytes]] = None

                for w_start in range(shot.start_frame, shot.end_frame + 1, _MAX_WINDOW_FRAMES):
                    w_end = min(w_start + _MAX_WINDOW_FRAMES - 1, shot.end_frame)
                    all_frames = loader.read_range(w_start, w_end)
                    if not all_frames:
                        continue

                    frame_idx_list = [idx for idx, _ in all_frames]

                    # Stage 1a: DAKE Candidate Selection
                    candidate_indices, steepness, aggregated_scores = self._dake.select_with_scores(all_frames)
                    candidate_map = {idx: img for idx, img in all_frames}
                    steepness_map = {
                        idx: score for idx, score in zip(frame_idx_list, aggregated_scores)
                    }

                    # Stage 1b: Transition Peak Detection
                    if self.enable_transition:
                        trans_peaks = self._transition_selector.find_transition_peaks(
                            frame_idx_list, steepness
                        )
                        combined_indices = sorted(set(candidate_indices) | trans_peaks)
                    else:
                        combined_indices = candidate_indices

                    # Stage 1d: Text Region Change Pre-scan
                    text_protected = text_scanner.scan(all_frames) if text_scanner else set()
                    if text_protected:
                        combined_indices = sorted(set(combined_indices) | text_protected)

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

                    # Frame text-protected vẫn phải sống sót qua veto như mọi candidate
                    # khác (frame có chữ không blank, nhưng fallback ở trên có thể đã
                    # thay sạch danh sách).
                    text_protected &= {idx for idx, _ in candidate_frames}

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

                    # Stage 3: Global Semantic Filter — frame text-protected được miễn
                    # similarity check (vẫn chịu min_frame_distance).
                    sem_indices, sem_embeddings, forced_set = self._semantic_filter.filter(
                        enc_indices,
                        embeddings,
                        history_embeddings=video_history,
                        last_keyframe_idx=last_keyframe_idx,
                        protected=text_protected,
                    )

                    # Frame force-pick (coverage ceiling) VÀ frame text-protected đều đi
                    # thẳng ra output, KHÔNG qua Diversity Filter: cả hai nhóm vốn rất
                    # giống nhau về whole-frame embedding — đó chính là lý do chúng cần
                    # được bảo vệ, nên để redundancy_threshold xét là xoá sạch chúng.
                    bypass = forced_set | text_protected
                    forced_pairs = [
                        (idx, emb) for idx, emb in zip(sem_indices, sem_embeddings) if idx in bypass
                    ]
                    normal_pairs = [
                        (idx, emb) for idx, emb in zip(sem_indices, sem_embeddings) if idx not in bypass
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
