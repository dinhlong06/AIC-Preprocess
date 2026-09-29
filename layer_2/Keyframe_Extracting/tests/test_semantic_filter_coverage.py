import numpy as np

from src.components.semantic_filter import SemanticFilter


def test_force_pick_triggers_on_long_static_run():
    """
    Mọi candidate có embedding giống hệt keyframe trước (max_sim ~ 1.0) — theo
    similarity-threshold thuần tuý sẽ KHÔNG có candidate nào được chọn, dù shot
    kéo dài bao lâu. Coverage ceiling (max_gap_frames) phải ép chọn ít nhất 1
    candidate mỗi khi gap vượt ngưỡng.
    """
    filt = SemanticFilter(
        similarity_threshold=0.90,
        min_frame_distance=5,
        gap_decay_start_frames=150,
        max_gap_frames=300,
    )

    same_vec = np.array([1.0, 0.0], dtype=np.float32)
    # Candidate dày (mỗi 20 frame) trên toàn bộ 700 frame — mô phỏng đúng thực
    # tế: DAKE/transition sinh candidate liên tục, force-pick phải trigger
    # nhiều lần (không phải chỉ 1 lần như khi candidate quá thưa).
    frame_indices = list(range(20, 700, 20))
    embeddings = np.stack([same_vec.copy() for _ in frame_indices])

    selected_indices, _, forced_indices = filt.filter(
        frame_indices,
        embeddings,
        history_embeddings=[same_vec.copy()],
        last_keyframe_idx=0,
    )

    assert len(selected_indices) >= 2, (
        "coverage ceiling phải force-pick nhiều lần trong 1 shot tĩnh dài "
        f"700 frame, nhưng chỉ chọn được {selected_indices}"
    )
    # Mọi frame chọn được trong kịch bản toàn-giống-nhau này đều phải là force-pick
    # (bình thường sẽ không có candidate nào qua nổi similarity check).
    assert forced_indices == set(selected_indices)
    # Không khoảng trống nào giữa 2 lần chọn liên tiếp vượt quá max_gap_frames
    # cộng thêm khoảng cách candidate xa nhất (force-pick chỉ trigger khi có
    # candidate mới tới, không chủ động quét trước).
    prev = 0
    for idx in selected_indices:
        assert idx - prev <= 305
        prev = idx


def test_similarity_only_would_select_nothing_without_ceiling():
    """Đối chứng: nếu tắt coverage ceiling (max_gap_frames rất lớn), không có
    candidate nào được chọn — chứng minh test trên thực sự nhờ force-pick."""
    filt = SemanticFilter(
        similarity_threshold=0.90,
        min_frame_distance=5,
        gap_decay_start_frames=10_000,
        max_gap_frames=10_000,
    )
    same_vec = np.array([1.0, 0.0], dtype=np.float32)
    frame_indices = [50, 100, 150, 200, 250, 305, 350]
    embeddings = np.stack([same_vec.copy() for _ in frame_indices])

    selected_indices, _, forced_indices = filt.filter(
        frame_indices,
        embeddings,
        history_embeddings=[same_vec.copy()],
        last_keyframe_idx=0,
    )
    assert selected_indices == []
    assert forced_indices == set()


def test_protected_frame_survives_identical_embedding():
    """
    Frame mà vùng text đổi có whole-frame embedding gần như trùng khớp keyframe
    trước (chỉ khác cái banner chữ) — similarity check luôn loại. protected phải
    miễn nó khỏi check đó.
    """
    filt = SemanticFilter(similarity_threshold=0.90, min_frame_distance=5)
    same_vec = np.array([1.0, 0.0], dtype=np.float32)
    frame_indices = [50, 100, 150]
    embeddings = np.stack([same_vec.copy() for _ in frame_indices])

    selected, _, forced = filt.filter(
        frame_indices,
        embeddings,
        history_embeddings=[same_vec.copy()],
        last_keyframe_idx=0,
        protected={100},
    )

    assert selected == [100]
    # Không phải force-pick của coverage ceiling — caller phân biệt hai nhóm này.
    assert forced == set()


def test_protected_frame_still_obeys_min_frame_distance():
    """Miễn similarity check KHÔNG có nghĩa là được sinh keyframe sát nhau."""
    filt = SemanticFilter(similarity_threshold=0.90, min_frame_distance=5)
    same_vec = np.array([1.0, 0.0], dtype=np.float32)
    frame_indices = [100, 102]
    embeddings = np.stack([same_vec.copy() for _ in frame_indices])

    selected, _, _ = filt.filter(
        frame_indices,
        embeddings,
        history_embeddings=[same_vec.copy()],
        last_keyframe_idx=0,
        protected={100, 102},
    )

    assert selected == [100]
