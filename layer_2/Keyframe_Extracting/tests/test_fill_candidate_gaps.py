from src.extractors.pipeline_g import _fill_candidate_gaps


def test_no_gap_exceeds_max_gap_when_input_sparse():
    frame_idx_list = list(range(0, 1000))
    combined_indices = [0, 999]  # DAKE chỉ chọn 2 candidate, để trống ở giữa
    filled = _fill_candidate_gaps(combined_indices, frame_idx_list, w_start=0, max_gap=100)

    assert filled[0] == 0
    for a, b in zip(filled, filled[1:]):
        assert b - a <= 100


def test_keeps_all_original_candidates():
    frame_idx_list = list(range(0, 500))
    combined_indices = [10, 20, 30]
    filled = _fill_candidate_gaps(combined_indices, frame_idx_list, w_start=0, max_gap=1000)
    assert set(combined_indices).issubset(set(filled))


def test_empty_input_returns_empty():
    assert _fill_candidate_gaps([], [], w_start=0, max_gap=300) == []
