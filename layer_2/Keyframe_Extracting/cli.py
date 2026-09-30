"""
cli.py — Command Line Interface cho Keyframe Extractor Benchmark.

Sử dụng:
  python cli.py --pipeline pipeline_g --video_dir dataset/raw_video --shots_dir dataset/shots

  hoặc dùng file config yaml:
  python cli.py --config configs/pipeline_h.yaml --video_dir dataset/raw_video

Toàn bộ tham số cũng có thể override qua CLI flag sau khi chỉ định --config.
"""

import argparse
import sys
from pathlib import Path
import yaml


def load_config(config_path: str) -> dict:
    """
    Đọc file YAML config và trả về dict.

    Args:
        config_path : Đường dẫn đến file .yaml.

    Returns:
        Dict chứa cấu hình keyframe pipeline.
    """
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pipeline_g(cfg: dict, args: argparse.Namespace):
    """Khởi tạo PipelineG (Upgraded multi-stage pipeline) từ config + CLI args."""
    from src.extractors.pipeline_g import PipelineG
    beit3_cfg = cfg.get("beit3", {})
    dake_cfg = cfg.get("dake", {})
    semantic_cfg = cfg.get("semantic", {})
    diversity_cfg = cfg.get("diversity", {})
    transition_cfg = cfg.get("transition", {})
    veto_cfg = cfg.get("veto", {})
    sharpness_cfg = cfg.get("sharpness", {})
    min_dist = args.min_distance or cfg.get("min_frame_distance", 5)
    return PipelineG(
        checkpoint_path=args.checkpoint_path or beit3_cfg.get("checkpoint_path", ""),
        spm_path=args.spm_path or beit3_cfg.get("spm_path", ""),
        candidate_ratio=args.candidate_ratio or dake_cfg.get("candidate_ratio", 0.05),
        window_size=dake_cfg.get("window_size", 3),
        min_frame_distance=min_dist,
        similarity_threshold=args.threshold or semantic_cfg.get("similarity_threshold", 0.90),
        redundancy_threshold=diversity_cfg.get("redundancy_threshold", 0.88),
        peak_prominence_window=transition_cfg.get("prominence_window", 3),
        peak_percentile=transition_cfg.get("peak_percentile", 90.0),
        min_keyframes_per_shot=diversity_cfg.get("min_keyframes_per_shot", 1),
        enable_transition=transition_cfg.get("enabled", True),
        max_history_size=semantic_cfg.get("max_history_size", 512),
        gap_decay_start_frames=semantic_cfg.get("gap_decay_start_frames", 150),
        max_gap_frames=semantic_cfg.get("max_gap_frames", 300),
        enable_veto=veto_cfg.get("enabled", True),
        veto_min_brightness=veto_cfg.get("min_brightness", 15.0),
        veto_max_brightness=veto_cfg.get("max_brightness", 240.0),
        veto_min_variance=veto_cfg.get("min_variance", 15.0),
        enable_sharpness=sharpness_cfg.get("enabled", True),
        sharpness_window_radius=sharpness_cfg.get("window_radius", 4),
        device=args.device or beit3_cfg.get("device"),
        batch_size=args.batch_size or beit3_cfg.get("batch_size", 32),
    )


def build_pipeline_h(cfg: dict, args: argparse.Namespace):
    """Khởi tạo PipelineH (pipeline_g + text-awareness) từ config + CLI args."""
    from src.extractors.pipeline_h import PipelineH
    beit3_cfg = cfg.get("beit3", {})
    dake_cfg = cfg.get("dake", {})
    semantic_cfg = cfg.get("semantic", {})
    diversity_cfg = cfg.get("diversity", {})
    transition_cfg = cfg.get("transition", {})
    veto_cfg = cfg.get("veto", {})
    sharpness_cfg = cfg.get("sharpness", {})
    text_cfg = cfg.get("text_prescan", {})
    min_dist = args.min_distance or cfg.get("min_frame_distance", 5)
    return PipelineH(
        checkpoint_path=args.checkpoint_path or beit3_cfg.get("checkpoint_path", ""),
        spm_path=args.spm_path or beit3_cfg.get("spm_path", ""),
        candidate_ratio=args.candidate_ratio or dake_cfg.get("candidate_ratio", 0.05),
        window_size=dake_cfg.get("window_size", 3),
        min_frame_distance=min_dist,
        similarity_threshold=args.threshold or semantic_cfg.get("similarity_threshold", 0.90),
        redundancy_threshold=diversity_cfg.get("redundancy_threshold", 0.88),
        peak_prominence_window=transition_cfg.get("prominence_window", 3),
        peak_percentile=transition_cfg.get("peak_percentile", 90.0),
        min_keyframes_per_shot=diversity_cfg.get("min_keyframes_per_shot", 1),
        enable_transition=transition_cfg.get("enabled", True),
        max_history_size=semantic_cfg.get("max_history_size", 512),
        gap_decay_start_frames=semantic_cfg.get("gap_decay_start_frames", 150),
        max_gap_frames=semantic_cfg.get("max_gap_frames", 300),
        enable_veto=veto_cfg.get("enabled", True),
        veto_min_brightness=veto_cfg.get("min_brightness", 15.0),
        veto_max_brightness=veto_cfg.get("max_brightness", 240.0),
        veto_min_variance=veto_cfg.get("min_variance", 15.0),
        enable_sharpness=sharpness_cfg.get("enabled", True),
        sharpness_window_radius=sharpness_cfg.get("window_radius", 4),
        enable_text_prescan=text_cfg.get("enabled", True),
        text_stride_seconds=text_cfg.get("stride_seconds", 1.0),
        text_band_ratio=text_cfg.get("band_ratio", 2.0),
        text_change_threshold=text_cfg.get("change_threshold", 8),
        text_min_gap_seconds=text_cfg.get("min_gap_seconds", 2.0),
        device=args.device or beit3_cfg.get("device"),
        batch_size=args.batch_size or beit3_cfg.get("batch_size", 32),
    )


_PIPELINE_BUILDERS = {
    "pipeline_g": (build_pipeline_g, "configs/pipeline_g.yaml"),
    "pipeline_h": (build_pipeline_h, "configs/pipeline_h.yaml"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Keyframe Extractor Benchmark — AI Challenge 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Chạy Pipeline G (Upgraded multi-stage pipeline):
  python cli.py --pipeline pipeline_g --video_dir dataset/raw_video

  # Chạy cả G và H:
  python cli.py --pipeline all --video_dir dataset/raw_video

  # Override tham số:
  python cli.py --pipeline pipeline_g --threshold 0.88 --candidate_ratio 0.05
        """,
    )

    # Pipeline selection
    parser.add_argument(
        "--pipeline",
        type=str,
        default="pipeline_g",
        choices=["pipeline_g", "pipeline_h", "all"],
        help="Pipeline để chạy. 'all' sẽ chạy lần lượt G và H.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Đường dẫn đến file YAML config (override mặc định).",
    )

    # I/O paths
    parser.add_argument(
        "--video_dir",
        type=str,
        default="dataset/raw_video",
        help="Thư mục chứa video .mp4.",
    )
    parser.add_argument(
        "--shots_dir",
        type=str,
        default="dataset/shots",
        help="Thư mục chứa shots.json của từng video.",
    )
    parser.add_argument(
        "--no-shots",
        action="store_true",
        default=False,
        help="Không sử dụng shot detection, trích xuất trực tiếp trên toàn bộ video.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark",
        help="Thư mục gốc lưu kết quả benchmark.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Thư mục chứa Ground Truth ảnh (.png/.jpg) của từng video để đánh giá.",
    )
    parser.add_argument(
        "--videos",
        type=str,
        default=None,
        help="Giới hạn tên video cụ thể, cách nhau bởi dấu phẩy (vd: K01_V001.mp4,K01_V002.mp4). Không truyền = chạy toàn bộ video_dir.",
    )

    # Model overrides
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path đến BEiT-3 checkpoint .pth (override config).")
    parser.add_argument("--spm_path", type=str, default=None,
                        help="Path đến BEiT-3 .spm tokenizer (override config).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda' hoặc 'cpu' (override config).")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size cho encoder (override config).")

    # Algorithm params
    parser.add_argument("--threshold", type=float, default=None,
                        help="Cosine similarity threshold [0-1] for extraction deduplication (override config).")
    parser.add_argument("--candidate_ratio", type=float, default=None,
                        help="DAKE candidate ratio [0.01-0.20] (override config).")
    parser.add_argument("--min_distance", type=int, default=None,
                        help="Khoảng cách khung hình tối thiểu giữa 2 keyframe.")

    # Evaluation flags
    parser.add_argument(
        "--evaluate_keyframes",
        action="store_true",
        default=False,
        help="Bật đánh giá chất lượng keyframe so với Ground Truth (cần --gt_dir).",
    )
    parser.add_argument(
        "--eval_threshold",
        type=float,
        default=0.85,
        help="Ngưỡng cosine similarity τ cho GT matching khi --evaluate_keyframes (default 0.85).",
    )

    args = parser.parse_args()

    # Validate paths
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"[CLI] ERROR: video_dir không tồn tại: {video_dir}")
        sys.exit(1)

    # kf_batch2 (keyframe BTC cắt sẵn) có metadata.json ở gốc: mỗi thư mục con là một
    # "video". Không rglob ở đây: ~1 triệu file webp trên NFS.
    if (video_dir / "metadata.json").exists():
        video_paths = sorted(p for p in video_dir.iterdir() if p.is_dir())
    else:
        video_paths = sorted(p for ext in ("*.mp4", "*.mov") for p in video_dir.rglob(ext))
    if args.videos:
        wanted = set(args.videos.split(","))
        video_paths = [p for p in video_paths if p.name in wanted or p.stem in wanted]
    if not video_paths:
        print(f"[CLI] ERROR: Không tìm thấy .mp4/.mov trong {video_dir}")
        sys.exit(1)

    print(f"[CLI] Found {len(video_paths)} video(s) in {video_dir}")

    shots_dir = Path(args.shots_dir)
    output_dir = Path(args.output_dir)

    # Xác định danh sách pipeline cần chạy
    if args.pipeline == "all":
        pipeline_names = list(_PIPELINE_BUILDERS.keys())
    else:
        pipeline_names = [args.pipeline]

    # Import runner
    from src.core.runner import KeyframeBenchmarkRunner
    runner = KeyframeBenchmarkRunner(output_dir=output_dir)

    # Chạy từng pipeline
    for pipeline_name in pipeline_names:
        builder_fn, default_config = _PIPELINE_BUILDERS[pipeline_name]

        # Load config
        config_path = args.config or default_config
        if Path(config_path).exists():
            cfg = load_config(config_path).get("keyframe", {})
        else:
            print(f"[CLI] WARNING: Config không tìm thấy: {config_path}. Dùng giá trị mặc định.")
            cfg = {}

        extractor = builder_fn(cfg, args)
        gt_dir = Path(args.gt_dir) if args.gt_dir else None

        if args.evaluate_keyframes and not gt_dir:
            print("[CLI] WARNING: --evaluate_keyframes được bật nhưng không có --gt_dir. Bỏ qua evaluation.")

        runner.run(
            extractor,
            video_paths,
            None if (args.no_shots or not shots_dir.exists()) else shots_dir,
            gt_dir=gt_dir,
            evaluate=args.evaluate_keyframes,
            eval_threshold=args.eval_threshold,
        )

    print(f"\n[CLI] Benchmark hoàn thành. Kết quả tại: {output_dir.resolve()}")
    print(f"[CLI] Summary CSV: {(output_dir / 'benchmark_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
