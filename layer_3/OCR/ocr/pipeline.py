"""
pipeline.py -- orchestrator cho OCR stage 1

run_paddle_pipeline: GPU detection+recognition (PaddleOCR PP-OCRv6), đã hiệu
đính dấu tiếng Việt ngay trong từng frame qua corrector.correct_record_locally.

Chia sẻ ý tưởng skip-heuristic (ported từ một pipeline Paddle+VietOCR cũ, đã
bỏ): frame giống/near-giống frame trước thì dùng lại kết quả thay vì chạy
lại inference (GPU).
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from . import frame_skip
from .corrector import correct_record_locally
from .formatter import load_checkpoint, save_output
from .loader import load_frames
from .paddle_engine import PaddleEngine


def _build_engine(engine_cfg: dict):
    return PaddleEngine(
        lang=engine_cfg.get("lang", "vi"),
        ocr_version=engine_cfg.get("ocr_version"),
        unclip_ratio=engine_cfg.get("unclip_ratio"),
    )


def _process_video_frames(
    group: list[tuple[str, str]],
    engine,
    blur_thresh: float,
    preprocess: bool,
) -> tuple[list[dict], list[dict]]:
    """Run the same per-frame skip-heuristic + engine.run() loop as
    run_paddle_pipeline, but scoped to one video's frames -- used by the
    claims-dir path where each video is processed and checkpointed in
    isolation."""
    records: list[dict] = []
    records_origin: list[dict] = []

    for frame_id, path in group:
        need_gray = blur_thresh or preprocess
        bgr = gray = None
        if need_gray:
            frame_data = frame_skip.read_frame(path)
            if frame_data is None:
                records.append({"frame_id": frame_id, "texts": []})
                records_origin.append({"frame_id": frame_id, "texts": []})
                continue
            bgr, gray = frame_data

        if blur_thresh and frame_skip.is_blurry(gray, blur_thresh):
            records.append({"frame_id": frame_id, "texts": []})
            records_origin.append({"frame_id": frame_id, "texts": []})
            continue

        source = frame_skip.preprocess(bgr) if preprocess else path
        texts, texts_origin = engine.run(source)
        record = correct_record_locally({"frame_id": frame_id, "texts": texts})
        record_origin = {"frame_id": frame_id, "texts": texts_origin}
        records.append(record)
        records_origin.append(record_origin)

    return records, records_origin


def _run_paddle_claimed(frames: list[tuple[str, str]], claims_dir: str, cfg: dict) -> None:
    """Worker-pool mode for parallel shards: each video is claimed atomically
    (O_EXCL file create, safe across concurrent containers/NFS) before
    processing, and its result written to claims_dir/done/<video>.json.
    A video already claimed or done is skipped -- so any number of these can
    run concurrently against the same claims_dir, and adding/removing one
    doesn't require restarting the others or recomputing what's done."""
    skip_cfg: dict = cfg.get("skip", {})
    blur_thresh = skip_cfg.get("blur_threshold", 0.0)
    preprocess: bool = cfg.get("preprocess", False)
    engine_cfg = cfg.get("ocr", {})

    claimed_dir = Path(claims_dir) / "claimed"
    done_dir = Path(claims_dir) / "done"
    claimed_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    by_video: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for frame_id, path in frames:
        by_video[Path(path).parent.name].append((frame_id, path))

    engine = None
    n_processed = 0
    for video_id, group in tqdm(by_video.items(), desc="claim", unit="video"):
        if (done_dir / f"{video_id}.json").exists():
            continue
        # Chạy song song layer_2: video chưa có statistics.json là đang ghi dở ảnh, bỏ
        # qua để vòng sau nhận (claim rồi thì không bao giờ làm lại).
        if not (Path(group[0][1]).parent / "statistics.json").exists():
            continue
        try:
            (claimed_dir / video_id).touch(exist_ok=False)
        except FileExistsError:
            continue

        if engine is None:
            engine = _build_engine(engine_cfg)

        records, records_origin = _process_video_frames(group, engine, blur_thresh, preprocess)
        done_marker = done_dir / f"{video_id}.json"
        tmp = done_marker.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"vietocr": records, "paddle_origin": records_origin}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(done_marker)
        n_processed += 1

    print(f"[->] Claimed+processed {n_processed}/{len(by_video)} video this run "
          f"(rest already done or claimed by another shard).")


def run_paddle_pipeline(cfg: dict) -> None:
    input_dir: str = cfg["input_dir"]
    output_file: str = cfg["output_file"]
    output_file_paddle_origin: str = cfg["output_file_paddle_origin"]
    limit: int = cfg.get("limit", 0) or None
    checkpoint_every: int = cfg.get("checkpoint_every", 0)
    skip_cfg: dict = cfg.get("skip", {})
    preprocess: bool = cfg.get("preprocess", False)
    claims_dir: str | None = cfg.get("claims_dir")

    blur_thresh = skip_cfg.get("blur_threshold", 0.0)

    frames = load_frames(input_dir)[:limit] if limit else load_frames(input_dir)

    if claims_dir:
        _run_paddle_claimed(frames, claims_dir, cfg)
        return

    done = load_checkpoint(output_file) if checkpoint_every else {}
    done_origin = load_checkpoint(output_file_paddle_origin) if checkpoint_every else {}
    if done:
        print(f"[->] Resume: {len(done)}/{len(frames)} frames already done, skipping.")

    engine_cfg = cfg.get("ocr", {})
    print(f"[->] {len(frames)} frames, engine={engine_cfg.get('engine', 'paddle')}, preprocess={preprocess}")
    engine = _build_engine(engine_cfg)

    t0 = time.time()
    records: list[dict] = []
    records_origin: list[dict] = []
    skipped_blur = 0

    for idx, (frame_id, path) in enumerate(tqdm(frames, desc="paddle", unit="frame")):
        if checkpoint_every and idx > 0 and idx % checkpoint_every == 0:
            save_output(records, output_file)
            save_output(records_origin, output_file_paddle_origin)

        if frame_id in done:
            records.append(done[frame_id])
            records_origin.append(done_origin.get(frame_id, {"frame_id": frame_id, "texts": []}))
            continue

        need_gray = blur_thresh or preprocess
        bgr = gray = None
        if need_gray:
            frame_data = frame_skip.read_frame(path)
            if frame_data is None:
                records.append({"frame_id": frame_id, "texts": []})
                records_origin.append({"frame_id": frame_id, "texts": []})
                continue
            bgr, gray = frame_data

        if blur_thresh and frame_skip.is_blurry(gray, blur_thresh):
            skipped_blur += 1
            records.append({"frame_id": frame_id, "texts": []})
            records_origin.append({"frame_id": frame_id, "texts": []})
            continue

        source = frame_skip.preprocess(bgr) if preprocess else path
        texts, texts_origin = engine.run(source)
        record = correct_record_locally({"frame_id": frame_id, "texts": texts})
        record_origin = {"frame_id": frame_id, "texts": texts_origin}
        records.append(record)
        records_origin.append(record_origin)

    elapsed = time.time() - t0
    print(f"[->] Skipped -- blur={skipped_blur}")
    print(f"[->] {len(frames)} frames in {elapsed:.1f}s ({len(frames) / max(elapsed, 1e-9):.2f} fps)")
    if hasattr(engine, "det_ms"):
        print(f"[->] det {engine.det_ms / max(len(frames), 1):.0f} ms/frame, rec {engine.rec_ms / max(len(frames), 1):.0f} ms/frame")
    save_output(records, output_file)
    save_output(records_origin, output_file_paddle_origin)
    print(f"[->] Saved {len(records)} records -> {Path(output_file).resolve()}")
    print(f"[->] Saved {len(records_origin)} records -> {Path(output_file_paddle_origin).resolve()}")
