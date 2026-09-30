"""
pipeline.py -- orchestrator cho OCR stage 1

run_ocr_pipeline: GPU detection (DeepSolo) + recognition (PARSeq-VN), đã hiệu
đính dấu tiếng Việt ngay trong từng frame qua corrector.correct_record_locally.

Frame giống/near-giống frame trước thì dùng lại kết quả thay vì chạy lại
inference (GPU).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from . import frame_skip
from .corrector import correct_record_locally
from .deepsolo_engine import DeepSoloParseqEngine
from .formatter import load_checkpoint, save_output
from .loader import load_frames


def _build_engine(engine_cfg: dict) -> DeepSoloParseqEngine:
    return DeepSoloParseqEngine(
        det_threshold=engine_cfg.get("det_threshold", 0.15),
        min_size=engine_cfg.get("min_size", 1080),
    )


def _process_video_frames(
    group: list[tuple[str, str]],
    engine: DeepSoloParseqEngine,
    blur_thresh: float,
    preprocess: bool,
) -> tuple[list[dict], list[dict]]:
    """Run the same per-frame skip-heuristic + engine.run() loop as
    run_ocr_pipeline, but scoped to one video's frames -- used by the
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
        texts, _ = engine.run(source)
        record = correct_record_locally({"frame_id": frame_id, "texts": texts})
        records.append(record)
        records_origin.append({"frame_id": frame_id, "texts": []})

    return records, records_origin


def _run_ocr_claimed(frames: list[tuple[str, str]], claims_dir: str, cfg: dict) -> None:
    """Worker-pool mode for parallel shards: each video is claimed atomically
    (O_EXCL file create, safe across concurrent containers/NFS) before
    processing, and its result written to claims_dir/done/<video>.json.
    A video already claimed or done is skipped -- so any number of these can
    run concurrently against the same claims_dir, and adding/removing one
    doesn't require restarting the others or recomputing what's done."""
    skip_cfg: dict = cfg.get("skip", {})
    blur_thresh = skip_cfg.get("blur_threshold", 0.0)
    preprocess: bool = cfg.get("preprocess", False)

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
            engine = _build_engine(cfg)

        records, records_origin = _process_video_frames(group, engine, blur_thresh, preprocess)
        done_marker = done_dir / f"{video_id}.json"
        tmp = done_marker.with_suffix(".json.tmp")
        # Giữ schema {"vietocr", "paddle_origin"} cho tương thích với các file
        # done cũ (batch2) -- "vietocr" ở đây là output PARSeq, origin luôn rỗng.
        tmp.write_text(
            json.dumps({"vietocr": records, "paddle_origin": records_origin}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(done_marker)
        n_processed += 1

    print(f"[->] Claimed+processed {n_processed}/{len(by_video)} video this run "
          f"(rest already done or claimed by another shard).")


def run_ocr_pipeline(cfg: dict) -> None:
    input_dir: str = cfg["input_dir"]
    output_file: str = cfg["output_file"]
    limit: int = cfg.get("limit", 0) or None
    checkpoint_every: int = cfg.get("checkpoint_every", 0)
    skip_cfg: dict = cfg.get("skip", {})
    preprocess: bool = cfg.get("preprocess", False)
    claims_dir: str | None = cfg.get("claims_dir")

    blur_thresh = skip_cfg.get("blur_threshold", 0.0)

    frames = load_frames(input_dir)[:limit] if limit else load_frames(input_dir)

    if claims_dir:
        _run_ocr_claimed(frames, claims_dir, cfg)
        return

    done = load_checkpoint(output_file) if checkpoint_every else {}

    engine = _build_engine(cfg)
    print(f"[->] {len(frames)} frames, preprocess={preprocess}")

    t0 = time.time()
    records: list[dict] = []

    for idx, (frame_id, path) in enumerate(tqdm(frames, desc="ocr", unit="frame")):
        if checkpoint_every and idx > 0 and idx % checkpoint_every == 0:
            save_output(records, output_file)

        if frame_id in done:
            records.append(done[frame_id])
            continue

        need_gray = blur_thresh or preprocess
        bgr = gray = None
        if need_gray:
            frame_data = frame_skip.read_frame(path)
            if frame_data is None:
                records.append({"frame_id": frame_id, "texts": []})
                continue
            bgr, gray = frame_data

        if blur_thresh and frame_skip.is_blurry(gray, blur_thresh):
            records.append({"frame_id": frame_id, "texts": []})
            continue

        source = frame_skip.preprocess(bgr) if preprocess else path
        texts, _ = engine.run(source)
        records.append(correct_record_locally({"frame_id": frame_id, "texts": texts}))

    elapsed = time.time() - t0
    print(f"[->] {len(frames)} frames in {elapsed:.1f}s ({len(frames) / max(elapsed, 1e-9):.2f} fps)")
    if hasattr(engine, "det_ms"):
        print(f"[->] det {engine.det_ms / max(len(frames), 1):.0f} ms/frame, rec {engine.rec_ms / max(len(frames), 1):.0f} ms/frame")
    save_output(records, output_file)
    print(f"[->] Saved {len(records)} records -> {Path(output_file).resolve()}")
