"""
runner.py — Keyframe Benchmark Runner.

Điều phối toàn bộ quá trình benchmark:
  1. Đọc shots.json / shot.jsonl của từng video.
  2. Gọi extractor.extract() để lấy keyframe.
  3. Lưu ảnh keyframe vào thư mục output.
  4. Ghi keyframes.jsonl (metadata).
  5. Lưu embedding .npy và ids .json (nếu encoder trả về embedding).
  6. Tính và ghi statistics.json.
  7. Append vào benchmark_summary.csv.

Không biết chi tiết của pipeline — chỉ giao tiếp qua BaseKeyframeExtractor.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.core.interfaces import BaseKeyframeExtractor
from src.core.models import ShotRecord, ShotKeyframes, PipelineStatistics
from src.core.metrics import (
    compute_diversity_score,
    compute_coverage_score,
    evaluate_with_ground_truth,
)


_MAX_VIDEO_RETRIES = 2


def _claim(video_id: str) -> bool:
    """Giành video giữa nhiều shard. O_CREAT|O_EXCL nguyên tử trên NFSv4 nên đúng
    một shard thắng. CLAIMS_DIR rỗng = chạy đơn, luôn thắng.

    Ghi SHARD_ID vào file để _reclaim_own_stale() biết claim nào là của mình."""
    claims = os.environ.get("CLAIMS_DIR")
    if not claims:
        return True
    os.makedirs(claims, exist_ok=True)
    try:
        fd = os.open(os.path.join(claims, video_id), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, os.environ.get("SHARD_ID", "").encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _reclaim_own_stale(pipeline_out: Path) -> int:
    """Job chết giữa lúc xử lý một video sẽ để lại claim mồ côi, và vì claim không
    bao giờ được xoá, video đó bị bỏ qua ở MỌI lần chạy lại mà không báo gì.

    Chỉ xoá claim mang đúng SHARD_ID của mình: shard khác có thể đang chạy dở video
    của nó, xoá claim của chúng sẽ làm hai shard cùng ghi vào một thư mục."""
    claims = os.environ.get("CLAIMS_DIR")
    if not claims or not os.path.isdir(claims):
        return 0
    me = os.environ.get("SHARD_ID", "")
    reclaimed = 0
    for video_id in os.listdir(claims):
        if (pipeline_out / video_id / "statistics.json").exists():
            continue
        path = os.path.join(claims, video_id)
        try:
            with open(path) as f:
                owner = f.read()
            if owner == me:
                os.unlink(path)
                reclaimed += 1
        except OSError:
            continue
    return reclaimed


def _failure_count(pipeline_out: Path, video_id: str) -> int:
    """Số lần video này đã lỗi ở các lần chạy trước (đếm bền vững qua restart)."""
    path = pipeline_out / "_failures" / video_id
    return int(path.read_text()) if path.exists() else 0


def _record_failure(pipeline_out: Path, video_id: str) -> int:
    path = pipeline_out / "_failures" / video_id
    path.parent.mkdir(parents=True, exist_ok=True)
    count = _failure_count(pipeline_out, video_id) + 1
    path.write_text(str(count))
    return count


class KeyframeBenchmarkRunner:
    """
    Runner điều phối toàn bộ quá trình benchmark Keyframe Extraction.

    Args:
        output_dir : Thư mục gốc chứa kết quả benchmark (mặc định "benchmark/").
    """

    def __init__(self, output_dir: str | Path = "benchmark"):
        self.output_dir = Path(output_dir)
        # Nhiều shard dùng chung output_dir (mỗi video một thư mục con nên không
        # đụng nhau), chỉ file summary là điểm ghi chung -> tách theo shard.
        self.summary_csv = self.output_dir / f"benchmark_summary{os.environ.get('SHARD_ID', '')}.csv"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        extractor: BaseKeyframeExtractor,
        video_paths: List[Path],
        shots_dir: Optional[Path],
        gt_dir: Optional[Path] = None,
        evaluate: bool = False,
        eval_threshold: float = 0.85,
    ) -> None:
        """
        Chạy benchmark cho một extractor trên tất cả video.

        Args:
            extractor      : Pipeline extractor (PipelineA/B/C/D).
            video_paths    : Danh sách đường dẫn video .mp4.
            shots_dir      : Thư mục chứa shot.jsonl của từng video (None = no-shot mode).
            gt_dir         : Thư mục chứa Ground Truth keyframes.
            evaluate       : Bật chế độ đánh giá GT.
            eval_threshold : Ngưỡng cosine similarity τ cho GT matching (default 0.85).
        """
        print(f"\n{'='*60}")
        print(f"  Benchmarking: {extractor.name.upper()}")
        print(f"{'='*60}")

        pipeline_out = self.output_dir / extractor.name
        if pipeline_out.exists() and not pipeline_out.is_dir():
            pipeline_out.unlink()
        pipeline_out.mkdir(parents=True, exist_ok=True)

        reclaimed = _reclaim_own_stale(pipeline_out)
        if reclaimed:
            print(f"[Runner] Thu hồi {reclaimed} claim mồ côi của lần chạy trước.")

        extractor.setup()

        all_stats: List[PipelineStatistics] = []

        stop_flag = pipeline_out / f"_stop{os.environ.get('SHARD_ID', '')}"

        for video_path in video_paths:
            # Dừng êm: chỉ thoát ở ranh giới giữa hai video nên không mất video đang
            # làm dở, và không để lại claim mồ côi.
            if stop_flag.exists():
                stop_flag.unlink()
                print(f"[Runner] Nhận lệnh dừng êm, thoát sau khi xong video trước đó.")
                break
            if (pipeline_out / video_path.stem / "statistics.json").exists():
                print(f"[Runner] {video_path.name}: đã xong, bỏ qua.")
                continue
            fails = _failure_count(pipeline_out, video_path.stem)
            if fails >= _MAX_VIDEO_RETRIES:
                print(f"[Runner] {video_path.name}: đã lỗi {fails} lần, bỏ qua vĩnh viễn.")
                continue
            if not _claim(video_path.stem):
                continue
            try:
                stats = self._run_single_video(
                    extractor=extractor,
                    video_path=video_path,
                    shots_dir=shots_dir,
                    pipeline_out=pipeline_out,
                    gt_dir=gt_dir,
                    evaluate=evaluate,
                    eval_threshold=eval_threshold,
                )
            except Exception as exc:
                n = _record_failure(pipeline_out, video_path.stem)
                print(f"[Runner] LỖI {video_path.name} (lần {n}/{_MAX_VIDEO_RETRIES}): {type(exc).__name__}: {exc}")
                continue
            if stats:
                all_stats.append(stats)
                self._append_to_summary_csv([stats])

        extractor.teardown()
        print(f"\n[Runner] Finished: {extractor.name}. Results in {pipeline_out}\n")

    # ------------------------------------------------------------------
    # Private: single video processing
    # ------------------------------------------------------------------

    def _run_single_video(
        self,
        extractor: BaseKeyframeExtractor,
        video_path: Path,
        shots_dir: Optional[Path],
        pipeline_out: Path,
        gt_dir: Optional[Path] = None,
        evaluate: bool = False,
        eval_threshold: float = 0.85,
    ) -> Optional[PipelineStatistics]:
        """
        Xử lý một video: đọc shots, extract keyframes, lưu output, tính stats.

        Returns:
            PipelineStatistics nếu thành công, None nếu lỗi hoặc không có shot.
        """
        video_stem = video_path.stem
        print(f"\n[Runner] Processing {video_path.name} ...")

        # Đọc shots — shots_dir=None nghĩa là no-shot mode, tự chunk
        shots = self._load_shots(shots_dir, video_stem, video_path)
        if not shots:
            print(f"[Runner] WARNING: No shots found for {video_path.name}. Skipping.")
            return None

        if not self._can_decode(video_path):
            print(f"[Runner] WARNING: Không giải mã được {video_path.name} (codec không hỗ trợ, vd AV1). Skipping.")
            return None

        # Tạo thư mục output cho video này
        video_out = pipeline_out / video_stem
        if video_out.exists() and not video_out.is_dir():
            video_out.unlink()
        video_out.mkdir(parents=True, exist_ok=True)
        # Không có statistics.json nghĩa là lần trước chạy dở: xoá keyframe cũ,
        # nếu không Layer 3 sẽ đọc lẫn ảnh của hai lần chạy.
        for stale in video_out.glob("*.jpg"):
            stale.unlink()

        # Track VRAM
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Đo thời gian
        t0 = time.time()
        shot_keyframes_list = extractor.extract(video_path, shots)
        elapsed = time.time() - t0

        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        # Lưu ảnh keyframe và điền image_path
        total_storage_bytes = self._save_keyframe_images(
            video_path, shot_keyframes_list, video_out
        )

        # Ghi keyframes.jsonl
        self._write_keyframes_jsonl(shot_keyframes_list, video_out)

        # Lưu embedding .npy + ids.json (tái dùng vector đã tính lúc Semantic Filter)
        kf_ids, kf_embeddings = self._collect_embeddings(shot_keyframes_list)
        self._save_embeddings(kf_ids, kf_embeddings, video_out, video_stem)

        # Tính stats
        stats = self._compute_statistics(
            extractor_name=extractor.name,
            video_id=video_stem,
            shots=shots,
            shot_keyframes_list=shot_keyframes_list,
            elapsed=elapsed,
            peak_vram_gb=peak_vram_gb,
            total_storage_bytes=total_storage_bytes,
            video_path=video_path,
            embeddings=kf_embeddings,
        )

        # Ghi statistics.json
        with open(video_out / "statistics.json", "w", encoding="utf-8") as f:
            json.dump(stats.__dict__, f, indent=4, ensure_ascii=False)
            
        # Đánh giá Ground Truth (nếu bật --evaluate-keyframes)
        if evaluate and gt_dir:
            # Hỗ trợ cấu trúc gt_dir/Videos_L30/<video_stem>/ hoặc gt_dir/<video_stem>/
            video_gt_dir = self._find_gt_dir(gt_dir, video_stem)
            if video_gt_dir and video_gt_dir.exists():
                print(f"[Runner] Evaluating GT from {video_gt_dir} (τ={eval_threshold}) ...")
                result = self._evaluate_gt(video_out, video_gt_dir, eval_threshold)
                stats.precision          = result.precision
                stats.recall             = result.recall
                stats.f1_score           = result.f1
                stats.tp                 = result.tp
                stats.fp                 = result.fp
                stats.fn                 = result.fn
                stats.avg_matched_similarity = result.avg_matched_sim
                stats.eval_threshold     = eval_threshold
                # Ghi đè lại statistics.json sau khi update
                with open(video_out / "statistics.json", "w", encoding="utf-8") as f:
                    json.dump(stats.__dict__, f, indent=4, ensure_ascii=False, default=str)
                print(
                    f"[Eval] {video_path.name}: "
                    f"TP={result.tp}  FP={result.fp}  FN={result.fn}\n"
                    f"       Precision={result.precision:.3f}  "
                    f"Recall={result.recall:.3f}  F1={result.f1:.3f}\n"
                    f"       AvgMatchedSim={result.avg_matched_sim:.3f}"
                )
            else:
                print(f"[Runner] WARNING: GT dir not found for {video_stem} under {gt_dir}")

        print(
            f"[Runner] {video_path.name}: "
            f"{stats.total_keyframes} KFs, "
            f"{elapsed:.1f}s, "
            f"VRAM={peak_vram_gb:.2f}GB"
        )

        return stats

    # ------------------------------------------------------------------
    # Private: I/O helpers
    # ------------------------------------------------------------------

    def _load_shots(
        self,
        shots_dir: Optional[Path],
        video_stem: str,
        video_path: Path,
    ) -> List[ShotRecord]:
        """
        Đọc shots từ file JSON/JSONL.

        Nếu shots_dir là None (no-shot mode) hoặc không tìm thấy shot file,
        tự động chia video thành các chunk 300 frames để tránh tràn RAM.

        Tìm kiếm theo thứ tự:
          1. shots_dir/<video_stem>/shots.json
          2. shots_dir/<video_stem>/shot.jsonl
          3. shots_dir/<video_stem>.json
          4. shots_dir/<video_stem>.jsonl
        """
        # No-shot mode: shots_dir is None — chunk full video directly
        if shots_dir is None:
            print(f"[Runner] No-shot mode for {video_stem}: chunking full video.")
            return self._chunk_video(video_stem, video_path)

        candidates = [
            shots_dir / video_stem / "shots.json",
            shots_dir / video_stem / "shot.jsonl",
            shots_dir / f"{video_stem}.json",
            shots_dir / f"{video_stem}.jsonl",
        ]

        # Tìm file tồn tại đầu tiên
        shot_file = next((p for p in candidates if p.exists()), None)

        if shot_file is None:
            print(f"[Runner] No shots file found for {video_stem}. Chunking full video to prevent RAM OOM.")
            return self._chunk_video(video_stem, video_path)

        shots: List[ShotRecord] = []
        with open(shot_file, encoding="utf-8") as f:
            # Hỗ trợ cả JSON array và JSONL (mỗi dòng 1 object)
            content = f.read().strip()
            if content.startswith("["):
                records = json.loads(content)
            else:
                records = [json.loads(line) for line in content.splitlines() if line.strip()]

        for i, rec in enumerate(records):
            shot_id = rec.get("shot_id", f"S{i + 1:04d}")
            # Hỗ trợ cả int shot_id và string shot_id
            if isinstance(shot_id, int):
                shot_id = f"S{shot_id:04d}"
            shots.append(ShotRecord(
                video_id=video_stem,
                shot_id=str(shot_id),
                start_frame=int(rec.get("start_frame", 0)),
                end_frame=int(rec.get("end_frame", 0)),
                fps=float(rec.get("fps", 25.0)),
            ))

        return shots

    def _chunk_video(self, video_stem: str, video_path: Path) -> List[ShotRecord]:
        """Chia video thành các chunk 300 frames khi không có shot file."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        shots = []
        chunk_size = 300
        for i in range(0, max(1, total), chunk_size):
            shots.append(ShotRecord(
                video_id=video_stem,
                shot_id=f"S{len(shots) + 1:04d}",
                start_frame=i,
                end_frame=min(i + chunk_size - 1, total - 1),
                fps=fps,
            ))
        return shots

    def _find_gt_dir(self, gt_dir: Path, video_stem: str) -> Optional[Path]:
        """
        Tìm thư mục Ground Truth cho video theo cấu trúc mới.

        Hỗ trợ cả hai cấu trúc:
          1. gt_dir/<video_stem>/           (flat)
          2. gt_dir/<subdir>/<video_stem>/  (nested, e.g. gt_dir/Videos_L30/L30_V001/)
        """
        # Flat lookup first
        flat = gt_dir / video_stem
        if flat.exists():
            return flat
        # Nested: search one level deep
        for subdir in gt_dir.iterdir():
            if subdir.is_dir():
                nested = subdir / video_stem
                if nested.exists():
                    return nested
        return None

    def _can_decode(self, video_path: Path) -> bool:
        """Đọc thử frame đầu để phát hiện sớm codec không hỗ trợ (vd AV1), tránh loop hết mọi shot rồi mới lộ ra rỗng."""
        cap = cv2.VideoCapture(str(video_path))
        ret, _ = cap.read()
        cap.release()
        return ret

    def _collect_embeddings(
        self, shot_keyframes_list: List[ShotKeyframes]
    ) -> Tuple[List[str], List[np.ndarray]]:
        """Gom keyframe_id + embedding theo đúng thứ tự 1-1, xuyên suốt mọi shot của video."""
        ids: List[str] = []
        vectors: List[np.ndarray] = []
        for skf in shot_keyframes_list:
            for kf in skf.keyframes:
                if kf.embedding is not None:
                    ids.append(kf.keyframe_id)
                    vectors.append(kf.embedding)
        return ids, vectors

    def _save_embeddings(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        video_out: Path,
        video_stem: str,
    ) -> None:
        """Lưu {video_stem}.npy (N, D) float32 + {video_stem}_ids.json (list keyframe_id, cùng thứ tự dòng)."""
        if not vectors:
            return
        np.save(video_out / f"{video_stem}.npy", np.stack(vectors).astype(np.float32))
        with open(video_out / f"{video_stem}_ids.json", "w", encoding="utf-8") as f:
            json.dump(ids, f, ensure_ascii=False)

    def _save_keyframe_images(
        self,
        video_path: Path,
        shot_keyframes_list: List[ShotKeyframes],
        video_out: Path,
    ) -> int:
        """
        Lưu ảnh keyframe vào thư mục output và điền image_path.

        Cấu trúc output (phẳng, không lồng theo shot — để Layer 3 quét
        thư mục trực tiếp bằng frame_id = tên file, không cần đọc jsonl):
          video_out/
            <keyframe_id>.jpg    (vd: K01_V001_000000_kf0001.jpg)

        Args:
            video_path          : Đường dẫn video gốc.
            shot_keyframes_list : Danh sách ShotKeyframes (sẽ được update in-place).
            video_out           : Thư mục output của video.

        Returns:
            Tổng dung lượng ảnh đã lưu (bytes).
        """
        total_bytes = 0
        # Pipeline nào đã mã hoá sẵn lúc trích xuất thì ghi thẳng; chỉ mở video khi
        # còn keyframe phải tua lại. Tua chiếm 12% tổng thời gian chạy (~163 ms/ảnh).
        need_seek = [
            kf for skf in shot_keyframes_list for kf in skf.keyframes if kf.image_jpeg is None
        ]

        for skf in shot_keyframes_list:
            for kf in skf.keyframes:
                if kf.image_jpeg is None:
                    continue
                img_filename = f"{kf.keyframe_id}.jpg"
                (video_out / img_filename).write_bytes(kf.image_jpeg)
                kf.image_path = img_filename
                total_bytes += len(kf.image_jpeg)

        if not need_seek:
            return total_bytes

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[Runner] Cannot open {video_path} for frame saving.")
            return total_bytes

        for kf in need_seek:
            cap.set(cv2.CAP_PROP_POS_FRAMES, kf.frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            img_filename = f"{kf.keyframe_id}.jpg"
            img_path = video_out / img_filename
            cv2.imwrite(str(img_path), frame)

            kf.image_path = img_filename
            total_bytes += img_path.stat().st_size if img_path.exists() else 0

        cap.release()
        return total_bytes

    def _write_keyframes_jsonl(
        self,
        shot_keyframes_list: List[ShotKeyframes],
        video_out: Path,
    ) -> None:
        """
        Ghi keyframes.jsonl — mỗi dòng là một keyframe record JSON.

        Format mỗi dòng:
          {"video_id": "...", "shot_id": "...", "keyframe_id": "...", ...}
        """
        jsonl_path = video_out / "keyframes.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for skf in shot_keyframes_list:
                for record in skf.to_records():
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Private: statistics
    # ------------------------------------------------------------------

    def _compute_statistics(
        self,
        extractor_name: str,
        video_id: str,
        shots: List[ShotRecord],
        shot_keyframes_list: List[ShotKeyframes],
        elapsed: float,
        peak_vram_gb: float,
        total_storage_bytes: int,
        video_path: Path,
        embeddings: List[np.ndarray],
    ) -> PipelineStatistics:
        """
        Tính PipelineStatistics từ kết quả extraction.

        Bao gồm: số KF, tốc độ xử lý, VRAM, storage, diversity, coverage.
        """
        total_kf = sum(len(skf.keyframes) for skf in shot_keyframes_list)
        avg_kf_per_shot = total_kf / max(len(shots), 1)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        processing_fps = total_frames / max(elapsed, 1e-6)
        storage_mb = total_storage_bytes / (1024 * 1024)

        # Tính coverage score (trung bình các shot)
        coverage_scores = []
        for shot, skf in zip(shots, shot_keyframes_list):
            kf_indices = [kf.frame_idx for kf in skf.keyframes]
            coverage_scores.append(compute_coverage_score(shot, kf_indices))
        avg_coverage = sum(coverage_scores) / max(len(coverage_scores), 1)

        return PipelineStatistics(
            pipeline=extractor_name,
            video_id=video_id,
            num_shots=len(shots),
            total_keyframes=total_kf,
            avg_keyframes_per_shot=avg_kf_per_shot,
            processing_time_sec=elapsed,
            processing_fps=processing_fps,
            peak_vram_gb=peak_vram_gb,
            total_storage_mb=storage_mb,
            diversity_score=compute_diversity_score(np.stack(embeddings)) if embeddings else 0.0,
            coverage_score=avg_coverage,
        )

    def _append_to_summary_csv(self, all_stats: List[PipelineStatistics]) -> None:
        """
        Append hàng thống kê vào benchmark_summary.csv.
        Tạo header nếu file chưa tồn tại.
        """
        if not all_stats:
            return

        file_exists = self.summary_csv.exists()
        fieldnames = list(all_stats[0].to_dict().keys())

        # Thử lưu file, nếu bị lock bởi Excel (PermissionError) thì báo lỗi và tạo file fallback
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(self.summary_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists and f.tell() == 0:
                        writer.writeheader()
                    for stats in all_stats:
                        writer.writerow(stats.to_dict())
                break  # Thành công thì thoát loop
            except PermissionError:
                if attempt < max_retries - 1:
                    print(f"\n[Runner] CẢNH BÁO: File {self.summary_csv.name} đang bị khóa (có thể do đang mở trong Excel).")
                    print("[Runner] Vui lòng đóng file. Đang thử lại trong 5 giây...")
                    time.sleep(5)
                else:
                    fallback_csv = self.output_dir / f"benchmark_summary_{int(time.time())}.csv"
                    print(f"\n[Runner] LỖI: Vẫn không thể ghi vào {self.summary_csv.name}.")
                    print(f"[Runner] Đang lưu tạm vào: {fallback_csv.name}")
                    with open(fallback_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for stats in all_stats:
                            writer.writerow(stats.to_dict())

    @staticmethod
    def _encode_images_from_paths(
        encoder,
        image_paths: List[Path],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode ảnh từ danh sách đường dẫn theo từng batch nhỏ.

        Thay vì load toàn bộ ảnh vào RAM trước, method này đọc và xử lý
        từng batch riêng biệt để tránh OOM khi có hàng trăm ảnh lớn.

        Args:
            encoder    : MobileNetEncoder đã load().
            image_paths: Danh sách đường dẫn file ảnh.
            batch_size : Số ảnh xử lý mỗi lần.

        Returns:
            numpy array shape (N, D) — embeddings đã L2-normalize.
        """
        all_embeddings: List[np.ndarray] = []
        for batch_start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_start : batch_start + batch_size]
            batch_frames = []
            for local_idx, p in enumerate(batch_paths):
                img = cv2.imread(str(p))
                if img is not None:
                    batch_frames.append((batch_start + local_idx, img))
            if batch_frames:
                _, emb = encoder.encode_batch(batch_frames)
                all_embeddings.append(emb)
        if not all_embeddings:
            return np.empty((0, 960), dtype=np.float32)
        return np.concatenate(all_embeddings, axis=0)

    def _evaluate_gt(
        self,
        pred_dir: Path,
        gt_dir: Path,
        threshold: float = 0.85,
    ):
        """
        Đánh giá Precision/Recall/F1 bằng one-to-one greedy matching.

        Sử dụng MobileNetV3 để encode ảnh nhanh, sau đó áp dụng
        thuật toán matching từ Layer_2_Testing.txt với ngưỡng τ.

        Ảnh được đọc theo từng batch nhỏ (không load toàn bộ vào RAM)
        để tránh OOM với video có nhiều keyframe.

        Returns:
            EvalResult dataclass.
        """
        from src.components.mobilenet_encoder import MobileNetEncoder
        from src.core.metrics import EvalResult

        # --- Collect predicted keyframe paths (exclude embedding .npy / .json files) ---
        pred_paths = sorted(pred_dir.glob("*.jpg"))
        # Filter out unreadable files
        pred_paths = [p for p in pred_paths if p.stat().st_size > 0]

        # --- Collect GT keyframe paths (.jpg or .png) ---
        gt_paths: List[Path] = []
        for ext in ("*.jpg", "*.png"):
            gt_paths.extend(sorted(gt_dir.glob(ext)))
        gt_paths = sorted(gt_paths)
        gt_paths = [p for p in gt_paths if p.stat().st_size > 0]

        if not pred_paths or not gt_paths:
            print(
                f"[Runner] Cannot evaluate: {len(pred_paths)} Pred, {len(gt_paths)} GT images found."
            )
            return EvalResult(
                tp=0, fp=len(pred_paths), fn=len(gt_paths),
                gt_best_similarities=[-1.0] * len(gt_paths),
            )

        print(
            f"[Runner] Encoding {len(pred_paths)} Pred + {len(gt_paths)} GT images "
            f"(MobileNetV3, τ={threshold})..."
        )
        encoder = MobileNetEncoder(batch_size=32)
        encoder.load()
        pred_emb = self._encode_images_from_paths(encoder, pred_paths, batch_size=32)
        gt_emb   = self._encode_images_from_paths(encoder, gt_paths,   batch_size=32)
        encoder.unload()

        return evaluate_with_ground_truth(pred_emb, gt_emb, threshold=threshold)
