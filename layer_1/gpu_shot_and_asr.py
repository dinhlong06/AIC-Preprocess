import argparse
import gc
import json
import os
import subprocess
import tempfile
import wave
from typing import List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

_BASE_PATH = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")

VIDEO_EXTENSIONS = (".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv", ".MKV")

# Silero VAD (Voice Activity Detection) - dùng để cắt audio theo đoạn có tiếng nói
# thay vì cắt cố định theo độ dài, tránh cắt giữa câu và tránh đưa đoạn im lặng vào ASR.
VAD_SAMPLE_RATE = 16000            # phải khớp với -ar của extract_audio_to_wav
VAD_THRESHOLD = 0.5                # ngưỡng xác suất "có tiếng nói" của Silero VAD
VAD_MAX_SPEECH_DURATION_S = 28.0   # giới hạn 1 đoạn VAD, chừa buffer dưới cửa sổ 30s của Whisper
VAD_MIN_SILENCE_DURATION_MS = 300  # khoảng lặng tối thiểu để tách 2 đoạn nói liền nhau
VAD_SPEECH_PAD_MS = 100            # đệm 2 đầu mỗi đoạn để không cụt âm đầu/cuối từ


# --------------------------------------------------------------------------- #
# Helper dùng chung
# --------------------------------------------------------------------------- #

def list_videos(input_dir: str) -> List[str]:
    """Liệt kê toàn bộ video hợp lệ trong 1 thư mục, trả về TÊN FILE (không phải path đầy đủ)."""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục input: {input_dir}")
    return sorted(f for f in os.listdir(input_dir) if f.endswith(VIDEO_EXTENSIONS))


def _read_done_video_ids(jsonl_path: str, key: str = "video_id") -> set:
    """Video được coi là xong khi tên nó nằm trong file `<jsonl>.done`.

    KHÔNG suy ra từ dữ liệu: một video bị kill giữa chừng vẫn để lại vài dòng
    trong jsonl, suy từ đó sẽ bỏ qua vĩnh viễn phần còn thiếu mà không báo gì.
    """
    marker = jsonl_path + ".done"
    if not os.path.exists(marker) and os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            seeded = {json.loads(line)[key] for line in f if line.strip()}
        _write_lines_atomic(marker, sorted(seeded))
    if not os.path.exists(marker):
        return set()
    with open(marker, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _claim(claims_dir: Optional[str], video_id: str) -> bool:
    """Giành video giữa nhiều shard chạy song song.

    O_CREAT|O_EXCL là nguyên tử trên NFSv4 nên đúng một shard thắng, không cần lock.
    Coordinator phải gieo sẵn claim cho video đã xong ở BẤT KỲ shard nào, vì mỗi
    shard chỉ đọc file .done của riêng nó — không gieo là làm lại và ghi trùng.
    """
    if not claims_dir:
        return True
    try:
        os.close(os.open(os.path.join(claims_dir, video_id), os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        return False


def _write_lines_atomic(path: str, lines) -> None:
    """Ghi qua .tmp rồi os.replace: đứt điện giữa chừng không mất file cũ."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(f"{line}\n" if not str(line).endswith("\n") else line for line in lines)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _append_video(jsonl_path: str, entries: list, video_id: str) -> None:
    """Ghi trọn kết quả 1 video rồi mới đánh dấu xong, theo đúng thứ tự đó."""
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
        f.flush()
        os.fsync(f.fileno())
    with open(jsonl_path + ".done", "a", encoding="utf-8") as f:
        f.write(video_id + "\n")
        f.flush()
        os.fsync(f.fileno())


def _drop_video_ids(jsonl_path: str, video_ids: set, key: str = "video_id") -> None:
    """Xóa các dòng thuộc video_ids khỏi file jsonl (phục vụ --force chạy lại 1 phần)."""
    if not os.path.exists(jsonl_path):
        return
    with open(jsonl_path, "r", encoding="utf-8") as f:
        kept = [line for line in f if json.loads(line)[key] not in video_ids]
    _write_lines_atomic(jsonl_path, kept)
    marker = jsonl_path + ".done"
    if os.path.exists(marker):
        with open(marker, "r", encoding="utf-8") as f:
            _write_lines_atomic(marker, [v for v in (x.strip() for x in f) if v and v not in video_ids])


def has_audio(video_path: str) -> bool:
    """Video không có luồng audio thì ffmpeg -vn báo "Output file does not contain any stream"."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def extract_audio_to_wav(video_path: str, wav_path: str) -> None:
    """Tách audio từ video sang wav mono 16kHz bằng ffmpeg.

    Bắt buộc phải qua bước này trước khi đưa vào ASR pipeline: thư viện đọc
    audio không tự nhận diện được container video (.mp4/.mkv), chỉ đọc được
    các định dạng audio thuần (wav/flac/mp3).
    """
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg lỗi khi tách audio: {result.stderr[-500:]}")


# --------------------------------------------------------------------------- #
# Layer 1a: TransNetV2 - Shot Detection
# --------------------------------------------------------------------------- #

def run_shot_detection(
    video_paths: List[str],
    output_jsonl_path: str,
    threshold: float = 0.5,
    claims_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Input:
        video_paths: danh sách ĐƯỜNG DẪN ĐẦY ĐỦ tới các file video.
        output_jsonl_path: nơi ghi kết quả (append + resumable).

    Output:
        Trả về chính output_jsonl_path (để hàm gọi biết chỗ lấy kết quả).
        Mỗi dòng jsonl: {"video_id", "shot_id", "start_frame", "end_frame", "fps"}
    """
    from transnetv2 import TransNetV2  # import trễ để tránh load TF nếu --skip_shots

    print("=== Layer 1: TransNetV2 (Shot Detection) ===")
    transnet_model = TransNetV2()
    print("Model TransNetV2 đã load.")

    os.makedirs(os.path.dirname(output_jsonl_path) or ".", exist_ok=True)
    if force:
        _drop_video_ids(output_jsonl_path, {os.path.splitext(os.path.basename(p))[0] for p in video_paths})
    done_video_ids = _read_done_video_ids(output_jsonl_path)
    if done_video_ids:
        print(f"Đã có {len(done_video_ids)} video xử lý từ trước, sẽ bỏ qua: {sorted(done_video_ids)}")

    total_shots_written = 0

    for video_path in tqdm(video_paths, desc="TransNetV2"):
        video_file = os.path.basename(video_path)
        video_id = os.path.splitext(video_file)[0]

        if video_id in done_video_ids or not _claim(claims_dir, video_id):
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [SKIP] Không mở được: {video_file}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0:
            print(f"  [SKIP] FPS không hợp lệ: {video_file}")
            continue

        print(f"  Đang xử lý {video_file}...")

        # ffmpeg 7.0.2 static (dav1d) nhanh hơn ~4,7 lần trên AV1 nhưng chết
        # "Assertion pkt failed" ở vài file; bản 4.2.7 của apt giải mã được.
        # Giải mã AV1/H.264 là normative nên hai bản cho ra pixel giống hệt.
        scenes = None
        for path_prefix in ("", "/usr/bin:"):
            os.environ["PATH"] = path_prefix + _BASE_PATH
            try:
                video_frames, single_frame_predictions, all_frame_predictions = \
                    transnet_model.predict_video(video_path)
                scenes = transnet_model.predictions_to_scenes(single_frame_predictions, threshold=threshold)
                break
            except Exception as e:
                print(f"  [{'apt' if path_prefix else 'static'} ffmpeg] lỗi trên {video_file}: {e}")
            finally:
                try:
                    del video_frames, single_frame_predictions, all_frame_predictions
                except NameError:
                    pass
                gc.collect()
        os.environ["PATH"] = _BASE_PATH
        if scenes is None:
            print(f"  [ERROR] TransNetV2 bó tay với {video_file}")
            continue

        entries = [
            {
                "video_id": video_id,
                "shot_id": f"{video_id}_{shot_idx:06d}",
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "fps": float(fps),
            }
            for shot_idx, (start_frame, end_frame) in enumerate(scenes)
        ]
        _append_video(output_jsonl_path, entries, video_id)
        total_shots_written += len(entries)
        print(f"  -> Đã ghi {len(entries)} shot cho {video_id}")

    print(f"Hoàn tất Shot Detection. Tổng {total_shots_written} shot mới -> {output_jsonl_path}")

    del transnet_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_jsonl_path


# --------------------------------------------------------------------------- #
# Layer 1b: ChunkFormer - ASR
# --------------------------------------------------------------------------- #

def _build_entries(video_id, speech_segments, results):
    entries = []
    for seg_idx, (seg, result) in enumerate(zip(speech_segments, results)):
        text = result["text"].strip()
        if not text:
            continue
        entries.append({
            "video_id": video_id,
            "seg_id": f"{video_id}_{seg_idx:06d}",
            "start_ms": int(round(seg["start"] * 1000)),
            "end_ms": int(round(seg["end"] * 1000)),
            "text": text,
            "confidence": float(result.get("confidence", 0.0)),
        })
    return entries


def _load_chunkformer_model(model_name):
    from chunkformer import ChunkFormerModel
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ChunkFormerModel.from_pretrained(model_name).to(device)
    model.eval()
    return model


def _chunkformer_text(result):
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return " ".join(_chunkformer_text(item) for item in result).strip()


def _build_chunkformer(model_name):
    model = _load_chunkformer_model(model_name)
    print(f"Model {model_name} đã load.")

    def recognize(chunks, batch_size):
        results = []
        for chunk in chunks:
            # ChunkFormer nhận đường dẫn file, không nhận mảng -> WAV tạm trên
            # /dev/shm (RAM) vì out_dir nằm trên NFS.
            handle = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False,
                dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
            handle.close()
            try:
                pcm = (np.clip(chunk["array"], -1.0, 1.0) * 32767.0).astype("<i2")
                with wave.open(handle.name, "wb") as wav_out:
                    wav_out.setnchannels(1)
                    wav_out.setsampwidth(2)
                    wav_out.setframerate(chunk["sampling_rate"])
                    wav_out.writeframes(pcm.tobytes())
                with torch.inference_mode():
                    raw = model.endless_decode(
                        audio_path=handle.name, chunk_size=64,
                        left_context_size=128, right_context_size=128,
                        total_batch_duration=1800, return_timestamps=False)
                    text = _chunkformer_text(raw)
                results.append({"text": text.strip(), "confidence": 0.0})
            finally:
                if os.path.exists(handle.name):
                    os.remove(handle.name)
        return results

    return recognize


def run_asr(
    video_paths: List[str],
    output_jsonl_path: str,
    model_name: str = "khanhld/chunkformer-ctc-large-vie",
    tmp_dir: Optional[str] = None,
    vad_threshold: float = VAD_THRESHOLD,
    vad_max_speech_duration_s: float = VAD_MAX_SPEECH_DURATION_S,
    vad_min_silence_duration_ms: int = VAD_MIN_SILENCE_DURATION_MS,
    vad_speech_pad_ms: int = VAD_SPEECH_PAD_MS,
    claims_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Input:
        video_paths: danh sách ĐƯỜNG DẪN ĐẦY ĐỦ tới các file video (có audio).
        output_jsonl_path: nơi ghi kết quả (append + resumable).
        model_name: model ChunkFormer CTC tiếng Việt; chỉ cần đổi khi thử model khác.
        tmp_dir: thư mục lưu file .wav tạm (mặc định: cùng thư mục output).
        vad_*: tham số Silero VAD (xem hằng VAD_* ở đầu file để biết ý nghĩa).

    Output:
        Trả về output_jsonl_path.
        Mỗi dòng jsonl: {"video_id", "seg_id", "start_ms", "end_ms", "text"}
        (mỗi dòng ứng với 1 đoạn tiếng nói do Silero VAD phát hiện, không phải 1 chunk cố định)
    """
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps  # noqa: E501

    print(f"=== Layer 1: ASR {model_name} + Silero VAD ===")

    vad_model = load_silero_vad()
    print("Model Silero VAD đã load.")

    asr_pipeline = _build_chunkformer(model_name)

    out_dir = os.path.dirname(output_jsonl_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    # out_dir nằm trên NFS: mỗi video ghi rồi đọc lại 33 MB wav qua mạng, 605 video
    # là ~20 GB đi vòng vô ích. /dev/shm là RAM.
    tmp_dir = tmp_dir or ("/dev/shm" if os.path.isdir("/dev/shm") else out_dir)

    if force:
        _drop_video_ids(output_jsonl_path, {os.path.splitext(os.path.basename(p))[0] for p in video_paths})
    done_video_ids = _read_done_video_ids(output_jsonl_path)
    if done_video_ids:
        print(f"Đã có {len(done_video_ids)} video transcribe từ trước, sẽ bỏ qua: {sorted(done_video_ids)}")

    total_segments_written = 0

    for video_path in tqdm(video_paths, desc="ASR chunkformer"):
        video_file = os.path.basename(video_path)
        video_id = os.path.splitext(video_file)[0]

        if video_id in done_video_ids or not _claim(claims_dir, video_id):
            continue

        # Đánh dấu xong với 0 segment: không làm vậy thì video này bị coi là chưa xong mãi,
        # tiến độ ASR không bao giờ đủ và bước merge chờ vô hạn.
        if not has_audio(video_path):
            print(f"  {video_file}: không có audio, bỏ qua ASR")
            _append_video(output_jsonl_path, [], video_id)
            continue

        print(f"  Đang transcribe {video_file}...")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tmp_wav_path = os.path.join(tmp_dir, f"_tmp_{video_id}.wav")
        try:
            extract_audio_to_wav(video_path, tmp_wav_path)
            wav = read_audio(tmp_wav_path, sampling_rate=VAD_SAMPLE_RATE)
        except Exception as e:
            print(f"  [ERROR] Tách/đọc audio lỗi trên {video_file}: {e}")
            continue
        finally:
            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)

        try:
            speech_segments = get_speech_timestamps(
                wav, vad_model,
                sampling_rate=VAD_SAMPLE_RATE,
                threshold=vad_threshold,
                max_speech_duration_s=vad_max_speech_duration_s,
                min_silence_duration_ms=vad_min_silence_duration_ms,
                speech_pad_ms=vad_speech_pad_ms,
                return_seconds=True,
            )
        except Exception as e:
            print(f"  [ERROR] VAD lỗi trên {video_file}: {e}")
            continue

        # Một lần gọi cho cả video thay vì mỗi segment một lần: WhisperFeatureExtractor
        # pad/truncate MỌI input về đúng 30s bất kể batch, và VAD_MAX_SPEECH_DURATION_S=28
        # đảm bảo không segment nào bị cắt -> batch không đổi kết quả.
        chunks = [
            {
                "array": wav[
                    int(s["start"] * VAD_SAMPLE_RATE): int(s["end"] * VAD_SAMPLE_RATE)
                ].numpy(),
                "sampling_rate": VAD_SAMPLE_RATE,
            }
            for s in speech_segments
        ]
        try:
            results = asr_pipeline(chunks) if chunks else []
        except Exception as e:
            print(f"  [ERROR] ASR lỗi trên {video_file}: {e}")
            continue
        if not results and chunks:
            continue

        entries = _build_entries(video_id, speech_segments, results)

        _append_video(output_jsonl_path, entries, video_id)
        total_segments_written += len(entries)
        print(f"  -> Đã ghi {len(entries)} segment cho {video_id}")

        del wav, speech_segments
        gc.collect()

    print(f"Hoàn tất ASR. Tổng {total_segments_written} segment mới -> {output_jsonl_path}")

    del asr_pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_jsonl_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Layer 1 (GPU): TransNetV2 shot detection + ChunkFormer ASR."
    )
    parser.add_argument("--input_dir", required=True, help="Thư mục chứa video đầu vào.")
    parser.add_argument("--output_dir", required=True, help="Thư mục ghi shots.jsonl / whisper.jsonl.")
    parser.add_argument(
        "--videos", default=None,
        help="Danh sách tên file video cụ thể, phân cách bằng dấu phẩy. "
             "Để trống sẽ xử lý toàn bộ video trong --input_dir.",
    )
    parser.add_argument("--model_name", default="khanhld/chunkformer-ctc-large-vie",
                        help="Model ChunkFormer CTC cho tiếng Việt.")
    parser.add_argument("--shot_threshold", type=float, default=0.5,
                         help="Ngưỡng quyết định ranh giới shot của TransNetV2 (mặc định 0.5).")
    parser.add_argument("--vad_threshold", type=float, default=VAD_THRESHOLD,
                         help=f"Ngưỡng xác suất 'có tiếng nói' của Silero VAD (mặc định {VAD_THRESHOLD}).")
    parser.add_argument("--vad_max_speech_duration_s", type=float, default=VAD_MAX_SPEECH_DURATION_S,
                         help=f"Giới hạn độ dài 1 đoạn VAD, giây (mặc định {VAD_MAX_SPEECH_DURATION_S}, giữ <30).")
    parser.add_argument("--vad_min_silence_duration_ms", type=int, default=VAD_MIN_SILENCE_DURATION_MS,
                         help=f"Khoảng lặng tối thiểu để tách 2 đoạn, ms (mặc định {VAD_MIN_SILENCE_DURATION_MS}).")
    parser.add_argument("--vad_speech_pad_ms", type=int, default=VAD_SPEECH_PAD_MS,
                         help=f"Đệm 2 đầu mỗi đoạn VAD, ms (mặc định {VAD_SPEECH_PAD_MS}).")
    parser.add_argument("--claims_dir", default=None,
                        help="Thư mục claim dùng chung khi chạy nhiều shard song song.")
    parser.add_argument("--skip_shots", action="store_true", help="Bỏ qua bước TransNetV2.")
    parser.add_argument("--skip_asr", action="store_true", help="Bỏ qua bước ASR.")
    parser.add_argument("--force", action="store_true",
                         help="Bỏ qua resume: xử lý lại các video đang chọn (xóa dòng cũ của chúng "
                              "trong file output trước khi ghi), không đụng video khác.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.videos:
        video_files = [v.strip() for v in args.videos.split(",") if v.strip()]
    else:
        video_files = list_videos(args.input_dir)

    video_paths = [os.path.join(args.input_dir, v) for v in video_files]
    print(f"Sẽ xử lý {len(video_paths)} video: {video_files}")

    shots_path = os.path.join(args.output_dir, "shots.jsonl")
    whisper_path = os.path.join(args.output_dir, "whisper.jsonl")

    if not args.skip_shots:
        run_shot_detection(video_paths, shots_path, threshold=args.shot_threshold,
                           claims_dir=args.claims_dir, force=args.force)
    else:
        print("Bỏ qua Shot Detection theo --skip_shots.")

    if not args.skip_asr:
        run_asr(
            video_paths, whisper_path,
            model_name=args.model_name,
            vad_threshold=args.vad_threshold,
            vad_max_speech_duration_s=args.vad_max_speech_duration_s,
            vad_min_silence_duration_ms=args.vad_min_silence_duration_ms,
            vad_speech_pad_ms=args.vad_speech_pad_ms,
            claims_dir=args.claims_dir,
            force=args.force,
        )
    else:
        print("Bỏ qua ASR theo --skip_asr.")

    print("\n=== XONG LAYER 1 (GPU) ===")
    print(f"shots.jsonl   -> {shots_path}")
    print(f"whisper.jsonl -> {whisper_path}")
    print("Bước tiếp theo (Layer 2 - map transcript vào shot) chạy trên máy CPU/RAM "
          "bằng file cpu_map_transcript.py, KHÔNG chạy trên máy GPU này.")


if __name__ == "__main__":
    main()
