import argparse
import json
import os
from typing import Dict, List

GAP_TOLERANCE_MS = 500


def run_shot_transcript_mapping(
    shots_jsonl_path: str,
    whisper_jsonl_path: str,
    output_jsonl_path: str,
) -> str:
    """
    Input:
        shots_jsonl_path, whisper_jsonl_path: đường dẫn 2 file input theo đúng
        schema đã thống nhất (xem docstring module).
        output_jsonl_path: nơi ghi kết quả.

    Output:
        Trả về output_jsonl_path.
    """
    print("=== Layer 2: Ghép transcript vào shot ===")

    with open(shots_jsonl_path, "r", encoding="utf-8") as f:
        shots_data = [json.loads(line) for line in f]

    with open(whisper_jsonl_path, "r", encoding="utf-8") as f:
        whisper_data = [json.loads(line) for line in f]

    shots_by_video: Dict[str, List[dict]] = {}
    for shot in shots_data:
        vid = shot["video_id"]
        fps = shot["fps"]
        shot["start_ms"] = int((shot["start_frame"] / fps) * 1000)
        shot["end_ms"] = int((shot["end_frame"] / fps) * 1000)
        shots_by_video.setdefault(vid, []).append(shot)

    whisper_by_video: Dict[str, List[dict]] = {}
    for seg in whisper_data:
        whisper_by_video.setdefault(seg["video_id"], []).append(seg)
    for vid in whisper_by_video:
        whisper_by_video[vid].sort(key=lambda x: x["start_ms"])

    shot_transcripts = []
    for video_id, shots in shots_by_video.items():
        segs = whisper_by_video.get(video_id, [])
        for shot in shots:
            shot_start, shot_end = shot["start_ms"], shot["end_ms"]
            overlapping_texts = [
                seg["text"] for seg in segs
                if seg["start_ms"] <= shot_end and seg["end_ms"] >= shot_start
            ]
            # Shot rơi vào khe lặng ngắn giữa 2 câu (VAD cắt sớm, cảnh cắt trễ vài trăm
            # ms) vẫn thuộc về câu kề đó; chỉ nới khi không có gì chồng lấn để shot
            # đã có lời không bị trộn thêm câu bên cạnh.
            overlapping_texts = overlapping_texts or [
                seg["text"] for seg in segs
                if seg["start_ms"] <= shot_end + GAP_TOLERANCE_MS and seg["end_ms"] >= shot_start - GAP_TOLERANCE_MS
            ]
            combined_text = " ".join(overlapping_texts).strip()
            if combined_text:
                shot_transcripts.append({
                    "video_id": video_id,
                    "shot_id": shot["shot_id"],
                    "text": combined_text,
                })

    os.makedirs(os.path.dirname(output_jsonl_path) or ".", exist_ok=True)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for entry in shot_transcripts:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Đã ghi {len(shot_transcripts)} shot có transcript vào {output_jsonl_path}")
    return output_jsonl_path


def main():
    parser = argparse.ArgumentParser(description="Layer 2 (CPU): map transcript vào shot.")
    parser.add_argument("--shots_path", required=True)
    parser.add_argument("--whisper_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    run_shot_transcript_mapping(args.shots_path, args.whisper_path, args.output_path)


if __name__ == "__main__":
    main()
