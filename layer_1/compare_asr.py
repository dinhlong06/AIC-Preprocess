"""So sánh hai file whisper.jsonl sinh bởi hai backend ASR khác nhau.

Báo cáo độ phủ và khối lượng, KHÔNG phải độ chính xác: nhiều segment hơn hay
nhiều ký tự hơn không có nghĩa là đọc đúng hơn. WER cần ground truth có nhãn,
không có trong repo này.
"""

import argparse
import json


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compare(a_path, b_path):
    a, b = _rows(a_path), _rows(b_path)
    va = {r["video_id"] for r in a}
    vb = {r["video_id"] for r in b}
    return {
        "videos": sorted(va | vb),
        "segments_a": len(a),
        "segments_b": len(b),
        "chars_a": sum(len(r["text"].strip()) for r in a),
        "chars_b": sum(len(r["text"].strip()) for r in b),
        "empty_a": sum(1 for r in a if not r["text"].strip()),
        "empty_b": sum(1 for r in b if not r["text"].strip()),
        "only_a": sorted(va - vb),
        "only_b": sorted(vb - va),
        "both": sorted(va & vb),
    }


def main():
    ap = argparse.ArgumentParser(description="So sánh hai whisper.jsonl")
    ap.add_argument("--a", required=True, help="whisper.jsonl của backend A")
    ap.add_argument("--b", required=True, help="whisper.jsonl của backend B")
    args = ap.parse_args()
    report = compare(args.a, args.b)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nBáo cáo độ phủ, không phải độ chính xác. "
          "Đọc kèm việc nghe lại một mẫu segment trước khi kết luận.")


if __name__ == "__main__":
    main()
