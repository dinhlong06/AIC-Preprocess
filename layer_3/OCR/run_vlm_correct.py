"""
run_vlm_correct.py -- CLI entry point, stage 3 (VLM correction for boxes
still merged into a space-less blob after merge_recognizers.py)

Usage
-----
    python run_vlm_correct.py                                # uses default config.yaml
    python run_vlm_correct.py --config my_config.yaml
    python run_vlm_correct.py --frames ./frames --merged in.json --output out.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM correction stage of the OCR pipeline")
    parser.add_argument("--config", default="config.yaml", metavar="PATH")
    parser.add_argument("--frames", default=None, metavar="DIR", help="Override vlm_correct.frames_dir")
    parser.add_argument("--merged", default=None, metavar="PATH", help="Override vlm_correct.merged_input")
    parser.add_argument("--output", default=None, metavar="PATH", help="Override vlm_correct.output_file")
    parser.add_argument("--model-dir", default=None, metavar="PATH", help="Override vlm_correct.model_dir")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[x] Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["vlm_correct"]

    if args.frames:
        cfg["frames_dir"] = args.frames
    if args.merged:
        cfg["merged_input"] = args.merged
    if args.output:
        cfg["output_file"] = args.output
    if args.model_dir:
        cfg["model_dir"] = args.model_dir

    from ocr.pipeline import run_vlm_correct_pipeline
    run_vlm_correct_pipeline(cfg)


if __name__ == "__main__":
    main()
