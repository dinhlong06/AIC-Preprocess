"""
run_ocr.py -- CLI entry point, OCR stage 1 (DeepSolo detection + PARSeq-VN
recognition, GPU).

Usage
-----
    python run_ocr.py                                # uses default config.yaml
    python run_ocr.py --config my_config.yaml
    python run_ocr.py --input ./frames --output ./out.json --limit 60

Runs inside the ocr-deepsolo-parseq image (see run.sh) -- needs detectron2,
DeepSolo and strhub plus the weights mounted at /exp.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSolo+PARSeq stage of the OCR pipeline")
    parser.add_argument("--config", default="config.yaml", metavar="PATH")
    parser.add_argument("--input", default=None, metavar="DIR", help="Override ocr.input_dir")
    parser.add_argument("--output", default=None, metavar="FILE", help="Override ocr.output_file")
    parser.add_argument("--limit", type=int, default=None, help="Override ocr.limit")
    parser.add_argument("--claims-dir", default=None, metavar="DIR",
                         help="Worker-pool mode: claim+process one video at a time under DIR, "
                              "skipping videos already claimed/done -- for running several shards "
                              "against the same input concurrently")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[x] Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["ocr"]

    if args.input:
        cfg["input_dir"] = args.input
    if args.output:
        cfg["output_file"] = args.output
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.claims_dir:
        cfg["claims_dir"] = args.claims_dir

    from ocr.pipeline import run_ocr_pipeline
    run_ocr_pipeline(cfg)


if __name__ == "__main__":
    main()
