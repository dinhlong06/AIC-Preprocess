from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import (
    DEFAULT_SIGLIP_MODEL_ID,
    DEFAULT_SIGLIP_NUM_WORKERS,
    extract_siglip_dataset,
    extract_siglip_from_dir,
    KeyframePipelineError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="keyframe-pipeline")
    parser.add_argument("--debug", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    for name, dest in (("siglip-video", "keyframe_dir"), ("siglip-dataset", "dataset_root")):
        sub = commands.add_parser(name)
        sub.add_argument(f"--{dest.replace('_', '-')}", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--model-id", default=DEFAULT_SIGLIP_MODEL_ID)
        sub.add_argument("--device")
        sub.add_argument("--batch-size", type=int, default=32)
        sub.add_argument("--num-workers", type=int, default=DEFAULT_SIGLIP_NUM_WORKERS)
        sub.add_argument("--overwrite", action="store_true")
    return parser


def _summary(kind: str, videos: int, keyframes: int, failed: int = 0) -> None:
    print(json.dumps({"kind": kind, "videos": videos, "keyframes": keyframes, "failed": failed}))


def _execute(args: argparse.Namespace) -> None:
    if args.command == "siglip-video":
        result = extract_siglip_from_dir(
            args.keyframe_dir,
            model_id=args.model_id,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            overwrite=args.overwrite,
        )
        _summary("siglip", 1, len(result.keyframe_ids))
    else:
        result = extract_siglip_dataset(
            args.dataset_root,
            output_dir=args.output_dir,
            model_id=args.model_id,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            overwrite=args.overwrite,
        )
        _summary("siglip", len(result.results), sum(len(x.keyframe_ids) for x in result.results))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _execute(args)
    except KeyframePipelineError as exc:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
