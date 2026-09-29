from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import (
    DEFAULT_SIGLIP_MODEL_ID,
    DEFAULT_SIGLIP_NUM_WORKERS,
    KeyframePipeline,
    KeyframePipelineError,
    extract_siglip_dataset,
    extract_siglip_from_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="keyframe-pipeline")
    parser.add_argument("--debug", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    for name, with_dir in (("siglip-video", False), ("siglip-dataset", True)):
        sub = commands.add_parser(name)
        sub.add_argument("--keyframe-dir" if not with_dir else "--dataset-root", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--model-id", default=DEFAULT_SIGLIP_MODEL_ID)
        sub.add_argument("--device")
        sub.add_argument("--batch-size", type=int, default=32)
        sub.add_argument("--num-workers", type=int, default=DEFAULT_SIGLIP_NUM_WORKERS)
        sub.add_argument("--overwrite", action="store_true")

    run = commands.add_parser("run")
    run.add_argument("--dataset-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--model-id", default=DEFAULT_SIGLIP_MODEL_ID)
    run.add_argument("--device")
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--num-workers", type=int, default=DEFAULT_SIGLIP_NUM_WORKERS)
    run.add_argument("--overwrite", action="store_true")
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
    elif args.command == "siglip-dataset":
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
    else:
        pipeline = KeyframePipeline(siglip_model_id=args.model_id, siglip_device=args.device)
        result = pipeline.run_siglip_dataset(
            args.dataset_root,
            output_root=args.output_root,
            siglip_batch_size=args.batch_size,
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
