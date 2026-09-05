from __future__ import annotations

import argparse
import json
from pathlib import Path

import tifffile
import torch

from tensorrt_codec import (
    ParallelEntropy,
    SynthesisRunner,
    configure_runtime,
    decode_one,
    load_codec,
    make_geometry,
    resolve_engine,
    sha256,
)
from recon import safe_torch_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--stream-input", type=Path, required=True)
    parser.add_argument("--output-tiff", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "amp"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--entropy-workers", type=int, required=True)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT decoding requires CUDA")
    if not args.stream_input.is_file():
        raise FileNotFoundError(args.stream_input)
    if args.batch_size < 1 or args.entropy_workers < 1:
        raise ValueError("Invalid entropy worker count")
    report_path = args.report_output or args.output_tiff.with_suffix(".json")
    for path in (args.output_tiff, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    configure_runtime()
    device = torch.device("cuda")
    codec, _, saved_args, model_path = load_codec(args.experiment, args.epoch, device)
    geometry = make_geometry(saved_args)
    stream = safe_torch_load(args.stream_input)
    if not isinstance(stream, list) or not stream:
        raise TypeError("Entropy stream must be a non-empty list")
    engine_path = resolve_engine(
        "synthesis",
        args.engine_dir,
        args.precision,
        args.batch_size,
        codec,
        geometry,
        model_path,
    )
    runner = SynthesisRunner(engine_path, args.batch_size)
    with ParallelEntropy(codec, args.entropy_workers, "decode"):
        output, timing = decode_one(
            codec, runner, stream, geometry, args.batch_size
        )

    temporary = args.output_tiff.with_suffix(args.output_tiff.suffix + ".tmp")
    tifffile.imwrite(temporary, output, bigtiff=True, photometric="minisblack")
    temporary.replace(args.output_tiff)
    raw_bytes = geometry.t * geometry.x * geometry.y * 2
    report = {
        "status": "complete",
        "operation_count": 1,
        "operation": "decode",
        "precision": args.precision,
        "batch_size": args.batch_size,
        "entropy_workers": args.entropy_workers,
        "experiment": str(args.experiment.resolve()),
        "epoch": args.epoch,
        "checkpoint": str(model_path),
        "checkpoint_sha256": sha256(model_path),
        "engine": str(engine_path.resolve()),
        "stream": str(args.stream_input.resolve()),
        "stream_sha256": sha256(args.stream_input),
        "output_tiff": str(args.output_tiff.resolve()),
        "output_tiff_sha256": sha256(args.output_tiff),
        "raw_video_bytes": raw_bytes,
        **timing,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
