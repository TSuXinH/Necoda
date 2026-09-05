from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tensorrt_codec import (
    AnalysisRunner,
    ParallelEntropy,
    configure_runtime,
    encode_one,
    load_codec,
    make_dataset,
    make_geometry,
    resolve_engine,
    serialized_stream,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--input-tiff", type=Path, required=True)
    parser.add_argument("--stream-output", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "amp"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--entropy-workers", type=int, required=True)
    parser.add_argument("--loader-workers", type=int, required=True)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT encoding requires CUDA")
    if not args.input_tiff.is_file():
        raise FileNotFoundError(args.input_tiff)
    if args.batch_size < 1 or args.entropy_workers < 1 or args.loader_workers < 0:
        raise ValueError("Invalid worker count")
    report_path = args.report_output or args.stream_output.with_suffix(".json")
    for path in (args.stream_output, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    configure_runtime()
    device = torch.device("cuda")
    codec, _, saved_args, model_path = load_codec(args.experiment, args.epoch, device)
    dataset = make_dataset(args.input_tiff, saved_args, args.batch_size)
    geometry = make_geometry(saved_args)
    if len(dataset) != len(geometry.coordinate_list):
        raise RuntimeError("TIFF patch layout does not match checkpoint geometry")
    engine_path = resolve_engine(
        "analysis",
        args.engine_dir,
        args.precision,
        args.batch_size,
        codec,
        dataset,
        model_path,
    )
    runner = AnalysisRunner(engine_path)
    with ParallelEntropy(codec, args.entropy_workers, "encode"):
        stream, timing = encode_one(
            codec, runner, dataset, args.batch_size, args.loader_workers
        )

    payload = serialized_stream(stream)
    temporary = args.stream_output.with_suffix(args.stream_output.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(args.stream_output)
    raw_bytes = geometry.t * geometry.x * geometry.y * 2
    report = {
        "status": "complete",
        "operation_count": 1,
        "operation": "encode",
        "precision": args.precision,
        "batch_size": args.batch_size,
        "entropy_workers": args.entropy_workers,
        "loader_workers": args.loader_workers,
        "input_tiff": str(args.input_tiff.resolve()),
        "experiment": str(args.experiment.resolve()),
        "epoch": args.epoch,
        "checkpoint": str(model_path),
        "checkpoint_sha256": sha256(model_path),
        "engine": str(engine_path.resolve()),
        "stream": str(args.stream_output.resolve()),
        "stream_sha256": sha256(args.stream_output),
        "serialized_stream_bytes": len(payload),
        "raw_video_bytes": raw_bytes,
        "stream_cr": raw_bytes / timing["entropy_payload_bytes"],
        **timing,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
