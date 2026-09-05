from __future__ import annotations

import hashlib
import io
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import tensorrt as trt
import torch
from compressai.entropy_models.entropy_models import EntropyModel

from auxiliary.aux_dataset import (
    DatasetTifPatchTest,
    create_overlap_patch_info_test,
    get_interp_coord,
)
from recon import (
    build_model,
    nested_byte_count,
    safe_torch_load,
    strip_module_prefix,
    value,
)


STABLE_FP32_OPS = (
    "InstanceNormalization",
    "ReduceMean",
    "ReduceSum",
    "Sqrt",
    "Pow",
    "Div",
    "Exp",
    "Log",
)


def configure_runtime() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_codec(experiment: Path, epoch: int, device: torch.device):
    experiment = experiment.resolve()
    gdir = experiment / "g1"
    model_path = gdir / f"raw_model_{epoch}.pth"
    args_path = gdir / "args.pth"
    if not model_path.is_file() or not args_path.is_file():
        raise FileNotFoundError(
            f"Missing epoch artifacts: model={model_path}, args={args_path}"
        )
    raw_model = safe_torch_load(model_path)
    saved_args = safe_torch_load(args_path)
    if "raw_ckt" not in raw_model or "m_args" not in raw_model:
        raise KeyError(f"{model_path} must contain raw_ckt and m_args")
    codec = build_model(raw_model["m_args"])
    codec.load_state_dict(strip_module_prefix(raw_model["raw_ckt"]), strict=True)
    codec.to(device).eval()
    codec.update()
    return codec, raw_model["m_args"], saved_args, model_path


def make_dataset(input_tiff: Path, saved_args, batch_size: int):
    dataset = DatasetTifPatchTest(
        input_tiff,
        patch_x=int(value(saved_args, "patch_x")),
        patch_t=int(value(saved_args, "patch_t")),
        interp_x=int(value(saved_args, "interp_size_x")),
        interp_t=int(value(saved_args, "interp_size_t")),
        pre_norm=value(saved_args, "pre_norm", "robust_min_max"),
        norm_stats=value(saved_args, "norm_stats"),
        robust_low=float(value(saved_args, "robust_low", 0.001)),
        robust_high=float(value(saved_args, "robust_high", 99.99)),
    )
    if len(dataset) % batch_size:
        raise RuntimeError(
            f"Patch count {len(dataset)} is not divisible by TensorRT batch {batch_size}"
        )
    if value(saved_args, "norm_stats") is None:
        raise RuntimeError("Checkpoint args do not contain normalization statistics")
    return dataset


def make_geometry(saved_args) -> SimpleNamespace:
    required = {}
    for key in ("t", "x", "y", "patch_t", "patch_x", "interp_size_t", "interp_size_x"):
        item = value(saved_args, key)
        if item is None:
            raise KeyError(f"Missing reconstruction geometry field: {key}")
        required[key] = int(item)
    coordinates = create_overlap_patch_info_test(
        required["patch_x"],
        required["x"],
        required["patch_t"],
        required["t"],
        required["y"],
    )
    return SimpleNamespace(
        t=required["t"],
        x=required["x"],
        y=required["y"],
        patch_t=required["patch_t"],
        patch_x=required["patch_x"],
        patch_y=required["patch_x"],
        interp_t=required["interp_size_t"],
        interp_x=required["interp_size_x"],
        interp_y=required["interp_size_x"],
        coordinate_list=coordinates,
        norm_stats=value(saved_args, "norm_stats"),
    )


class AnalysisTransform(torch.nn.Module):
    def __init__(self, codec: torch.nn.Module) -> None:
        super().__init__()
        self.codec = codec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        branch = self.codec._encode_to_branch(x)
        spatial = self.codec._make_spatial_code(branch) * self.codec.spatial_gain()
        temporal = self.codec._make_temporal_code(branch) * self.codec.temporal_gain()
        return spatial, self.codec._flatten_temporal(temporal)


class SynthesisTransform(torch.nn.Module):
    def __init__(self, codec: torch.nn.Module) -> None:
        super().__init__()
        self.codec = codec

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        return self.codec.decode_latent(
            self.codec._fuse_to_latent(spatial, temporal)
        )


def _export_analysis(codec, dataset, batch_size: int, path: Path) -> None:
    sample_shape = tuple(int(v) for v in dataset[0]["patch"].shape)
    sample = torch.empty((batch_size, *sample_shape), device="cuda")
    torch.onnx.export(
        AnalysisTransform(codec).cuda().eval(),
        (sample,),
        path,
        input_names=("video",),
        output_names=("spatial_scaled", "temporal_flat"),
        opset_version=18,
        do_constant_folding=True,
    )


def _export_synthesis(codec, geometry: SimpleNamespace, batch_size: int, path: Path):
    sample = torch.empty(
        (
            batch_size,
            1,
            geometry.patch_t + 2 * geometry.interp_t,
            geometry.patch_x + 2 * geometry.interp_x,
            geometry.patch_y + 2 * geometry.interp_y,
        ),
        device="cuda",
    )
    with torch.inference_mode():
        spatial_scaled, temporal_flat = AnalysisTransform(codec).cuda().eval()(sample)
        spatial = spatial_scaled / codec.spatial_gain()
        temporal = codec._unflatten_temporal(temporal_flat) / codec.temporal_gain()
    torch.onnx.export(
        SynthesisTransform(codec).cuda().eval(),
        (spatial, temporal),
        path,
        input_names=("spatial", "temporal"),
        output_names=("reconstruction",),
        dynamic_axes={
            "spatial": {0: "batch"},
            "temporal": {0: "batch"},
            "reconstruction": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
    )
    return tuple(spatial.shape), tuple(temporal.shape)


def _convert_amp(source: Path, destination: Path) -> None:
    import onnx
    from onnxconverter_common import float16

    model = float16.convert_float_to_float16(
        onnx.load(source),
        keep_io_types=True,
        disable_shape_infer=False,
        op_block_list=list(STABLE_FP32_OPS),
    )
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    profile_shapes: dict[str, tuple[int, ...]] | None = None,
) -> float:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if precision == "amp":
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if profile_shapes is not None:
        profile = builder.create_optimization_profile()
        for name, shape in profile_shapes.items():
            profile.set_shape(name, shape, shape, shape)
        config.add_optimization_profile(profile)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    seconds = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None")
    engine_path.write_bytes(serialized)
    return seconds


def resolve_engine(
    kind: str,
    engine_dir: Path,
    precision: str,
    batch_size: int,
    codec,
    layout,
    model_path: Path,
) -> Path:
    if kind not in ("analysis", "synthesis"):
        raise ValueError(kind)
    if precision not in ("fp32", "amp"):
        raise ValueError(precision)
    engine_dir.mkdir(parents=True, exist_ok=True)
    if batch_size < 1:
        raise ValueError("TensorRT batch size must be positive")
    stem = f"{kind}_{precision}_b{batch_size}"
    engine_path = engine_dir / f"{stem}.engine"
    metadata_path = engine_dir / f"{stem}.json"
    checkpoint_hash = sha256(model_path)
    expected = {
        "kind": kind,
        "precision": precision,
        "batch_size": batch_size,
        "checkpoint": str(model_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
    }
    if engine_path.is_file() or metadata_path.is_file():
        if not engine_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"Incomplete TensorRT engine artifact: {stem}")
        metadata = json.loads(metadata_path.read_text())
        if any(metadata.get(key) != item for key, item in expected.items()):
            raise RuntimeError(f"TensorRT engine provenance mismatch: {metadata_path}")
        if sha256(engine_path) != metadata.get("engine_sha256"):
            raise RuntimeError(f"TensorRT engine checksum mismatch: {engine_path}")
        return engine_path

    fp32_onnx = engine_dir / f"{kind}_fp32_b{batch_size}.onnx"
    profile_shapes = None
    if kind == "analysis":
        _export_analysis(codec, layout, batch_size, fp32_onnx)
    else:
        spatial_shape, temporal_shape = _export_synthesis(
            codec, layout, batch_size, fp32_onnx
        )
        profile_shapes = {"spatial": spatial_shape, "temporal": temporal_shape}
    build_onnx = fp32_onnx
    if precision == "amp":
        build_onnx = engine_dir / f"{kind}_mixed_fp16_b{batch_size}.onnx"
        _convert_amp(fp32_onnx, build_onnx)
    build_seconds = _build_engine(
        build_onnx, engine_path, precision, profile_shapes=profile_shapes
    )
    metadata = expected | {
        "status": "complete",
        "onnx": str(build_onnx),
        "onnx_sha256": sha256(build_onnx),
        "engine_sha256": sha256(engine_path),
        "build_seconds": build_seconds,
        "tensorrt_version": trt.__version__,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    return engine_path


class AnalysisRunner:
    def __init__(self, engine_path: Path) -> None:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT analysis engine")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.input_shape = tuple(self.engine.get_tensor_shape("video"))
        if self.input_shape[0] < 1:
            raise RuntimeError(f"Analysis engine batch is unresolved: {self.input_shape}")

    def __call__(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        video = video.contiguous()
        if tuple(video.shape) != self.input_shape:
            raise RuntimeError(f"Unexpected analysis input shape: {tuple(video.shape)}")
        spatial = torch.empty(
            tuple(self.context.get_tensor_shape("spatial_scaled")),
            device="cuda", dtype=torch.float32,
        )
        temporal = torch.empty(
            tuple(self.context.get_tensor_shape("temporal_flat")),
            device="cuda", dtype=torch.float32,
        )
        self.context.set_tensor_address("video", video.data_ptr())
        self.context.set_tensor_address("spatial_scaled", spatial.data_ptr())
        self.context.set_tensor_address("temporal_flat", temporal.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT analysis execution failed")
        return spatial, temporal


class SynthesisRunner:
    def __init__(self, engine_path: Path, batch_size: int) -> None:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT synthesis engine")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        dtype_map = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.input_dtype = dtype_map[self.engine.get_tensor_dtype("spatial")]
        self.output_dtype = dtype_map[self.engine.get_tensor_dtype("reconstruction")]
        self.batch_size = batch_size
        self._input_refs = None

    def __call__(self, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        spatial = spatial.to(dtype=self.input_dtype).contiguous()
        temporal = temporal.to(dtype=self.input_dtype).contiguous()
        if spatial.shape[0] != self.batch_size or temporal.shape[0] != self.batch_size:
            raise RuntimeError(
                f"TensorRT synthesis requires batch {self.batch_size}"
            )
        self._input_refs = (spatial, temporal)
        self.context.set_input_shape("spatial", tuple(spatial.shape))
        self.context.set_input_shape("temporal", tuple(temporal.shape))
        output_shape = tuple(self.context.get_tensor_shape("reconstruction"))
        if any(dim <= 0 for dim in output_shape):
            raise RuntimeError(f"Unresolved TensorRT output shape: {output_shape}")
        output = torch.empty(
            output_shape,
            device="cuda", dtype=self.output_dtype,
        )
        self.context.set_tensor_address("spatial", spatial.data_ptr())
        self.context.set_tensor_address("temporal", temporal.data_ptr())
        self.context.set_tensor_address("reconstruction", output.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT synthesis execution failed")
        return output


_POOL = None
_WORKER_STATE = None
_ORIGINAL_COMPRESS = EntropyModel.compress
_ORIGINAL_DECOMPRESS = EntropyModel.decompress


def _init_entropy_worker(specs: dict[int, dict[str, Any]], operation: str) -> None:
    global _WORKER_STATE
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from compressai import ans

    _WORKER_STATE = {}
    for coder_id, spec in specs.items():
        coder = ans.RansEncoder() if operation == "encode" else ans.RansDecoder()
        _WORKER_STATE[coder_id] = {"coder": coder, **spec}


def _encode_sample(task):
    coder_id, symbols, indexes = task
    state = _WORKER_STATE[coder_id]
    return state["coder"].encode_with_indexes(
        symbols.reshape(-1).astype(np.int32, copy=False).tolist(),
        indexes.reshape(-1).astype(np.int32, copy=False).tolist(),
        state["cdf"], state["lengths"], state["offsets"],
    )


def _decode_sample(task):
    coder_id, encoded, indexes = task
    state = _WORKER_STATE[coder_id]
    values = state["coder"].decode_with_indexes(
        encoded,
        indexes.reshape(-1).astype(np.int32, copy=False).tolist(),
        state["cdf"], state["lengths"], state["offsets"],
    )
    return np.asarray(values, dtype=np.int32).reshape(indexes.shape)


def _parallel_compress(self, inputs, indexes, means=None):
    symbols = self.quantize(inputs, "symbols", means)
    if len(inputs.size()) < 2 or inputs.size() != indexes.size():
        raise ValueError("Invalid entropy input/index shape")
    self._check_cdf_size()
    self._check_cdf_length()
    self._check_offsets_size()
    symbols_np = symbols.detach().to("cpu", torch.int32).numpy()
    indexes_np = indexes.detach().to("cpu", torch.int32).numpy()
    tasks = [
        (self._parallel_coder_id, symbols_np[i], indexes_np[i])
        for i in range(symbols_np.shape[0])
    ]
    return list(_POOL.map(_encode_sample, tasks))


def _parallel_decompress(self, strings, indexes, dtype=torch.float, means=None):
    if not isinstance(strings, (tuple, list)) or len(strings) != indexes.size(0):
        raise ValueError("Invalid entropy strings/indexes")
    self._check_cdf_size()
    self._check_cdf_length()
    self._check_offsets_size()
    indexes_np = indexes.detach().to("cpu", torch.int32).numpy()
    tasks = [
        (self._parallel_coder_id, strings[i], indexes_np[i])
        for i in range(indexes_np.shape[0])
    ]
    decoded = np.stack(list(_POOL.map(_decode_sample, tasks)), axis=0)
    outputs = torch.from_numpy(decoded).to(indexes.device)
    return self.dequantize(outputs, means, dtype)


class ParallelEntropy(AbstractContextManager):

    def __init__(self, codec, workers: int, operation: str) -> None:
        self.codec = codec
        self.workers = workers
        self.operation = operation

    def __enter__(self):
        global _POOL
        if self.operation not in ("encode", "decode") or _POOL is not None:
            raise RuntimeError("Invalid parallel entropy lifecycle")
        specs = {}
        for coder_id, module in enumerate(
            item for item in self.codec.modules() if isinstance(item, EntropyModel)
        ):
            module._parallel_coder_id = coder_id
            specs[coder_id] = {
                "cdf": module._quantized_cdf.tolist(),
                "lengths": module._cdf_length.reshape(-1).int().tolist(),
                "offsets": module._offset.reshape(-1).int().tolist(),
            }
        if not specs:
            raise RuntimeError("No entropy models found")
        _POOL = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_entropy_worker,
            initargs=(specs, self.operation),
        )
        list(_POOL.map(time.sleep, [0.0] * self.workers))
        if self.operation == "encode":
            EntropyModel.compress = _parallel_compress
        else:
            EntropyModel.decompress = _parallel_decompress
        return len(specs)

    def __exit__(self, exc_type, exc_value, traceback):
        global _POOL
        EntropyModel.compress = _ORIGINAL_COMPRESS
        EntropyModel.decompress = _ORIGINAL_DECOMPRESS
        if _POOL is not None:
            _POOL.shutdown(wait=True)
            _POOL = None
        return False


def make_loader(dataset, batch_size: int, workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=False,
    )


def encode_one(codec, runner: AnalysisRunner, dataset, batch_size: int, loader_workers: int):
    encoded = []
    patch_count = 0
    all_finite = torch.ones((), dtype=torch.bool, device="cuda")
    loader = make_loader(dataset, batch_size, loader_workers)
    runner.stream.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.cuda.stream(runner.stream):
        for sample in loader:
            patch = sample["patch"].cuda(non_blocking=True)
            spatial_scaled, temporal_flat = runner(patch)
            all_finite.logical_and_(torch.isfinite(spatial_scaled).all())
            all_finite.logical_and_(torch.isfinite(temporal_flat).all())
            spatial = codec.spatial_codec.compress(spatial_scaled)
            temporal = codec.temporal_bottleneck.compress(temporal_flat)
            encoded.append({
                "strings": {"spatial": spatial["strings"], "temporal": temporal},
                "shape": {
                    "spatial": spatial["shape"],
                    "temporal": tuple(int(v) for v in temporal_flat.shape[-2:]),
                },
            })
            patch_count += int(patch.shape[0])
    runner.stream.synchronize()
    seconds = time.perf_counter() - started
    if patch_count != len(dataset) or not bool(all_finite):
        raise RuntimeError("Incomplete or non-finite TensorRT encoding")
    return encoded, {
        "seconds": seconds,
        "patches": patch_count,
        "entropy_payload_bytes": int(sum(nested_byte_count(x["strings"]) for x in encoded)),
    }


def serialized_stream(stream: list[dict]) -> bytes:
    buffer = io.BytesIO()
    torch.save(stream, buffer)
    return buffer.getvalue()


def merge_streams(source: list[dict], target_batch: int) -> list[dict]:
    if not source:
        raise ValueError("Empty entropy stream")
    native_batch = len(source[0]["strings"]["temporal"])
    if native_batch < 1 or any(
        len(item["strings"]["temporal"]) != native_batch for item in source
    ):
        raise ValueError("Entropy stream has inconsistent native batch sizes")
    if native_batch == target_batch:
        return source
    if target_batch % native_batch:
        raise ValueError(f"Cannot regroup batch {native_batch} into {target_batch}")
    group = target_batch // native_batch
    if len(source) % group:
        raise ValueError("Stream batch count cannot be regrouped exactly")
    merged = []
    for start in range(0, len(source), group):
        items = source[start:start + group]
        first = items[0]
        if any(item["shape"] != first["shape"] for item in items[1:]):
            raise RuntimeError("Cannot merge entropy items with different shapes")
        spatial_count = len(first["strings"]["spatial"])
        merged.append({
            "strings": {
                "spatial": [
                    sum((item["strings"]["spatial"][index] for item in items), [])
                    for index in range(spatial_count)
                ],
                "temporal": sum((item["strings"]["temporal"] for item in items), []),
            },
            "shape": first["shape"],
        })
    return merged


def make_axis_weights(geometry: SimpleNamespace) -> torch.Tensor:
    axes = []
    specs = (
        (0, geometry.t, geometry.patch_t, geometry.interp_t),
        (1, geometry.x, geometry.patch_x, geometry.interp_x),
        (2, geometry.y, geometry.patch_y, geometry.interp_y),
    )
    for index, total, patch, interp in specs:
        counts = np.zeros(total, dtype=np.float32)
        for start in sorted({int(c[index]) for c in geometry.coordinate_list}):
            lo, hi = get_interp_coord(start, total, patch, interp)
            counts[lo:hi] += 1.0
        if np.any(counts == 0):
            raise RuntimeError("Overlap weights do not cover the output")
        axes.append(torch.from_numpy(counts).cuda())
    expected_visits = int(
        np.sum(axes[0].cpu().numpy())
        * np.sum(axes[1].cpu().numpy())
        * np.sum(axes[2].cpu().numpy())
    )
    decoded_voxels = len(geometry.coordinate_list) * (
        geometry.patch_t + 2 * geometry.interp_t
    ) * (geometry.patch_x + 2 * geometry.interp_x) * (
        geometry.patch_y + 2 * geometry.interp_y
    )
    if expected_visits != decoded_voxels:
        raise RuntimeError(
            f"Separable overlap accounting mismatch: {expected_visits} != {decoded_voxels}"
        )
    weights = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    if tuple(weights.shape) != (geometry.t, geometry.x, geometry.y):
        raise RuntimeError("Invalid full overlap-weight shape")
    return weights


def decode_one(
    codec, runner: SynthesisRunner, source: list[dict], geometry, batch_size: int
):
    stream = merge_streams(source, batch_size)
    stream_patches = sum(len(item["strings"]["temporal"]) for item in stream)
    if stream_patches != len(geometry.coordinate_list):
        raise RuntimeError(
            f"Stream/geometry patch mismatch: {stream_patches} != "
            f"{len(geometry.coordinate_list)}"
        )
    weights = make_axis_weights(geometry)
    expected_shape = (geometry.t, geometry.x, geometry.y)
    runner.stream.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.cuda.stream(runner.stream):
        reconstruction = torch.zeros(expected_shape, dtype=torch.float32, device="cuda")
        cursor = 0
        for item in stream:
            spatial = codec.spatial_codec.decompress(
                item["strings"]["spatial"], item["shape"]["spatial"]
            )["y_hat"] / codec.spatial_gain()
            temporal_flat = codec.temporal_bottleneck.decompress(
                item["strings"]["temporal"], tuple(item["shape"]["temporal"])
            )
            temporal = codec._unflatten_temporal(temporal_flat) / codec.temporal_gain()
            decoded = runner(spatial.contiguous(), temporal.contiguous())[:, 0]
            for patch in decoded:
                t0, x0, y0 = geometry.coordinate_list[cursor]
                st, et = get_interp_coord(t0, geometry.t, geometry.patch_t, geometry.interp_t)
                sx, ex = get_interp_coord(x0, geometry.x, geometry.patch_x, geometry.interp_x)
                sy, ey = get_interp_coord(y0, geometry.y, geometry.patch_y, geometry.interp_y)
                reconstruction[st:et, sx:ex, sy:ey].add_(patch)
                cursor += 1
        if cursor != len(geometry.coordinate_list):
            raise RuntimeError("Incomplete TensorRT patch reconstruction")
        reconstruction.div_(weights)
        runner.stream.synchronize()
        if not bool(torch.isfinite(reconstruction).all()):
            raise FloatingPointError("TensorRT reconstruction contains NaN/Inf")
        stats = geometry.norm_stats
        if stats["mode"] not in ("min_max", "robust_min_max"):
            raise RuntimeError(f"Unsupported normalization mode: {stats['mode']}")
        lo = float(stats["min"])
        scale = float(stats["max"]) - lo
        output_gpu = reconstruction.clamp(0.0, 1.0).mul(scale).add(lo)
        output_gpu.clamp_(0.0, 65535.0)
        output_gpu = output_gpu.to(torch.uint16)
        runner.stream.synchronize()
    seconds = time.perf_counter() - started
    output = output_gpu.cpu().numpy()
    if output.shape != expected_shape or output.dtype != np.uint16:
        raise RuntimeError("Invalid TensorRT reconstruction output")
    return output, {"seconds": seconds, "patches": cursor}
