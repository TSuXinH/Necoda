from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import tifffile as tif
import torch
import torch.nn as nn
from tqdm import tqdm

from auxiliary.aux_dataset import (
    create_overlap_patch_info_test,
    get_interp_coord,
)
from auxiliary.process_tif import denormalize_to_uint16
from model_box.NeRP_parallel_st_entropy import NeRPParallelSTEntropy


def configure_codec_runtime() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def value(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def nested_byte_count(obj: Any) -> int:
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj)
    if isinstance(obj, dict):
        return sum(nested_byte_count(item) for item in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(nested_byte_count(item) for item in obj)
    return 0


def strip_module_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {
            key[len("module."):]: tensor
            for key, tensor in state_dict.items()
        }
    return state_dict


def build_model(args):
    act = nn.GELU if args.act == "gelu" else nn.ReLU
    return NeRPParallelSTEntropy(
        raw_size_x=args.patch_x,
        raw_size_t=args.patch_t,
        interp_size_x=args.patch_x + 2 * args.interp_size_x,
        interp_size_t=args.patch_t + 2 * args.interp_size_t,
        interp_chn=args.interp_chn,
        pre_s_rate=args.pre_s_rate,
        pre_t_rate=args.pre_t_rate,
        embedding_dim=args.emb_dim,
        s_rate_list=args.s_rate_list,
        t_rate_list=args.t_rate_list,
        chns_list=args.chns_list,
        branch_level=args.branch_level,
        temporal_channels=args.temporal_channels,
        temporal_pool_h=args.temporal_pool_h,
        temporal_pool_w=args.temporal_pool_w,
        spatial_stat_channels=args.spatial_stat_channels,
        latent_gain_init=args.latent_gain_init,
        latent_gain_min=args.latent_gain_min,
        latent_gain_max=args.latent_gain_max,
        checkerboard_kernel=args.checkerboard_kernel,
        act=act,
    )


def normalize_patch_output(decoded: torch.Tensor) -> np.ndarray:
    if not torch.isfinite(decoded).all():
        finite_fraction = torch.isfinite(decoded).float().mean().item()
        raise FloatingPointError(
            "Decoder output contains NaN/Inf; "
            f"finite fraction={finite_fraction:.6f}"
        )

    patches = decoded.detach().float().cpu().numpy()
    if patches.ndim == 5 and patches.shape[1] == 1:
        patches = patches[:, 0]
    elif patches.ndim != 4:
        raise ValueError(
            f"Expected [B,1,T,H,W] or [B,T,H,W], got {patches.shape}"
        )
    return np.clip(patches, 0.0, 1.0)


def load_geometry_and_norm(gdir: Path, model_args):
    args_path = gdir / "args.pth"
    if not args_path.is_file():
        raise FileNotFoundError(args_path)
    saved_args = safe_torch_load(args_path)

    geometry = {}
    for key in (
        "x",
        "y",
        "t",
        "patch_x",
        "patch_t",
        "interp_size_x",
        "interp_size_t",
    ):
        item = value(saved_args, key, value(model_args, key))
        if item is None:
            raise KeyError(f"Missing reconstruction field {key!r} in {args_path}")
        geometry[key] = int(item)

    norm_stats = value(saved_args, "norm_stats")
    if norm_stats is None:
        raise KeyError(f"{args_path} does not contain norm_stats")
    return geometry, norm_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--ckpt-store-dir",
        "--ckpt_store_dir",
        dest="ckpt_store_dir",
        type=Path,
        required=True,
    )
    parser.add_argument("-e", "--epoch", type=int, required=True)
    parser.add_argument(
        "-g",
        "--generalization-id-list",
        "--generalization_id_list",
        "--generalization-id",
        dest="generalization_id_list",
        type=int,
        nargs="+",
        default=[1],
    )
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    configure_codec_runtime()

    exp_dir = args.ckpt_store_dir.resolve()
    if not exp_dir.is_dir():
        raise FileNotFoundError(exp_dir)

    model_path = exp_dir / "g1" / f"raw_model_{args.epoch}.pth"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    device = torch.device(
        "cuda"
        if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    raw_model = safe_torch_load(model_path)
    if "raw_ckt" not in raw_model or "m_args" not in raw_model:
        raise KeyError(f"{model_path} must contain raw_ckt and m_args")

    model_args = raw_model["m_args"]
    model = build_model(model_args)
    model.load_state_dict(
        strip_module_prefix(raw_model["raw_ckt"]),
        strict=True,
    )
    model.to(device)
    model.eval()
    model.update()

    print(f"Experiment: {exp_dir}")
    print(f"Epoch: {args.epoch}")
    print(f"Device: {device}")
    print(f"Model checkpoint: {model_path}")

    for gid in args.generalization_id_list:
        gdir = exp_dir / f"g{gid}"
        compressed_path = gdir / f"c_dict_{args.epoch}.pth"
        if not compressed_path.is_file():
            raise FileNotFoundError(compressed_path)

        compressed_all = safe_torch_load(compressed_path)
        if not isinstance(compressed_all, list):
            raise TypeError(
                f"Expected a list in {compressed_path}, got {type(compressed_all)}"
            )

        geometry, norm_stats = load_geometry_and_norm(gdir, model_args)
        full_t = geometry["t"]
        full_x = geometry["x"]
        full_y = geometry["y"]
        patch_x = geometry["patch_x"]
        patch_t = geometry["patch_t"]
        interp_x = geometry["interp_size_x"]
        interp_t = geometry["interp_size_t"]

        coordinate_list = create_overlap_patch_info_test(
            patch_x,
            full_x,
            patch_t,
            full_t,
            full_y,
        )
        reconstruction = np.zeros(
            (full_t, full_x, full_y),
            dtype=np.float32,
        )
        weight = np.zeros_like(reconstruction)
        patch_cursor = 0
        decode_start = time.perf_counter()

        with torch.inference_mode():
            for compressed in tqdm(
                compressed_all,
                desc=f"Decode g{gid}",
                dynamic_ncols=True,
            ):
                if "strings" not in compressed or "shape" not in compressed:
                    raise KeyError(
                        "Each compressed dictionary must contain strings and shape"
                    )

                patches = normalize_patch_output(
                    model.decompress(
                        compressed["strings"],
                        compressed["shape"],
                    )
                )
                batch_size = patches.shape[0]
                if patch_cursor + batch_size > len(coordinate_list):
                    raise RuntimeError("Decoded patches exceed saved geometry")

                for batch_index in range(batch_size):
                    t_start, x_start, y_start = coordinate_list[
                        patch_cursor + batch_index
                    ]
                    start_t, end_t = get_interp_coord(
                        t_start, full_t, patch_t, interp_t
                    )
                    start_x, end_x = get_interp_coord(
                        x_start, full_x, patch_x, interp_x
                    )
                    start_y, end_y = get_interp_coord(
                        y_start, full_y, patch_x, interp_x
                    )

                    patch = patches[batch_index]
                    expected_shape = (
                        end_t - start_t,
                        end_x - start_x,
                        end_y - start_y,
                    )
                    if patch.shape != expected_shape:
                        raise ValueError(
                            f"Patch {patch_cursor + batch_index} has {patch.shape}; "
                            f"expected {expected_shape}"
                        )

                    reconstruction[
                        start_t:end_t,
                        start_x:end_x,
                        start_y:end_y,
                    ] += patch
                    weight[
                        start_t:end_t,
                        start_x:end_x,
                        start_y:end_y,
                    ] += 1.0

                patch_cursor += batch_size

        if patch_cursor != len(coordinate_list):
            raise RuntimeError(
                f"Decoded {patch_cursor} patches, expected {len(coordinate_list)}"
            )
        if np.any(weight == 0):
            raise RuntimeError(
                f"Uncovered voxels: {np.count_nonzero(weight == 0)}"
            )

        np.divide(
            reconstruction,
            weight,
            out=reconstruction,
            where=weight > 0,
        )
        if not np.isfinite(reconstruction).all():
            raise FloatingPointError("Final reconstruction contains NaN/Inf")

        output = denormalize_to_uint16(reconstruction, norm_stats)
        output_name = args.name or f"recon_{args.epoch}"
        output_path = gdir / f"{output_name}.tif"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_path} exists; use --overwrite or a different --name"
            )
        tif.imwrite(output_path, output)

        stream_bytes = sum(
            nested_byte_count(compressed.get("strings"))
            for compressed in compressed_all
        )
        raw_bytes = full_t * full_x * full_y * 2
        decode_seconds = time.perf_counter() - decode_start

        print(f"g{gid} reconstructed from: {compressed_path}")
        print(f"Saved TIFF: {output_path}")
        print(f"Decode + overlap + denormalize + write: {decode_seconds:.3f} s")
        print(f"Stream bytes: {stream_bytes}")
        if stream_bytes:
            print(f"Stream compression ratio: {raw_bytes / stream_bytes:.3f}x")


if __name__ == "__main__":
    main()
