from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import tifffile as tif
import torch
import torch.nn as nn
from compressai.optimizers import net_aux_optimizer
from tqdm import tqdm

from auxiliary.aux_dataset import (
    DatasetTifPatchTest,
    DatasetTifPatchTrainWithPadding,
    get_interp_coord,
)
from auxiliary.process_tif import denormalize_to_uint16
from model_box.NeRP_parallel_st_entropy import NeRPParallelSTEntropy
from model_box.sub_assembly import RateDistortionLoss
from nerp_utils import adjust_lr, msssim_fn_patch, psnr_fn_patch, worker_init_fn


@contextlib.contextmanager
def codec_export_mode():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    previous = {
        "algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.use_deterministic_algorithms(
            previous["algorithms"], warn_only=previous["warn_only"]
        )
        torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = previous["matmul_allow_tf32"]


def codec_export(function):
    def wrapped(*args, **kwargs):
        with codec_export_mode():
            return function(*args, **kwargs)

    return wrapped



def build_parser():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument(
        "-g",
        "--generalization_data_path",
        type=str,
        nargs="+",
        default=[],
    )
    p.add_argument(
        "--pre_norm",
        default="robust_min_max",
        choices=[
            "min_max",
            "robust_min_max",
            "mean",
            "mean_std",
            "mean_max",
        ],
    )
    p.add_argument("--robust_low", type=float, default=0.001)
    p.add_argument("--robust_high", type=float, default=99.99)

    p.add_argument("--patch_x", type=int, default=128)
    p.add_argument("--patch_t", type=int, default=128)
    p.add_argument("--gap_x", type=int, default=64)
    p.add_argument("--gap_t", type=int, default=64)
    p.add_argument("--interp_size_x", type=int, default=4)
    p.add_argument("--interp_size_t", type=int, default=4)
    p.add_argument("--interp_chn", type=int, default=32)
    p.add_argument("--apply_augmentation", action="store_true")
    p.add_argument("--apply_sampling", action="store_true")

    # Model
    p.add_argument("--pre_s_rate", type=int, default=2)
    p.add_argument("--pre_t_rate", type=int, default=2)
    p.add_argument("--emb_dim", type=int, default=4)
    p.add_argument("--s_rate_list", type=int, nargs="+", required=True)
    p.add_argument("--t_rate_list", type=int, nargs="+", required=True)
    p.add_argument("--chns_list", type=int, nargs="+", required=True)
    p.add_argument(
        "--act",
        default="gelu",
        choices=["gelu", "relu"],
    )
    p.add_argument(
        "--checkerboard_kernel",
        type=int,
        default=5,
        choices=[3, 5],
    )
    p.add_argument("--latent_gain_init", type=float, default=4.0)
    p.add_argument("--latent_gain_min", type=float, default=0.25)
    p.add_argument("--latent_gain_max", type=float, default=64.0)

    # RD training
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument(
        "--rd_metric",
        default="normalized-mse",
        choices=["mse", "normalized-mse", "ms-ssim"],
    )
    p.add_argument(
        "--lam_temporal",
        type=float,
        default=0.0,
        help="Weight of variance-normalized temporal derivative loss.",
    )
    p.add_argument(
        "--distortion_only_epochs",
        type=int,
        default=5,
    )
    p.add_argument(
        "--rate_warmup_epochs",
        type=int,
        default=10,
    )
    p.add_argument("--likelihood_floor", type=float, default=1e-9)

    # Optimizer
    p.add_argument("-e", "--epochs", type=int, default=100)
    p.add_argument("-b", "--batchSize", type=int, default=2)
    p.add_argument("-j", "--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--aux_lr", type=float, default=1e-4)
    p.add_argument(
        "--lr_type",
        type=str,
        default="cosine_0.1_1_0.1",
    )
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--manualSeed", type=int, default=1)
    p.add_argument("--start_epoch", type=int, default=0)

    # Evaluation/output
    p.add_argument("--eval_freq", type=int, default=10)
    p.add_argument("-p", "--print_freq", type=int, default=50)
    p.add_argument("--remark", default="")
    p.add_argument("--overwrite", action="store_true")

    # Backward-compatible accepted arguments.
    p.add_argument("--model_type", default="nerp_base")
    p.add_argument("--norm", default="none")
    p.add_argument("--loss", default="L2")
    p.add_argument("--lam_perceptual", type=float, default=0.0)
    p.add_argument("--lam_spatial", type=float, default=0.0)
    p.add_argument(
        "--selected_perceptual_layer",
        type=int,
        nargs="+",
        default=[],
    )
    p.add_argument("--quant_model_bit", type=int, default=8)
    p.add_argument("--quant_embed_bit", type=int, default=6)

    p.add_argument(
        "--branch_level",
        type=int,
        default=1,
        help=(
            "Number of PassWay encoder blocks before the parallel split. "
            "Default 1 preserves T=32 for the common 128-frame setup, "
            "instead of the original final T=8."
        ),
    )

    p.add_argument(
        "--temporal_channels",
        type=int,
        default=4,
        help="Channels in the sparse temporal token grid.",
    )

    p.add_argument(
        "--temporal_pool_h",
        type=int,
        default=16,
        help="Height of the learned temporal token grid.",
    )

    p.add_argument(
        "--temporal_pool_w",
        type=int,
        default=16,
        help="Width of the learned temporal token grid.",
    )

    p.add_argument(
        "--spatial_stat_channels",
        type=int,
        default=1,
        choices=[1],
        help=(
            "Kept at one so explicit temporal mean and log-std yield exactly "
            "two dense spatial code channels."
        ),
    )

    return p


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def nested_byte_count(obj: Any) -> int:
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj)
    if isinstance(obj, dict):
        return sum(nested_byte_count(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(nested_byte_count(v) for v in obj)
    return 0


def get_base_model(model):
    return model.module if hasattr(model, "module") else model


def rate_weight_at(
    epoch,
    step,
    steps_per_epoch,
    distortion_only_epochs,
    rate_warmup_epochs,
):
    progress_epoch = epoch + step / max(steps_per_epoch, 1)

    if progress_epoch < distortion_only_epochs:
        return 0.0

    if rate_warmup_epochs <= 0:
        return 1.0

    warmup_progress = (
        progress_epoch - distortion_only_epochs
    ) / rate_warmup_epochs

    return float(np.clip(warmup_progress, 0.0, 1.0))


def finite_or_raise(name, tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(
            f"{name} contains NaN/Inf"
        )


def build_loaders(args):
    train_dataset = DatasetTifPatchTrainWithPadding(
        args.data_path,
        patch_x=args.patch_x,
        patch_t=args.patch_t,
        gap_x=args.gap_x,
        gap_t=args.gap_t,
        interp_x=args.interp_size_x,
        interp_t=args.interp_size_t,
        apply_aug=args.apply_augmentation,
        apply_sampling=args.apply_sampling,
        pre_norm=args.pre_norm,
        robust_low=args.robust_low,
        robust_high=args.robust_high,
    )

    # Test and generalization data use the training normalization parameters.
    shared_norm_stats = train_dataset.norm_stats

    test_dataset = DatasetTifPatchTest(
        args.data_path,
        patch_x=args.patch_x,
        patch_t=args.patch_t,
        interp_x=args.interp_size_x,
        interp_t=args.interp_size_t,
        pre_norm=args.pre_norm,
        norm_stats=shared_norm_stats,
        robust_low=args.robust_low,
        robust_high=args.robust_high,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    generalization_loaders = []
    for path in args.generalization_data_path:
        dataset = DatasetTifPatchTest(
            path,
            patch_x=args.patch_x,
            patch_t=args.patch_t,
            interp_x=args.interp_size_x,
            interp_t=args.interp_size_t,
            pre_norm=args.pre_norm,
            norm_stats=shared_norm_stats,
            robust_low=args.robust_low,
            robust_high=args.robust_high,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=worker_init_fn,
        )
        generalization_loaders.append(loader)

    return (
        train_loader,
        test_loader,
        generalization_loaders,
        shared_norm_stats,
    )


@torch.no_grad()
@codec_export
def evaluate(
    model,
    data_loader,
    exp_dir,
    epoch,
    generalization_id,
    device,
):
    base_model = get_base_model(model)
    base_model.eval()
    base_model.update()

    dataset = data_loader.dataset
    gdir = exp_dir / f"g{generalization_id}"
    gdir.mkdir(parents=True, exist_ok=True)

    full_t = dataset.t
    full_x = dataset.x
    full_y = dataset.y

    reconstruction = np.zeros(
        (full_t, full_x, full_y),
        dtype=np.float32,
    )
    weight = np.zeros_like(reconstruction)

    compressed_all = []
    psnr_values = []
    ssim_values = []
    temporal_std_ratios = []
    baseline_psnr_values = []

    start = time.perf_counter()

    for batch_idx, sample in enumerate(
        tqdm(
            data_loader,
            desc=f"Evaluate g{generalization_id}",
            dynamic_ncols=True,
        )
    ):
        patch_data = sample["patch"].to(
            device,
            non_blocking=True,
        )
        patch_ids = sample["patch_id"]

        compressed = base_model.compress(patch_data)
        compressed.pop("y_hat", None)
        compressed_all.append(compressed)

        decoded = base_model.decompress(
            compressed["strings"],
            compressed["shape"],
        )
        finite_or_raise("decoded", decoded)

        psnr_batch = psnr_fn_patch(
            decoded,
            patch_data,
        )
        ssim_batch = msssim_fn_patch(
            decoded,
            patch_data,
        )
        psnr_values.extend(psnr_batch.tolist())
        ssim_values.extend(ssim_batch.tolist())

        pred_tstd = decoded.std(
            dim=2,
            unbiased=False,
        ).mean(dim=(1, 2, 3))
        gt_tstd = patch_data.std(
            dim=2,
            unbiased=False,
        ).mean(dim=(1, 2, 3)).clamp_min(1e-8)
        temporal_std_ratios.extend(
            (pred_tstd / gt_tstd)
            .detach()
            .cpu()
            .tolist()
        )

        baseline = patch_data.mean(
            dim=(2, 3, 4),
            keepdim=True,
        ).expand_as(patch_data)
        baseline_psnr_values.extend(
            psnr_fn_patch(
                baseline,
                patch_data,
            ).tolist()
        )

        patches = decoded.detach().float().cpu().numpy()
        if patches.ndim != 5 or patches.shape[1] != 1:
            raise ValueError(
                f"Expected [B,1,T,H,W], got {patches.shape}"
            )
        patches = patches[:, 0]

        for j in range(patches.shape[0]):
            patch_id = int(patch_ids[j].item())
            t0, x0, y0 = dataset.coordinate_list[patch_id]

            start_t, end_t = get_interp_coord(
                t0,
                full_t,
                dataset.patch_t,
                dataset.interp_t,
            )
            start_x, end_x = get_interp_coord(
                x0,
                full_x,
                dataset.patch_x,
                dataset.interp_x,
            )
            start_y, end_y = get_interp_coord(
                y0,
                full_y,
                dataset.patch_y,
                dataset.interp_y,
            )

            expected = (
                end_t - start_t,
                end_x - start_x,
                end_y - start_y,
            )
            if patches[j].shape != expected:
                raise ValueError(
                    f"Patch {patch_id} has {patches[j].shape}; "
                    f"expected {expected}"
                )

            reconstruction[
                start_t:end_t,
                start_x:end_x,
                start_y:end_y,
            ] += patches[j]
            weight[
                start_t:end_t,
                start_x:end_x,
                start_y:end_y,
            ] += 1.0

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
        raise FloatingPointError(
            "Final reconstruction contains NaN/Inf"
        )

    elapsed = time.perf_counter() - start
    stream_bytes = sum(
        nested_byte_count(item.get("strings"))
        for item in compressed_all
    )
    raw_bytes = full_t * full_x * full_y * 2

    actual_tif = denormalize_to_uint16(
        reconstruction,
        dataset.norm_stats,
    )

    output_tif = gdir / f"recon_{epoch + 1}.tif"
    tif.imwrite(output_tif, actual_tif)

    torch.save(
        compressed_all,
        gdir / f"c_dict_{epoch + 1}.pth",
    )

    result = {
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "constant_baseline_psnr": float(
            np.mean(baseline_psnr_values)
        ),
        "temporal_std_ratio": float(
            np.mean(temporal_std_ratios)
        ),
        "stream_bytes": int(stream_bytes),
        "stream_compression_ratio": (
            float(raw_bytes / stream_bytes)
            if stream_bytes > 0
            else float("inf")
        ),
        "decode_seconds": float(elapsed),
        "decoded_patches_per_second": (
            float(len(dataset) / elapsed)
        ),
        "normalized_min": float(reconstruction.min()),
        "normalized_max": float(reconstruction.max()),
        "normalized_std": float(reconstruction.std()),
        "output_tif": str(output_tif),
    }

    print(
        f"[EVAL g{generalization_id}] "
        f"PSNR={result['psnr']:.3f}, "
        f"constant_baseline_PSNR="
        f"{result['constant_baseline_psnr']:.3f}, "
        f"SSIM={result['ssim']:.5f}, "
        f"temporal_std_ratio="
        f"{result['temporal_std_ratio']:.4f}, "
        f"stream_CR="
        f"{result['stream_compression_ratio']:.2f}x, "
        f"range="
        f"[{result['normalized_min']:.5f}, "
        f"{result['normalized_max']:.5f}], "
        f"std={result['normalized_std']:.5f}"
    )

    with (gdir / f"metrics_{epoch + 1}.json").open("w") as f:
        json.dump(result, f, indent=2)

    return result


def make_experiment_dir(args):
    chns = ",".join(str(x) for x in args.chns_list)
    s_total = int(np.prod(args.s_rate_list))
    t_total = int(np.prod(args.t_rate_list))

    exp_id = (
        "nerp_parallel_st_entropy"
        f"_B{args.batchSize}"
        f"_E{args.epochs}"
        f"_chns{chns}"
        f"_lr{args.lr}"
        f"_{args.pre_norm}"
        f"_s{s_total}t{t_total}"
        f"_lam{args.lam}"
        f"_metric{args.rd_metric}"
        f"_branch{args.branch_level}"
        f"_tc{args.temporal_channels}"
        f"_token{args.temporal_pool_h}x{args.temporal_pool_w}"
        f"_gain{args.latent_gain_init}"
        f"_rw{args.rate_warmup_epochs}"
    )
    if args.apply_augmentation:
        exp_id += "_aug"
    if args.remark:
        exp_id += f"_{args.remark}"

    exp_dir = Path(args.output_path) / exp_id
    if exp_dir.exists() and args.overwrite:
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


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


def print_training_line(
    exp_dir,
    epoch,
    args,
    step,
    total_steps,
    lr,
    aux_lr,
    rate_weight,
    psnr_value,
    losses,
    aux_loss,
    codec_stats,
    temporal_std_ratio,
):
    spatial_rate = sum(
        value
        for name, value in losses.items()
        if name.startswith("spatial_") and name.endswith("_bpp")
    )
    temporal_rate = losses.get("temporal_bpp", losses["bpp_loss"].new_zeros(()))

    parts = [
        f"[{datetime.now():%Y/%m/%d %H:%M:%S}]",
        f"Epoch[{epoch + 1}/{args.epochs}]",
        f"Step[{step + 1}/{total_steps}]",
        f"lr={lr:.2e}",
        f"aux_lr={aux_lr:.2e}",
        f"rate_w={rate_weight:.3f}",
        f"PSNR={psnr_value:.3f}",
        f"total={losses['loss'].detach().item():.6e}",
        f"bpp={losses['bpp_loss'].detach().item():.6e}",
        f"sp_bpp={spatial_rate.detach().item():.6e}",
        f"tm_bpp={temporal_rate.detach().item():.6e}",
        f"mse={losses['mse_loss'].detach().item():.6e}",
        f"nmse={losses['normalized_mse_loss'].detach().item():.6e}",
        f"temp={losses['temporal_loss'].detach().item():.6e}",
        f"aux={aux_loss:.6e}",
        f"tstd_ratio={temporal_std_ratio:.4f}",
    ]

    if codec_stats:
        parts.extend([
            f"sp_std={codec_stats.get('spatial_pre_std', float('nan')):.4e}",
            f"tm_std={codec_stats.get('temporal_pre_std', float('nan')):.4e}",
            f"sp_zero={codec_stats.get('spatial_post_zero_fraction', float('nan')):.4f}",
            f"tm_zero={codec_stats.get('temporal_post_zero_fraction', float('nan')):.4f}",
            (
                "sp_gain="
                f"{codec_stats.get('spatial_gain_min', float('nan')):.3f}"
                "-"
                f"{codec_stats.get('spatial_gain_max', float('nan')):.3f}"
            ),
            (
                "tm_gain="
                f"{codec_stats.get('temporal_gain_min', float('nan')):.3f}"
                "-"
                f"{codec_stats.get('temporal_gain_max', float('nan')):.3f}"
            ),
        ])

    line = " | ".join(parts)
    print(line, flush=True)
    with (exp_dir / "train.log").open("a") as f:
        f.write(line + "\n")


def main():
    args = build_parser().parse_args()

    if args.model_type != "nerp_base":
        print(
            f"Warning: model_type={args.model_type} is ignored; "
            "this script trains NeRPParallelSTEntropy."
        )
    if args.lam_perceptual != 0 or args.lam_spatial != 0:
        print(
            "Warning: lam_perceptual and lam_spatial are compatibility-only "
            "arguments and are not used."
        )

    seed_everything(args.manualSeed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    device = torch.device("cuda")
    exp_dir = make_experiment_dir(args)
    with (exp_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    (
        train_loader,
        test_loader,
        generalization_loaders,
        norm_stats,
    ) = build_loaders(args)

    # Preserve the same checkpoint metadata convention as the base script.
    args.norm_stats = norm_stats
    args.x = test_loader.dataset.x
    args.y = test_loader.dataset.y
    args.t = test_loader.dataset.t
    args.patch_y = args.patch_x
    args.interp_size_y = args.interp_size_x

    model = build_model(args).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Total trainable parameters: {parameter_count / 1e6:.3f} M")

    optimizer_config = {
        "net": {"type": "Adam", "lr": args.lr},
        "aux": {"type": "Adam", "lr": args.aux_lr},
    }
    optimizers = net_aux_optimizer(model, optimizer_config)
    optimizer = optimizers["net"]
    aux_optimizer = optimizers["aux"]

    criterion = RateDistortionLoss(
        lam=args.lam,
        metric=args.rd_metric,
        temporal_weight=args.lam_temporal,
        likelihood_floor=args.likelihood_floor,
    )

    training_start = time.perf_counter()

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        epoch_start = time.perf_counter()

        for step, sample in enumerate(train_loader):
            patch_data = sample["patch"].to(device, non_blocking=True)
            patch_gt = sample["target"].to(device, non_blocking=True)
            finite_or_raise("patch_data", patch_data)
            finite_or_raise("patch_gt", patch_gt)

            optimizer.zero_grad(set_to_none=True)
            aux_optimizer.zero_grad(set_to_none=True)

            cur_epoch = (epoch + step / len(train_loader)) / args.epochs
            lr = adjust_lr(optimizer, cur_epoch, args)
            aux_lr = adjust_lr(aux_optimizer, cur_epoch, args, "aux_lr")
            rate_weight = rate_weight_at(
                epoch,
                step,
                len(train_loader),
                args.distortion_only_epochs,
                args.rate_warmup_epochs,
            )

            patch_out, _, likelihoods = model(patch_data)
            finite_or_raise("patch_out", patch_out)

            losses = criterion(
                {"x_hat": patch_out, "likelihoods": likelihoods},
                patch_gt,
                rate_weight=rate_weight,
            )
            losses["loss"].backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.grad_clip,
                    error_if_nonfinite=True,
                )
            optimizer.step()

            aux_loss_tensor = model.aux_loss()
            if not torch.isfinite(aux_loss_tensor):
                raise FloatingPointError("aux_loss contains NaN/Inf")
            aux_loss_tensor.backward()
            aux_optimizer.step()

            with torch.no_grad():
                psnr_value = float(psnr_fn_patch(patch_out, patch_gt).mean())
                pred_tstd = patch_out.std(dim=2, unbiased=False).mean()
                gt_tstd = patch_gt.std(dim=2, unbiased=False).mean().clamp_min(1e-8)
                temporal_std_ratio = float(pred_tstd / gt_tstd)

            if step % args.print_freq == 0 or step == len(train_loader) - 1:
                print_training_line(
                    exp_dir=exp_dir,
                    epoch=epoch,
                    args=args,
                    step=step,
                    total_steps=len(train_loader),
                    lr=lr,
                    aux_lr=aux_lr,
                    rate_weight=rate_weight,
                    psnr_value=psnr_value,
                    losses=losses,
                    aux_loss=float(aux_loss_tensor.detach().item()),
                    codec_stats=model.last_codec_stats,
                    temporal_std_ratio=temporal_std_ratio,
                )

        print(f"Epoch {epoch + 1} time: {time.perf_counter() - epoch_start:.2f} s")

        should_eval = (epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs
        if should_eval:
            model.update()
            raw_model = {"raw_ckt": model.state_dict(), "m_args": args}

            g1_dir = exp_dir / "g1"
            g1_dir.mkdir(parents=True, exist_ok=True)
            torch.save(raw_model, g1_dir / f"raw_model_{epoch + 1}.pth")

            args_payload = {**vars(args), "norm_stats": norm_stats}
            torch.save(args_payload, g1_dir / "args.pth")

            evaluate(
                model,
                test_loader,
                exp_dir,
                epoch,
                generalization_id=1,
                device=device,
            )

            for idx, loader in enumerate(generalization_loaders, start=2):
                gdir = exp_dir / f"g{idx}"
                gdir.mkdir(parents=True, exist_ok=True)
                torch.save(raw_model, gdir / f"raw_model_{epoch + 1}.pth")
                torch.save(args_payload, gdir / "args.pth")
                evaluate(
                    model,
                    loader,
                    exp_dir,
                    epoch,
                    generalization_id=idx,
                    device=device,
                )
            model.train()

        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "aux_optimizer": aux_optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(checkpoint, exp_dir / "model_latest.pth")

    print(f"Training completed in {time.perf_counter() - training_start:.2f} s")
    print("Results:", exp_dir)


if __name__ == "__main__":
    main()
