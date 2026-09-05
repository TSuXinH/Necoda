from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn as nn

import train as base
from model_box.NeRP_parallel_st_entropy import NeRPParallelSTEntropy


def build_parser():
    p = base.build_parser()
    p.set_defaults(epochs=None)
    p.add_argument(
        "--coarse_epochs",
        type=int,
        required=True,
        help="Epochs of 8x local-random downsampled distortion-only training.",
    )
    p.add_argument(
        "--full_epochs",
        type=int,
        required=True,
        help="Epochs of full-resolution entropy/rate-distortion training.",
    )
    return p


def parse_phase_config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--_phase_config", required=True)
    parsed = parser.parse_args()
    with Path(parsed._phase_config).open() as f:
        return argparse.Namespace(**json.load(f))


def make_experiment_dir(args):
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir).resolve()
        if exp_dir.exists() and args.overwrite:
            shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

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


def build_loaders_with_fixed_norm(args):
    with Path(args.fixed_norm_stats_json).open() as f:
        norm_stats = json.load(f)

    dataset_kwargs = dict(
        patch_x=args.patch_x,
        patch_t=args.patch_t,
        interp_x=args.interp_size_x,
        interp_t=args.interp_size_t,
        pre_norm=args.pre_norm,
        norm_stats=norm_stats,
        robust_low=args.robust_low,
        robust_high=args.robust_high,
    )
    train_dataset = base.DatasetTifPatchTrainWithPadding(
        args.data_path,
        gap_x=args.gap_x,
        gap_t=args.gap_t,
        apply_aug=args.apply_augmentation,
        apply_sampling=args.apply_sampling,
        **dataset_kwargs,
    )
    test_dataset = base.DatasetTifPatchTest(args.data_path, **dataset_kwargs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=base.worker_init_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=base.worker_init_fn,
    )

    generalization_loaders = []
    for path in args.generalization_data_path:
        dataset = base.DatasetTifPatchTest(path, **dataset_kwargs)
        generalization_loaders.append(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batchSize,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
                worker_init_fn=base.worker_init_fn,
            )
        )
    return train_loader, test_loader, generalization_loaders, norm_stats


def forward_without_entropy(model, x):
    """Run the shared dual-stream representation while bypassing all codecs."""
    branch_feature = model._encode_to_branch(x)
    spatial_code = model._make_spatial_code(branch_feature)
    temporal_code = model._make_temporal_code(branch_feature)
    latent = model._fuse_to_latent(spatial_code, temporal_code)
    return model.decode_latent(latent), latent


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
    spatial_rate = losses["bpp_loss"].new_zeros(())
    for name, value in losses.items():
        if name.startswith("spatial_") and name.endswith("_bpp"):
            spatial_rate = spatial_rate + value
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


def run_phase(args):

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

    base.seed_everything(args.manualSeed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    device = torch.device("cuda")
    exp_dir = make_experiment_dir(args)
    stop_epoch = args.epochs if args.stop_epoch is None else int(args.stop_epoch)
    if not (0 < stop_epoch <= args.epochs):
        raise ValueError("stop_epoch must be in [1, epochs]")
    eval_epochs = sorted(set(int(v) for v in args.eval_epochs))
    if any(v < 1 or v > args.epochs for v in eval_epochs):
        raise ValueError("eval_epochs must lie in [1, epochs]")

    with (exp_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)
    with (exp_dir / f"config_{args.phase_name}.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    if args.schedule_epochs <= 0:
        raise ValueError("schedule_epochs must be positive")
    if args.schedule_start_epoch < 0:
        raise ValueError("schedule_start_epoch must be nonnegative")

    (
        train_loader,
        test_loader,
        generalization_loaders,
        norm_stats,
    ) = build_loaders_with_fixed_norm(args)

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

    effective_lr = args.lr if args.phase_lr is None else args.phase_lr
    schedule_args = argparse.Namespace(**vars(args))
    schedule_args.lr = effective_lr
    optimizer_config = {
        "net": {"type": "Adam", "lr": effective_lr},
        "aux": {"type": "Adam", "lr": args.aux_lr},
    }
    optimizers = base.net_aux_optimizer(model, optimizer_config)
    optimizer = optimizers["net"]
    aux_optimizer = optimizers["aux"]

    start_epoch = int(args.start_epoch)
    prior_training_seconds = 0.0
    epoch_seconds = {}
    if args.resume_path:
        resume_path = Path(args.resume_path).resolve()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if not args.resume_model_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        start_epoch = int(checkpoint["epoch"])
        prior_training_seconds = float(checkpoint.get("training_seconds", 0.0))
        epoch_seconds = {
            str(k): float(v) for k, v in checkpoint.get("epoch_seconds", {}).items()
        }
        print(
            f"Resumed exact state from {resume_path} at epoch {start_epoch}",
            flush=True,
        )
    if start_epoch >= stop_epoch:
        raise ValueError(
            f"Nothing to train: start_epoch={start_epoch}, stop_epoch={stop_epoch}"
        )

    criterion = base.RateDistortionLoss(
        lam=args.lam,
        metric=args.rd_metric,
        temporal_weight=args.lam_temporal,
        likelihood_floor=args.likelihood_floor,
    )

    training_start = time.perf_counter()

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        epoch_start = time.perf_counter()

        for step, sample in enumerate(train_loader):
            patch_data = sample["patch"].to(device, non_blocking=True)
            patch_gt = sample["target"].to(device, non_blocking=True)
            base.finite_or_raise("patch_data", patch_data)
            base.finite_or_raise("patch_gt", patch_gt)

            optimizer.zero_grad(set_to_none=True)
            aux_optimizer.zero_grad(set_to_none=True)

            phase_epoch = epoch - args.schedule_start_epoch
            if not (0 <= phase_epoch < args.schedule_epochs):
                raise ValueError(
                    f"Epoch {epoch} is outside phase schedule "
                    f"[{args.schedule_start_epoch}, "
                    f"{args.schedule_start_epoch + args.schedule_epochs})"
                )
            cur_epoch = (phase_epoch + step / len(train_loader)) / args.schedule_epochs
            lr = base.adjust_lr(optimizer, cur_epoch, schedule_args)
            aux_lr = base.adjust_lr(aux_optimizer, cur_epoch, schedule_args, "aux_lr")
            rate_weight = 0.0 if args.phase_mode == "coarse" else 1.0

            if args.phase_mode == "coarse":
                patch_out, _ = forward_without_entropy(model, patch_data)
                likelihoods = {}
                codec_stats = {}
            else:
                patch_out, _, likelihoods = model(patch_data)
                codec_stats = model.last_codec_stats
            base.finite_or_raise("patch_out", patch_out)

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

            if args.phase_mode == "entropy":
                aux_loss_tensor = model.aux_loss()
                if not torch.isfinite(aux_loss_tensor):
                    raise FloatingPointError("aux_loss contains NaN/Inf")
                aux_loss_tensor.backward()
                aux_optimizer.step()
                aux_loss_value = float(aux_loss_tensor.detach().item())
            else:
                aux_loss_value = 0.0

            with torch.no_grad():
                psnr_value = float(base.psnr_fn_patch(patch_out, patch_gt).mean())
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
                    aux_loss=aux_loss_value,
                    codec_stats=codec_stats,
                    temporal_std_ratio=temporal_std_ratio,
                )

        this_epoch_seconds = time.perf_counter() - epoch_start
        epoch_seconds[str(epoch + 1)] = float(this_epoch_seconds)
        print(f"Epoch {epoch + 1} time: {this_epoch_seconds:.2f} s")

        should_eval = epoch + 1 in eval_epochs
        if should_eval:
            if args.phase_mode != "entropy":
                raise RuntimeError("Deployable evaluation requires entropy phase")
            model.update()
            raw_model = {"raw_ckt": model.state_dict(), "m_args": args}

            g1_dir = exp_dir / "g1"
            g1_dir.mkdir(parents=True, exist_ok=True)
            torch.save(raw_model, g1_dir / f"raw_model_{epoch + 1}.pth")

            args_payload = {**vars(args), "norm_stats": norm_stats}
            torch.save(args_payload, g1_dir / "args.pth")

            base.evaluate(
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
                base.evaluate(
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
            "training_seconds": float(sum(epoch_seconds.values())),
            "epoch_seconds": epoch_seconds,
        }
        torch.save(checkpoint, exp_dir / "model_latest.pth")

        if epoch + 1 in eval_epochs or epoch + 1 == stop_epoch:
            timing = {
                "phase_name": args.phase_name,
                "epoch": epoch + 1,
                "global_target_epochs": args.epochs,
                "training_seconds": checkpoint["training_seconds"],
                "epoch_seconds": epoch_seconds,
                "evaluated": should_eval,
            }
            with (exp_dir / f"timing_epoch{epoch + 1}.json").open("w") as f:
                json.dump(timing, f, indent=2)

    print(
        f"Invocation completed through epoch {stop_epoch}; cumulative training "
        f"seconds={sum(epoch_seconds.values()):.2f}",
        flush=True,
    )
    print("Results:", exp_dir)


def prepare_full_norm_stats(args, output_path):
    dataset = base.DatasetTifPatchTrainWithPadding(
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
    norm_stats = dict(dataset.norm_stats)
    del dataset
    gc.collect()
    output_path.write_text(json.dumps(norm_stats, indent=2) + "\n")
    return norm_stats


def prepare_8x_local_random(input_path, output_path, seed, chunk_frames=64):
    source_tiff = tifffile.TiffFile(input_path)
    try:
        series = source_tiff.series[0]
        if len(series.shape) != 3 or series.dtype != np.uint16:
            raise ValueError(
                f"Expected uint16 [T,H,W], got {series.shape} {series.dtype}"
            )
        t, h, w = map(int, series.shape)
        if len(source_tiff.pages) != t:
            raise ValueError(
                f"Expected one TIFF page per frame, got "
                f"{len(source_tiff.pages)} for T={t}"
            )
        if t % 2 or h % 2 or w % 2:
            raise ValueError(f"All dimensions must be even, got {series.shape}")
        if chunk_frames <= 0 or chunk_frames % 2:
            raise ValueError("chunk_frames must be a positive even integer")

        output = tifffile.memmap(
            output_path,
            shape=(t // 2, h // 2, w // 2),
            dtype=np.uint16,
            bigtiff=True,
        )
        rng = np.random.default_rng(seed)
        for start in range(0, t, chunk_frames):
            stop = min(start + chunk_frames, t)
            block = np.stack(
                [source_tiff.pages[index].asarray() for index in range(start, stop)],
                axis=0,
            )
            n = block.shape[0]
            hb, wb = h // 2, w // 2

            # Preserve the accepted preparer's RNG stream exactly. Its 8x draw
            # followed the 4x and 2x draws within every source chunk.
            rng.integers(0, 4, size=(n, hb, wb), dtype=np.uint8)
            rng.integers(0, 2, size=(n // 2, h, w), dtype=np.uint8)
            txy_choice = rng.integers(
                0, 8, size=(n // 2, hb, wb), dtype=np.uint8
            )
            txy_blocks = (
                block.reshape(n // 2, 2, hb, 2, wb, 2)
                .transpose(0, 2, 4, 1, 3, 5)
                .reshape(n // 2, hb, wb, 8)
            )
            output[start // 2:stop // 2] = np.take_along_axis(
                txy_blocks, txy_choice[..., None], axis=-1
            )[..., 0]
        output.flush()
        del output
    finally:
        source_tiff.close()


def write_phase_config(path, args, **updates):
    payload = vars(args).copy()
    payload.pop("coarse_epochs", None)
    payload.pop("full_epochs", None)
    payload.update(updates)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run_phase_subprocess(config_path):
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--_phase_config",
            str(config_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parent,
    )


def run_two_stages(args):
    if args.coarse_epochs <= 0 or args.full_epochs <= 0:
        raise ValueError("coarse_epochs and full_epochs must both be positive")
    if args.start_epoch != 0:
        raise ValueError("Two-stage training starts at epoch 0")

    total_epochs = args.coarse_epochs + args.full_epochs
    if args.epochs is not None and args.epochs != total_epochs:
        raise ValueError(
            f"--epochs={args.epochs} conflicts with the two-stage total "
            f"{total_epochs}"
        )
    args.epochs = total_epochs

    original_remark = args.remark
    stage_tag = f"2stages_8x{args.coarse_epochs}_full{args.full_epochs}"
    args.remark = f"{original_remark}_{stage_tag}" if original_remark else stage_tag
    args.experiment_dir = ""
    output_root = Path(args.output_path).resolve()
    existing_paths = (
        {path.resolve() for path in output_root.iterdir()}
        if output_root.exists()
        else set()
    )
    exp_dir = make_experiment_dir(args).resolve()
    args.remark = original_remark

    if exp_dir in existing_paths and not args.overwrite:
        raise FileExistsError(f"Refusing to reuse existing run: {exp_dir}")
    model_latest = exp_dir / "model_latest.pth"

    args.data_path = str(Path(args.data_path).resolve())
    args.generalization_data_path = [
        str(Path(path).resolve()) for path in args.generalization_data_path
    ]

    stage_dir = exp_dir / "stage_data"
    stage_dir.mkdir(parents=True, exist_ok=True)
    norm_stats_path = stage_dir / "full_norm_stats.json"
    coarse_data_path = stage_dir / "Fsim_8x_txy2.tiff"
    coarse_config_path = stage_dir / "coarse_config.json"
    full_config_path = stage_dir / "full_config.json"

    prepare_full_norm_stats(args, norm_stats_path)
    prepare_8x_local_random(
        Path(args.data_path).resolve(),
        coarse_data_path,
        seed=args.manualSeed,
    )

    eval_epochs = list(range(args.eval_freq, total_epochs + 1, args.eval_freq))
    eval_epochs = sorted(
        {epoch for epoch in eval_epochs if epoch > args.coarse_epochs}
        | {total_epochs}
    )
    common = dict(
        epochs=total_epochs,
        eval_epochs=eval_epochs,
        fixed_norm_stats_json=str(norm_stats_path),
        experiment_dir=str(exp_dir),
        distortion_only_epochs=0,
        rate_warmup_epochs=0,
        overwrite=False,
    )
    write_phase_config(
        coarse_config_path,
        args,
        **common,
        data_path=str(coarse_data_path),
        stop_epoch=args.coarse_epochs,
        resume_path="",
        phase_name="8x_txy2",
        phase_mode="coarse",
        schedule_start_epoch=0,
        schedule_epochs=args.coarse_epochs,
        phase_lr=None,
        resume_model_only=False,
    )
    write_phase_config(
        full_config_path,
        args,
        **common,
        data_path=str(Path(args.data_path).resolve()),
        stop_epoch=total_epochs,
        resume_path=str(model_latest),
        phase_name="full_entropy",
        phase_mode="entropy",
        schedule_start_epoch=args.coarse_epochs,
        schedule_epochs=args.full_epochs,
        phase_lr=args.lr,
        resume_model_only=True,
    )

    run_phase_subprocess(coarse_config_path)
    run_phase_subprocess(full_config_path)


def main():
    if "--_phase_config" in sys.argv[1:]:
        run_phase(parse_phase_config())
    else:
        run_two_stages(build_parser().parse_args())


if __name__ == "__main__":
    main()
