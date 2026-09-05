import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim

class RateDistortionLoss(nn.Module):

    def __init__(
        self,
        lam=0.01,
        metric="normalized-mse",
        temporal_weight=0.0,
        likelihood_floor=1e-9,
        return_type="all",
    ):
        super().__init__()

        if metric not in {
            "mse",
            "normalized-mse",
            "ms-ssim",
        }:
            raise NotImplementedError(
                f"{metric} is not implemented."
            )

        self.metric = metric
        self.lam = float(lam)
        self.temporal_weight = float(temporal_weight)
        self.likelihood_floor = float(likelihood_floor)
        self.return_type = return_type

    @staticmethod
    def _safe_key(name):
        return str(name).replace(".", "_").replace("/", "_")

    def forward(
        self,
        output,
        target,
        rate_weight=1.0,
    ):
        if "x_hat" not in output:
            raise KeyError("output must contain 'x_hat'")
        if "likelihoods" not in output:
            raise KeyError("output must contain 'likelihoods'")

        n, c, t, h, w = target.shape
        num_values = n * c * t * h * w

        out = {}
        total_bpp = target.new_zeros(())

        for name, likelihood in output["likelihoods"].items():
            likelihood = likelihood.float().clamp(
                min=self.likelihood_floor,
                max=1.0,
            )

            stream_bpp = (
                -torch.log2(likelihood).sum()
                / float(num_values)
            )

            out[
                f"{self._safe_key(name)}_bpp"
            ] = stream_bpp
            total_bpp = total_bpp + stream_bpp

        out["bpp_loss"] = total_bpp

        prediction = output["x_hat"].float()
        target_fp32 = target.float()

        raw_mse_per_sample = (
            (prediction - target_fp32)
            .pow(2)
            .flatten(1)
            .mean(1)
        )
        out["mse_loss"] = raw_mse_per_sample.mean()

        target_var_per_sample = (
            target_fp32
            .flatten(1)
            .var(dim=1, unbiased=False)
            .clamp_min(1e-8)
        )
        normalized_mse_per_sample = (
            raw_mse_per_sample
            / target_var_per_sample
        )
        out["normalized_mse_loss"] = (
            normalized_mse_per_sample.mean()
        )

        if self.metric == "mse":
            distortion = out["mse_loss"]

        elif self.metric == "normalized-mse":
            distortion = out["normalized_mse_loss"]

        else:
            out["ms_ssim"] = ms_ssim(
                prediction,
                target_fp32,
                data_range=1.0,
                size_average=True,
            )
            out["ms_ssim_loss"] = 1.0 - out["ms_ssim"]
            distortion = out["ms_ssim_loss"]

        if t > 1:
            pred_dt = prediction[:, :, 1:] - prediction[:, :, :-1]
            target_dt = target_fp32[:, :, 1:] - target_fp32[:, :, :-1]

            temporal_mse_per_sample = (
                (pred_dt - target_dt)
                .pow(2)
                .flatten(1)
                .mean(1)
            )
            target_dt_var = (
                target_dt
                .flatten(1)
                .var(dim=1, unbiased=False)
                .clamp_min(1e-8)
            )
            out["temporal_loss"] = (
                temporal_mse_per_sample
                / target_dt_var
            ).mean()
        else:
            out["temporal_loss"] = target.new_zeros(())

        out["distortion_loss"] = (
            distortion
            + self.temporal_weight
            * out["temporal_loss"]
        )

        rate_weight_t = target.new_tensor(
            float(rate_weight)
        )
        out["rate_weight"] = rate_weight_t
        out["loss"] = (
            self.lam * out["distortion_loss"]
            + rate_weight_t * out["bpp_loss"]
        )

        if not torch.isfinite(out["loss"]):
            raise FloatingPointError(
                "RateDistortionLoss produced NaN/Inf. "
                f"bpp={out['bpp_loss'].detach().item()}, "
                f"distortion={out['distortion_loss'].detach().item()}"
            )

        if self.return_type == "all":
            return out

        return out[self.return_type]
