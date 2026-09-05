from __future__ import annotations

import math

from typing import Dict, Tuple

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from einops import rearrange

from compressai.entropy_models import EntropyBottleneck

from compressai.latent_codecs import (
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
    HyperpriorLatentCodec,
)

from compressai.layers import CheckerboardMaskedConv2d

from compressai.models import SimpleVAECompressionModel

from model_box.gap_module import GapModule

from model_box.pass_way import PassWay, SpatiotemporalCodeLayer

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=3,
        stride=stride,
        padding=1,
    )

def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_ch,
            out_ch * r**2,
            kernel_size=3,
            padding=1,
        ),
        nn.PixelShuffle(r),
    )

def _positive_gain(
    log_gain: torch.Tensor,
    gain_min: float,
    gain_max: float,
) -> torch.Tensor:
    """Same gain parameterization used by the original entropy model."""
    return log_gain.exp().clamp(min=gain_min, max=gain_max)

class NeRPParallelSTEntropy(SimpleVAECompressionModel):
    """Simple, truly parallel spatial-temporal dual-stream NeRP codec.

    ``branch_level`` is the number of PassWay encoder layers executed before
    splitting. With the common configuration

        pre_t_rate = 2, t_rate_list = [2, 2, 2], patch_t = 128,

    branch_level=1 gives T_branch=32, while the original final latent has
    T_final=8. Thus the temporal branch preserves 4x longer temporal support.
    """

    def __init__(
        self,
        raw_size_x: int,
        raw_size_t: int,
        interp_size_x: int,
        interp_size_t: int,
        interp_chn: int,
        pre_s_rate: int,
        pre_t_rate: int,
        embedding_dim: int,
        s_rate_list,
        t_rate_list,
        chns_list,
        branch_level: int = 1,
        temporal_channels: int = 4,
        temporal_pool_h: int = 16,
        temporal_pool_w: int = 16,
        spatial_stat_channels: int = 1,
        latent_gain_init: float = 4.0,
        latent_gain_min: float = 0.25,
        latent_gain_max: float = 64.0,
        checkerboard_kernel: int = 5,
        act=nn.GELU,
    ):
        super().__init__()

        if len(s_rate_list) != len(t_rate_list):
            raise ValueError("s_rate_list and t_rate_list must have equal length")
        if len(chns_list) != len(s_rate_list):
            raise ValueError("chns_list must have one item per PassWay block")
        if not (0 <= int(branch_level) < len(s_rate_list)):
            raise ValueError(
                "branch_level must be in [0, len(s_rate_list)-1] so that "
                "at least one original PassWay encoder block remains after fusion"
            )
        if temporal_channels < 1:
            raise ValueError("temporal_channels must be positive")
        if spatial_stat_channels != 1:
            raise ValueError(
                "This first model intentionally uses one learned statistic "
                "feature followed by mean/log-std, yielding exactly two "
                "spatial code channels. Set spatial_stat_channels=1."
            )
        if temporal_pool_h < 1 or temporal_pool_w < 1:
            raise ValueError("temporal token-grid dimensions must be positive")
        if latent_gain_init <= 0 or latent_gain_min <= 0:
            raise ValueError("latent gains must be positive")
        if latent_gain_max <= latent_gain_min:
            raise ValueError("latent_gain_max must exceed latent_gain_min")
        if checkerboard_kernel not in (3, 5):
            raise ValueError("checkerboard_kernel must be 3 or 5")

        self.embedding_dim = int(embedding_dim)
        self.branch_level = int(branch_level)
        self.temporal_channels = int(temporal_channels)
        self.temporal_pool_h = int(temporal_pool_h)
        self.temporal_pool_w = int(temporal_pool_w)
        self.spatial_code_channels = 2  # [mean, log-std]
        self.latent_gain_min = float(latent_gain_min)
        self.latent_gain_max = float(latent_gain_max)

        self.raw_size_x = int(raw_size_x)
        self.raw_size_t = int(raw_size_t)
        self.pre_s_rate = int(pre_s_rate)
        self.pre_t_rate = int(pre_t_rate)
        self.s_rate_list = tuple(int(v) for v in s_rate_list)
        self.t_rate_list = tuple(int(v) for v in t_rate_list)
        self.chns_list = tuple(int(v) for v in chns_list)

        self._validate_sizes()

        # Exactly the original interpolation front-end and original 3-D trunk.
        self.interp_encoder = GapModule(
            raw_size_x,
            raw_size_t,
            1,
            interp_chn,
            act=act,
        )
        self.interp_decoder = GapModule(
            interp_size_x,
            interp_size_t,
            interp_chn,
            1,
            act=act,
            final_act=False,
        )
        self.st_encoder = SpatiotemporalCodeLayer(
            pre_s_rate,
            pre_t_rate,
            interp_chn,
            self.chns_list[0],
            act=act,
            do_ds=True,
        )
        self.st_decoder = SpatiotemporalCodeLayer(
            pre_s_rate,
            pre_t_rate,
            self.chns_list[0],
            interp_chn,
            act=act,
            do_ds=False,
        )
        self.pass_way = PassWay(
            embedding_dim,
            self.s_rate_list,
            self.t_rate_list,
            self.chns_list,
            act=act,
        )

        # Feature dimensions at the point where both streams split.
        self.branch_channels = self.chns_list[self.branch_level]
        self.branch_t = self.raw_size_t // (
            self.pre_t_rate * int(np.prod(self.t_rate_list[: self.branch_level]))
        )
        self.branch_h = self.raw_size_x // (
            self.pre_s_rate * int(np.prod(self.s_rate_list[: self.branch_level]))
        )
        self.branch_w = self.branch_h

        self.final_t = self.raw_size_t // (
            self.pre_t_rate * int(np.prod(self.t_rate_list))
        )
        self.final_h = self.raw_size_x // (
            self.pre_s_rate * int(np.prod(self.s_rate_list))
        )
        self.final_w = self.final_h

        if self.temporal_pool_h > self.branch_h or self.temporal_pool_w > self.branch_w:
            raise ValueError(
                "temporal token grid cannot exceed the branch spatial size: "
                f"requested {(self.temporal_pool_h, self.temporal_pool_w)}, "
                f"branch {(self.branch_h, self.branch_w)}"
            )

        # ----- Parallel representation heads -----
        # One learned feature is intentionally summarized by explicit mean/std.
        self.stat_projection = nn.Conv3d(
            self.branch_channels,
            spatial_stat_channels,
            kernel_size=1,
        )

        # Temporal feature: preserve branch T, reduce spatial degrees of freedom.
        self.temporal_projection = nn.Sequential(
            nn.Conv3d(
                self.branch_channels,
                self.temporal_channels,
                kernel_size=1,
            ),
            act(),
        )

        # ----- Spatial entropy stream: retain original heavy codec -----
        self.spatial_codec = self._make_spatial_codec(
            n=self.spatial_code_channels,
            checkerboard_kernel=checkerboard_kernel,
            act=act,
        )

        # ----- Temporal entropy stream: deliberately lightweight -----
        self.temporal_flat_channels = self.temporal_channels * self.branch_t
        self.temporal_bottleneck = EntropyBottleneck(self.temporal_flat_channels)

        # Separate learned quantization scales for the two streams.
        init_log_gain = math.log(float(latent_gain_init))
        self.log_spatial_gain = nn.Parameter(
            torch.full(
                (1, self.spatial_code_channels, 1, 1),
                init_log_gain,
                dtype=torch.float32,
            )
        )
        self.log_temporal_gain = nn.Parameter(
            torch.full(
                (1, self.temporal_channels, self.branch_t, 1, 1),
                init_log_gain,
                dtype=torch.float32,
            )
        )

        # Decode both streams into the branch feature, then use the remaining
        # original PassWay encoder blocks to obtain the original final latent.
        fusion_width = max(self.branch_channels, self.temporal_channels + 2)
        self.fusion = nn.Sequential(
            nn.Conv3d(
                self.spatial_code_channels + self.temporal_channels,
                fusion_width,
                kernel_size=3,
                padding=1,
            ),
            act(),
            nn.Conv3d(
                fusion_width,
                self.branch_channels,
                kernel_size=3,
                padding=1,
            ),
            act(),
        )

        self.last_codec_stats: Dict[str, float] = {}

        print(
            "Parallel ST entropy branch: "
            f"level={self.branch_level}, "
            f"feature=[C={self.branch_channels}, T={self.branch_t}, "
            f"H={self.branch_h}, W={self.branch_w}], "
            f"final=[C={self.embedding_dim}, T={self.final_t}, "
            f"H={self.final_h}, W={self.final_w}]"
        )
        print(
            "Spatial code=[2, H, W], "
            f"temporal code=[C={self.temporal_channels}, T={self.branch_t}, "
            f"H={self.temporal_pool_h}, W={self.temporal_pool_w}]"
        )

    def _validate_sizes(self) -> None:
        pre_product_t = self.pre_t_rate
        pre_product_s = self.pre_s_rate
        if self.raw_size_t % pre_product_t != 0:
            raise ValueError("raw_size_t must be divisible by pre_t_rate")
        if self.raw_size_x % pre_product_s != 0:
            raise ValueError("raw_size_x must be divisible by pre_s_rate")

        total_t = pre_product_t * int(np.prod(self.t_rate_list))
        total_s = pre_product_s * int(np.prod(self.s_rate_list))
        if self.raw_size_t % total_t != 0:
            raise ValueError(
                f"raw_size_t={self.raw_size_t} is not divisible by total temporal stride={total_t}"
            )
        if self.raw_size_x % total_s != 0:
            raise ValueError(
                f"raw_size_x={self.raw_size_x} is not divisible by total spatial stride={total_s}"
            )

        for level in range(1, self.branch_level + 1):
            stride_t = self.pre_t_rate * int(np.prod(self.t_rate_list[:level]))
            stride_s = self.pre_s_rate * int(np.prod(self.s_rate_list[:level]))
            if self.raw_size_t % stride_t != 0 or self.raw_size_x % stride_s != 0:
                raise ValueError(
                    "Input sizes must remain integral at the selected branch level. "
                    f"level={level}, temporal_stride={stride_t}, spatial_stride={stride_s}"
                )

    @staticmethod
    def _make_spatial_codec(n: int, checkerboard_kernel: int, act) -> HyperpriorLatentCodec:
        padding = checkerboard_kernel // 2

        h_a = nn.Sequential(
            conv3x3(n, n),
            act(),
            conv3x3(n, n),
            act(),
            conv3x3(n, n, stride=2),
            act(),
            conv3x3(n, n),
            act(),
            conv3x3(n, n, stride=2),
        )
        h_s = nn.Sequential(
            conv3x3(n, n),
            act(),
            subpel_conv3x3(n, n, 2),
            act(),
            conv3x3(n, n * 3 // 2),
            act(),
            subpel_conv3x3(n * 3 // 2, n * 3 // 2, 2),
            act(),
            conv3x3(n * 3 // 2, n * 2),
        )

        return HyperpriorLatentCodec(
            latent_codec={
                "y": CheckerboardLatentCodec(
                    latent_codec={
                        "y": GaussianConditionalLatentCodec(
                            quantizer="ste",
                        ),
                    },
                    entropy_parameters=nn.Sequential(
                        nn.Conv2d(n * 4, n * 10 // 3, 1),
                        act(),
                        nn.Conv2d(n * 10 // 3, n * 8 // 3, 1),
                        act(),
                        nn.Conv2d(n * 8 // 3, n * 2, 1),
                    ),
                    context_prediction=CheckerboardMaskedConv2d(
                        n,
                        2 * n,
                        kernel_size=checkerboard_kernel,
                        stride=1,
                        padding=padding,
                    ),
                ),
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(n),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )

    def spatial_gain(self) -> torch.Tensor:
        return _positive_gain(
            self.log_spatial_gain,
            self.latent_gain_min,
            self.latent_gain_max,
        )

    def temporal_gain(self) -> torch.Tensor:
        return _positive_gain(
            self.log_temporal_gain,
            self.latent_gain_min,
            self.latent_gain_max,
        )

    def _encode_to_branch(self, x: torch.Tensor) -> torch.Tensor:
        x = self.interp_encoder(x)
        x, _ = self.st_encoder(x)

        # Execute the requested initial portion of the original PassWay encoder.
        for idx, layer in enumerate(self.pass_way.encoder):
            if idx >= self.branch_level:
                break
            x, _ = layer(x)

        expected = (
            self.branch_channels,
            self.branch_t,
            self.branch_h,
            self.branch_w,
        )
        if tuple(x.shape[1:]) != expected:
            raise RuntimeError(
                "Unexpected branch feature shape. "
                f"got={tuple(x.shape[1:])}, expected={expected}. "
                "Check GapModule and stride settings."
            )
        return x

    def _make_spatial_code(self, branch_feature: torch.Tensor) -> torch.Tensor:
        """Return exactly [mean_t, logstd_t] at every spatial location."""
        statistic_feature = self.stat_projection(branch_feature)
        mean = statistic_feature.mean(dim=2)
        std = statistic_feature.std(dim=2, unbiased=False).clamp_min(1e-6)
        return torch.cat([mean, torch.log(std)], dim=1)

    def _make_temporal_code(self, branch_feature: torch.Tensor) -> torch.Tensor:
        """Map dense xy feature to a small continuous temporal token grid."""
        temporal_feature = self.temporal_projection(branch_feature)
        return F.adaptive_avg_pool3d(
            temporal_feature,
            output_size=(
                self.branch_t,
                self.temporal_pool_h,
                self.temporal_pool_w,
            ),
        )

    def _flatten_temporal(self, temporal_code: torch.Tensor) -> torch.Tensor:
        expected = (
            self.temporal_channels,
            self.branch_t,
            self.temporal_pool_h,
            self.temporal_pool_w,
        )
        if tuple(temporal_code.shape[1:]) != expected:
            raise RuntimeError(
                f"Unexpected temporal code shape {tuple(temporal_code.shape[1:])}; "
                f"expected {expected}"
            )
        return rearrange(temporal_code, "b c t h w -> b (c t) h w")

    def _unflatten_temporal(self, temporal_flat: torch.Tensor) -> torch.Tensor:
        expected_channels = self.temporal_channels * self.branch_t
        if temporal_flat.ndim != 4 or temporal_flat.shape[1] != expected_channels:
            raise RuntimeError(
                "Unexpected temporal flat shape "
                f"{tuple(temporal_flat.shape)}; expected channel={expected_channels}"
            )
        return rearrange(
            temporal_flat,
            "b (c t) h w -> b c t h w",
            c=self.temporal_channels,
            t=self.branch_t,
        )

    def _fuse_to_latent(
        self,
        spatial_hat: torch.Tensor,
        temporal_hat: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(spatial_hat.shape[1:]) != (
            self.spatial_code_channels,
            self.branch_h,
            self.branch_w,
        ):
            raise RuntimeError(
                f"Unexpected decoded spatial code shape {tuple(spatial_hat.shape)}"
            )

        dense_temporal = F.interpolate(
            temporal_hat,
            size=(self.branch_t, self.branch_h, self.branch_w),
            mode="trilinear",
            align_corners=False,
        )
        spatial_broadcast = spatial_hat.unsqueeze(2).expand(
            -1,
            -1,
            self.branch_t,
            -1,
            -1,
        )
        fused = self.fusion(torch.cat([spatial_broadcast, dense_temporal], dim=1))

        # The remaining original encoder blocks synthesize the exact original
        # final-latent interface expected by the unchanged NeRP decoder.
        latent = fused
        for idx, layer in enumerate(self.pass_way.encoder):
            if idx < self.branch_level:
                continue
            latent, _ = layer(latent)

        expected = (
            self.embedding_dim,
            self.final_t,
            self.final_h,
            self.final_w,
        )
        if tuple(latent.shape[1:]) != expected:
            raise RuntimeError(
                "Unexpected fused final latent shape. "
                f"got={tuple(latent.shape[1:])}, expected={expected}"
            )
        return latent

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        x_hat = latent
        for layer in self.pass_way.decoder:
            x_hat, _ = layer(x_hat)
        x_hat, _ = self.st_decoder(x_hat)
        return self.interp_decoder(x_hat)

    @torch.no_grad()
    def _record_codec_stats(
        self,
        spatial_pre: torch.Tensor,
        spatial_post: torch.Tensor,
        temporal_pre: torch.Tensor,
        temporal_post: torch.Tensor,
    ) -> None:
        self.last_codec_stats = {
            "spatial_gain_min": float(self.spatial_gain().min().item()),
            "spatial_gain_max": float(self.spatial_gain().max().item()),
            "temporal_gain_min": float(self.temporal_gain().min().item()),
            "temporal_gain_max": float(self.temporal_gain().max().item()),
            "spatial_pre_std": float(spatial_pre.std().item()),
            "spatial_post_zero_fraction": float((spatial_post == 0).float().mean().item()),
            "temporal_pre_std": float(temporal_pre.std().item()),
            "temporal_post_zero_fraction": float((temporal_post == 0).float().mean().item()),
        }

    def forward(self, x: torch.Tensor, emb=None, training: bool = True):
        # Keep compatibility with the original model's decoder-only call path.
        if emb is not None:
            return self.decode_latent(emb), emb, None

        branch_feature = self._encode_to_branch(x)

        spatial_code = self._make_spatial_code(branch_feature)
        temporal_code = self._make_temporal_code(branch_feature)

        spatial_scaled = spatial_code * self.spatial_gain()
        temporal_scaled = temporal_code * self.temporal_gain()
        temporal_flat = self._flatten_temporal(temporal_scaled)

        spatial_out = self.spatial_codec(spatial_scaled)
        temporal_hat_flat, temporal_likelihood = self.temporal_bottleneck(temporal_flat)

        spatial_hat = spatial_out["y_hat"] / self.spatial_gain()
        temporal_hat = self._unflatten_temporal(
            temporal_hat_flat
        ) / self.temporal_gain()

        self._record_codec_stats(
            spatial_scaled.detach(),
            spatial_out["y_hat"].detach(),
            temporal_flat.detach(),
            temporal_hat_flat.detach(),
        )

        latent_hat = self._fuse_to_latent(spatial_hat, temporal_hat)
        x_hat = self.decode_latent(latent_hat)

        likelihoods = {
            f"spatial_{name}": likelihood
            for name, likelihood in spatial_out["likelihoods"].items()
        }
        likelihoods["temporal"] = temporal_likelihood

        return x_hat, latent_hat, likelihoods

    def compress(self, x: torch.Tensor) -> Dict[str, Dict]:
        branch_feature = self._encode_to_branch(x)

        spatial_code = self._make_spatial_code(branch_feature)
        temporal_code = self._make_temporal_code(branch_feature)

        spatial_scaled = spatial_code * self.spatial_gain()
        temporal_scaled = temporal_code * self.temporal_gain()
        temporal_flat = self._flatten_temporal(temporal_scaled)

        spatial_packed = self.spatial_codec.compress(spatial_scaled)
        temporal_strings = self.temporal_bottleneck.compress(temporal_flat)

        return {
            "strings": {
                "spatial": spatial_packed["strings"],
                "temporal": temporal_strings,
            },
            "shape": {
                "spatial": spatial_packed["shape"],
                "temporal": tuple(int(v) for v in temporal_flat.shape[-2:]),
            },
        }

    def decompress(self, strings: Dict, shape: Dict) -> torch.Tensor:
        spatial_out = self.spatial_codec.decompress(
            strings["spatial"],
            shape["spatial"],
        )
        temporal_hat_flat = self.temporal_bottleneck.decompress(
            strings["temporal"],
            tuple(shape["temporal"]),
        )

        spatial_hat = spatial_out["y_hat"] / self.spatial_gain()
        temporal_hat = self._unflatten_temporal(
            temporal_hat_flat
        ) / self.temporal_gain()

        latent_hat = self._fuse_to_latent(spatial_hat, temporal_hat)
        return self.decode_latent(latent_hat)
