import torch
import torch.nn as nn
from model_box.util import get_conv3d_convtranspose3d_spatiotemporal_parameter


class SpatiotemporalCodeLayer(nn.Module):
    def __init__(
        self,
        s_rate,
        t_rate,
        in_chns,
        out_chns,
        act=nn.GELU,
        do_ds=True,
        use_conv=True,
        apply_bn=False,
        apply_in=False,
    ):
        super().__init__()
        self.do_ds = do_ds
        if use_conv:
            k_s, s_s, p_s = get_conv3d_convtranspose3d_spatiotemporal_parameter(s_rate, t_rate, do_ds)  # down sample parameter
            st_sample = nn.Conv3d if do_ds else nn.ConvTranspose3d
            self.st_sample = nn.Sequential(
                st_sample(in_chns, out_chns, k_s, s_s, p_s),
                act(),
            )
        elif do_ds:  # seems we do not need to use this
            k_s = s_s = [t_rate, s_rate, s_rate]
            self.st_sample = nn.MaxPool3d(k_s, s_s)
        else:
            raise NotImplementedError
        k_m, s_m, p_m = get_conv3d_convtranspose3d_spatiotemporal_parameter(1, 1)  # maintaining parameter

        boost_chns = out_chns if do_ds else in_chns
        self.st_boost_seq = nn.Sequential(
            nn.Conv3d(boost_chns, boost_chns * 2, k_m, s_m, p_m),
            nn.BatchNorm3d(boost_chns * 2) if apply_bn else nn.InstanceNorm3d(boost_chns * 2) if apply_in else nn.Identity(),
            act(),
            nn.Conv3d(boost_chns * 2, boost_chns, k_m, s_m, p_m),
            nn.BatchNorm3d(boost_chns) if apply_bn else nn.InstanceNorm3d(boost_chns) if apply_in else nn.Identity(),
            act(),
        )

    def forward(self, x):
        if self.do_ds:
            x = self.st_sample(x)
            inter = x
            x = self.st_boost_seq(x)
            x = x + inter
        else:
            inter = x
            inter = self.st_boost_seq(inter)
            x = x + inter
            x = self.st_sample(x)
        return x, inter
