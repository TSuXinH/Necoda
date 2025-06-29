import torch.nn as nn
from torch.nn.functional import interpolate
from model_box.util import get_conv3d_convtranspose3d_spatiotemporal_parameter


def get_gap_module(
    output_x, output_t, interp_size_x, interp_size_t, in_channel, out_channel, do_ds=True, final_act=True, interp_method='interp'
):
    if interp_method == 'interp':
        return GapModule(interp_size_x, interp_size_t, in_channel, out_channel, final_act=final_act) \
            if do_ds else GapModule(output_x, output_t, in_channel, out_channel, final_act=final_act)
    elif interp_method == 'conv':
        interp_size_x = abs(int((interp_size_x - output_x) // 2))
        interp_size_t = abs(int((interp_size_t - output_x) // 2))
        return GapModuleConv(interp_size_x, interp_size_t, in_channel, out_channel, do_ds=do_ds)
    else:
        raise NotImplementedError


class GapModule(nn.Module):
    def __init__(
        self,
        output_x,
        output_t,
        in_channel,
        out_channel,
        mode='trilinear',
        act=nn.GELU,
        final_act=True,
    ):
        super().__init__()
        self.mode = mode
        self.output_x = self.output_y = output_x
        self.output_t = output_t
        k_s, s_s, p_s = get_conv3d_convtranspose3d_spatiotemporal_parameter(1, 1)  # down-sample feature
        hid_channel = in_channel if in_channel > out_channel else out_channel
        self.stable_layer = nn.Sequential(
            nn.Conv3d(in_channel, hid_channel, k_s, s_s, p_s),
            act(),
            nn.Conv3d(hid_channel, out_channel, k_s, s_s, p_s),
            act() if final_act else nn.Identity(),
        )

    def modify_size(self, output_x, output_t, output_y=0):
        self.output_x = output_x
        self.output_y = output_x if output_y == 0 else output_y
        self.output_t = output_t

    def forward(self, x):
        x = interpolate(x, size=(self.output_t, self.output_x, self.output_y), mode=self.mode)
        x = self.stable_layer(x)
        return x


class GapModuleConv(nn.Module):
    def __init__(
        self,
        interp_size_x,
        interp_size_t,
        in_chns,
        out_chns,
        act=nn.GELU,
        do_ds=True,
    ):
        super().__init__()
        self.do_ds = do_ds
        interp_kernel_x = interp_size_x * 2 + 1
        interp_kernel_t = interp_size_t * 2 + 1
        interp_kernel = (interp_kernel_t, interp_kernel_x, interp_kernel_x)  # expand or squeeze
        st_sample = nn.Conv3d if do_ds else nn.ConvTranspose3d
        boost_chns = out_chns if do_ds else in_chns
        s_s = (1, 1, 1)
        p_s = (0, 0, 0)
        k_m, s_m, p_m = get_conv3d_convtranspose3d_spatiotemporal_parameter(1, 1)  # maintain feature size

        self.st_sample = nn.Sequential(
            st_sample(in_chns, out_chns, interp_kernel, s_s, p_s),
            act(),
        )
        self.st_boost_seq = nn.Sequential(
            nn.Conv3d(boost_chns, boost_chns * 2, k_m, s_m, p_m),
            act(),
            nn.Conv3d(boost_chns * 2, boost_chns, k_m, s_m, p_m),
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
        return x
