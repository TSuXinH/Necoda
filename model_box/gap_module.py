import torch.nn as nn

from torch.nn.functional import interpolate

from model_box.util import get_conv3d_convtranspose3d_spatiotemporal_parameter

class GapModule(nn.Module):
    def __init__(
        self,
        output_x,
        output_t,
        in_channel,
        out_channel,
        mode='nearest',
        act=nn.GELU,
        final_act=True,
    ):
        super().__init__()
        self.mode = mode
        self.output_x = self.output_y = output_x
        self.output_t = output_t
        k_s, s_s, p_s = get_conv3d_convtranspose3d_spatiotemporal_parameter(1, 1)
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
