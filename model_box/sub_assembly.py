import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from model_box.util import get_conv3d_convtranspose3d_spatiotemporal_parameter


class CBAPLayer(nn.Module):
    def __init__(
        self,
        in_chn,
        out_chn,
        pooling_stride,
        activation=nn.GELU
    ):
        super().__init__()
        self.cbap_layer = nn.Sequential(
            nn.Conv3d(in_channels=in_chn, out_channels=out_chn, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_chn),
            activation(),
            nn.MaxPool3d(kernel_size=(pooling_stride, pooling_stride, pooling_stride), stride=(pooling_stride, pooling_stride, pooling_stride)),
        )

    def forward(self, x):
        return self.cbap_layer(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        channel_list,
        pooling_stride_list,
        linear_dim_list,
        selected_layer=None,
    ):
        super().__init__()
        self.ds_net = []
        self.linear_net = [nn.Flatten()]
        self.selected_layer = selected_layer
        for idx in range(len(channel_list)-1):
            self.ds_net.append(
                CBAPLayer(
                    in_chn=channel_list[idx],
                    out_chn=channel_list[idx+1],
                    pooling_stride=pooling_stride_list[idx]
                ),
            )
        for idx in range(len(linear_dim_list)-1):
            self.linear_net.extend([nn.Linear(linear_dim_list[idx], linear_dim_list[idx+1]), nn.GELU()]) \
                if idx != len(linear_dim_list)-2 else self.linear_net.extend([nn.Linear(linear_dim_list[-2], linear_dim_list[-1]), nn.Sigmoid()])
        self.ds_net = nn.Sequential(*self.ds_net)
        self.linear_net = nn.Sequential(*self.linear_net)

    def forward(self, x):
        if self.selected_layer is not None:
            p_loss_feat_list = []
            for idx in range(len(self.ds_net)):
                x = self.ds_net[idx](x)
                if idx in self.selected_layer:
                    p_loss_feat_list.append(x)
            x = self.linear_net(x)
            return x, p_loss_feat_list
        else:
            return self.linear_net(self.ds_net(x))


def cal_p_loss(discriminator, generation, target, loss_fn):
    discriminator.eval()
    _, generation_feat = discriminator(generation)
    _, target_feat = discriminator(target)
    final_loss = 0
    for feat_idx in range(len(generation_feat)):
        final_loss += loss_fn(generation_feat[feat_idx], target_feat[feat_idx])
    return final_loss


def cal_generator_loss(discriminator, generation, loss_fn, device='cuda'):
    discriminator.eval()
    generation_pred, _ = discriminator(generation)
    label = torch.zeros_like(generation_pred).to(device)
    loss = loss_fn(generation_pred, label)
    return loss


def cal_discriminator_loss(discriminator, target, loss_fn, device='cuda'):
    discriminator.train()
    target_pred, _ = discriminator(target)
    label = torch.ones_like(target_pred).to(device)
    loss = loss_fn(target_pred, label)
    return loss


class TemporalFusionLayer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        pass

