import torch.nn as nn
import torch.nn.init as init


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(
                    m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def get_conv3d_kernel_stride_padding(ds_rate):
    if ds_rate == 1:
        k = 3
        s = 1
        p = 1
    elif ds_rate == 2:
        k = 4
        s = 2
        p = 1
    elif ds_rate == 3:
        k = 5
        s = 3
        p = 1
    elif ds_rate == 4:
        k = 8
        s = 4
        p = 2
    elif ds_rate == 5:
        k = 7
        s = 5
        p = 1
    else:
        raise NotImplementedError
    return k, s, p


def get_convtranspose3d_kernel_stride_padding(us_rate):
    if us_rate == 1:
        k = 3
        s = 1
        p = 1
    elif us_rate == 2:
        k = 4
        s = 2
        p = 1
    elif us_rate == 3:
        k = 5
        s = 3
        p = 1
    elif us_rate == 4:
        k = 8
        s = 4
        p = 2
    elif us_rate == 5:
        k = 7
        s = 5
        p = 1
    else:
        raise NotImplementedError
    return k, s, p


def get_conv3d_convtranspose3d_spatiotemporal_parameter(s_rate, t_rate, ds=True):
    if ds:
        k_s, s_s, p_s = get_conv3d_kernel_stride_padding(s_rate)
        k_t, s_t, p_t = get_conv3d_kernel_stride_padding(t_rate)
    else:
        k_s, s_s, p_s = get_convtranspose3d_kernel_stride_padding(s_rate)
        k_t, s_t, p_t = get_convtranspose3d_kernel_stride_padding(t_rate)
    return (k_t, k_s, k_s), (s_t, s_s, s_s), (p_t, p_s, p_s)
