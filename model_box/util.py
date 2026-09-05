def get_conv3d_kernel_stride_padding(ds_rate):
    if ds_rate == 1:
        k = 3
        s = 1
        p = 1
    elif ds_rate == 2:
        k = 4
        s = 2
        p = 1
    elif ds_rate == 4:
        k = 8
        s = 4
        p = 2
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
    elif us_rate == 4:
        k = 8
        s = 4
        p = 2
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
