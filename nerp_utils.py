import math

import random

import numpy as np

import torch

import torch.nn.functional as F

from pytorch_msssim import ms_ssim

from torchvision.transforms import Resize

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

def adjust_lr(optimizer, cur_epoch, args, type='lr'):
    
    if 'hybrid' in args.lr_type:
        up_ratio, up_pow, down_pow, min_lr, final_lr = [float(x) for x in args.lr_type.split('_')[1:]]
        if cur_epoch < up_ratio:
            lr_mult = min_lr + (1. - min_lr) * (cur_epoch / up_ratio) ** up_pow
        else:
            lr_mult = 1 - (1 - final_lr) * ((cur_epoch - up_ratio) / (1. - up_ratio))**down_pow
    elif 'cosine' in args.lr_type:
        up_ratio, up_pow, min_lr = [float(x) for x in args.lr_type.split('_')[1:]]
        if cur_epoch < up_ratio:
            lr_mult = min_lr + (1. - min_lr) * (cur_epoch / up_ratio)** up_pow
        else:
            lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - up_ratio) / (1 - up_ratio)) + 1.0)
    else:
        raise NotImplementedError

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = args.lr * lr_mult if type == 'lr' else args.aux_lr * lr_mult

    return args.lr * lr_mult if type == 'lr' else args.aux_lr * lr_mult

def psnr_fn_patch(output, gt):
    b, c, t, h, w = output.shape
    l2_loss = F.mse_loss(output.detach(), gt.detach(), reduction='none')
    l2_loss = l2_loss.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    psnr = -10 * torch.log10(l2_loss.flatten(start_dim=1).mean(1) + 1e-9)
    psnr = psnr.reshape(b, -1).mean(dim=-1)
    return psnr.cpu()

def msssim_fn_patch(output, gt):
    b, c, t, h, w = output.shape
    output = output.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
    gt = gt.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
    resize_operation = Resize([h*2 if h*2 > 180 else 180, w*2 if w*2 > 180 else 180], antialias=True)
    output = resize_operation(output)
    gt = resize_operation(gt)
    msssim = ms_ssim(output.float().detach(), gt.detach(), data_range=1, size_average=False)
    msssim = msssim.reshape(b, -1).mean(dim=-1)
    return msssim.cpu()
