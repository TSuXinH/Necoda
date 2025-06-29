import torch
import os
import shutil
from tqdm import tqdm
import argparse
import time
import pandas as pd
import numpy as np
import json
import tifffile as tif
from datetime import datetime
from auxiliary import get_patch_position
import dahuffman
import matplotlib as mpl
mpl.use('TkAgg')

from model_box import NeRPST, NeRPSTDecoder
from auxiliary import (ndarray2tif_mean_clip, ndarray2tif_min_max_clip, ndarray2tif_mean_std_clip,
                       cal_tif_num, CONVERT_UINT16_FLOAT64, ndarray2tif_mean_max_clip)


def dequant_tensor(quant_t):
    quant_t, tmin, scale = quant_t['quant'], quant_t['min'].to(torch.float32), quant_t['scale'].to(torch.float32)
    new_t = tmin.expand_as(quant_t) + scale.expand_as(quant_t) * quant_t
    return new_t


def recover_from_huffman(enc_dict, dec_list):
    recovered_len = 0
    recovered_ten = torch.tensor(dec_list)
    embed_len = torch.prod(torch.tensor(enc_dict['embed'])).item()
    quant_embed = {'quant': recovered_ten[recovered_len: recovered_len + embed_len].reshape(enc_dict['embed'])}
    recovered_len += embed_len
    enc_dict.pop('embed')
    dec_dict = {}
    for k, v in enc_dict.items():
        cur_shape = v
        cur_len = torch.prod(torch.tensor(cur_shape)).item()
        dec_dict[k] = recovered_ten[recovered_len: recovered_len + cur_len].reshape(v)
        recovered_len += cur_len
    return quant_embed, dec_dict


def save_args(args, filename='args.json'):
    args_dict = vars(args)

    # Convert NumPy types to standard Python types
    for key, value in args_dict.items():
        if isinstance(value, np.float32):
            args_dict[key] = float(value)
        elif isinstance(value, np.ndarray):  # Handle other NumPy arrays
            args_dict[key] = value.tolist()

    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--ckpt_store_dir', type=str, help='path for raw ckpt and reconstruction storage.')
    parser.add_argument('-e', '--epoch', type=int, help='select corresponding ckpt.')
    parser.add_argument('--frames', type=int, default=32, help='video frames for output',) #
    parser.add_argument('--final_size', type=int, nargs='+', default=[], help='final reconstruction size, shape in [h, w]')
    parser.add_argument('--name', type=str, default='recon', help='recontructed name')
    parser.add_argument('-s', '--standard_range', action='store_true', default=False, help='if using standard range, then the image will be rescaled by dividing 65535, else using min-max scale.')
    parser.add_argument('--tif_max', type=float, default=CONVERT_UINT16_FLOAT64, help='max value of raw tiff')
    parser.add_argument('--tif_min', type=float, default=0, help='max value of raw tiff')
    parser.add_argument('--precision', type=str, default='uint16', choices=['uint6', 'float'], help='decide the precision of the stored file.')
    parser.add_argument('-u', '--use_state', type=int, choices=[1, 2, 3, 4, 5], help='Decides whether model stores.')

    args = parser.parse_args()
    # if not os.path.exists(args.dump_dir):
    #     os.makedirs(args.dump_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load video checkpoints and dequant them
    quant_all = torch.load(os.path.join(args.ckpt_store_dir, f'quant_all_{args.epoch}.pth'), map_location='cpu')
    vid_embed_s = dequant_tensor(quant_all['quant_embed_s']).to(device)
    vid_embed_t = dequant_tensor(quant_all['quant_embed_t']).to(device)
    m_args = quant_all['m_args']
    model = NeRPST(
        pre_s_rate=m_args.pre_s_rate,
        pre_t_rate=m_args.pre_t_rate,
        s_embedding_dim=m_args.s_emb_dim,
        t_embedding_dim=m_args.t_emb_dim,
        s_s_rate_list=m_args.s_s_rate_list,
        s_t_rate_list=m_args.s_t_rate_list,
        t_s_rate_list=m_args.t_s_rate_list,
        t_t_rate_list=m_args.t_t_rate_list,
        chns_list=m_args.chns_list,
    )
    img_decoder = NeRPSTDecoder(model).to(device)
    img_dec_dict = {k if 'pass_way.' not in k else k.replace('pass_way.', ''): dequant_tensor(v).to(device) for k, v in quant_all['quant_ckpt'].items()}
    img_decoder.load_state_dict(img_dec_dict)

    np_res = np.zeros(shape=(m_args.t, m_args.x, m_args.y), dtype=np.float_)
    print('vid_embed_s.shape: ', vid_embed_s.shape)
    print('vid_embed_t.shape: ', vid_embed_t.shape)
    start_time = datetime.now()
    # Select frame indices and reconstruct them
    for patch_idx in range(len(vid_embed_s)):
        patch_out = img_decoder(
            torch.unsqueeze(vid_embed_s[patch_idx], 0),
            torch.unsqueeze(vid_embed_t[patch_idx], 0)
        ).cpu()
        idx_x, idx_y, idx_t = get_patch_position(patch_idx, m_args.num_x)
        print(idx_t, idx_x, idx_y)
        if (idx_t+1) * m_args.patch_t < m_args.t:
            np_res[
                idx_t*m_args.patch_t: (idx_t+1)*m_args.patch_t,
                idx_x*m_args.patch_x: (idx_x+1)*m_args.patch_x,
                idx_y*m_args.patch_y: (idx_y+1)*m_args.patch_y,
            ] = torch.squeeze(patch_out.detach()).numpy()
        else:
            t_gap = m_args.t - idx_t*m_args.patch_t
            np_res[
                idx_t*m_args.patch_t:,
                idx_x*m_args.patch_x: (idx_x+1)*m_args.patch_x,
                idx_y*m_args.patch_y: (idx_y+1)*m_args.patch_y,
            ] = torch.squeeze(patch_out.detach()[-t_gap:]).numpy()

    decoding_time = (datetime.now() - start_time).total_seconds()
    print('Processing completes in {}'.format(str(decoding_time)))
    print('Decoding PPS: {}'.format(len(vid_embed_s) / decoding_time))
    print(np.max(np_res), np.min(np_res))

    tif_res = ndarray2tif_mean_max_clip(np_res, tif_mean=m_args.tif_mean, tif_max_minus_mean=m_args.tif_max_minus_mean)
    out_vid = os.path.join(args.ckpt_store_dir, args.name + '.tif')
    tif.imwrite(out_vid, tif_res)

    print('Reconstruction completes in {}'.format(str(datetime.now() - start_time)))
    print()


if __name__ == '__main__':
    main()
