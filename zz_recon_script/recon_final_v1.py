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
from dahuffman import HuffmanCodec
import matplotlib as mpl
mpl.use('TkAgg')

from model_box import NeRPSTPro, NeRPSTProDecoder
from auxiliary import (ndarray2tif_mean_clip, ndarray2tif_min_max_clip, ndarray2tif_mean_std_clip,
                       cal_tif_num, CONVERT_UINT16_FLOAT64, ndarray2tif_mean_max_clip, create_overlap_patch_info_test, get_interp_coord)


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


def decode_huffman(huff_path, storage_name):
    codec = HuffmanCodec.load(os.path.join(huff_path, f'codec_{storage_name}'))
    with open(os.path.join(huff_path, f'encode_{storage_name}.bin'), 'rb') as f:
        huff_emb = f.read()
    dec = codec.decode(huff_emb)
    dec_tensor = torch.tensor(dec).to(torch.uint8)
    uncompressed_tensor = torch.zeros(size=(2, len(dec_tensor)), dtype=torch.uint8)
    uncompressed_tensor[0] = (dec_tensor << 4) >> 4
    uncompressed_tensor[1] = dec_tensor >> 4
    return uncompressed_tensor


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
    parser.add_argument('-g', '--generalization_id_list', type=int, nargs='+', default=[], help='Add generalization embedding and decoding')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load video checkpoints and dequant them
    print(args.ckpt_store_dir)
    print('Current epoch: {}'.format(args.epoch))
    model_all = torch.load(os.path.join(args.ckpt_store_dir, f'model_quant_{args.epoch}.pth'), map_location='cpu')
    args_dict = torch.load(f'{args.ckpt_store_dir}/hyper_param.pth')
    model = NeRPSTPro(
        raw_size_x=args_dict['patch_x'],
        raw_size_t=args_dict['patch_t'],
        interp_size_x=args_dict['interp_size_x'] * 2 + args_dict['patch_x'],
        interp_size_t=args_dict['interp_size_t'] * 2 + args_dict['patch_t'],
        interp_chn=args_dict['interp_chn'],
        pre_s_rate=args_dict['pre_s_rate'],
        pre_t_rate=args_dict['pre_t_rate'],
        s_embedding_dim=args_dict['s_emb_dim'],
        t_embedding_dim=args_dict['t_emb_dim'],
        s_s_rate_list=args_dict['s_s_rate_list'],
        s_t_rate_list=args_dict['s_t_rate_list'],
        t_s_rate_list=args_dict['t_s_rate_list'],
        t_t_rate_list=args_dict['t_t_rate_list'],
        chns_list=args_dict['chns_list'],
    )
    img_decoder = NeRPSTProDecoder(model).to(device)
    img_dec_dict = {k if 'pass_way.' not in k else k.replace('pass_way.', ''): dequant_tensor(v).to(device) for k, v in
                    model_all.items()}
    img_decoder.load_state_dict(img_dec_dict)
    # print()
    # print('args_dict_record')
    # print(args_dict['x'])
    # print(args_dict['t'])
    # print(args_dict['patch_x'])
    # print(args_dict['patch_t'])
    #
    # print()
    for item in args.generalization_id_list:
        print('Current generalization id: {}'.format(item))
        # todo:  2, write a 7z unzip to uncompress the 7z folder to get the reconstruction results, test before writing
        emb_storage_path = os.path.join(args.ckpt_store_dir, 'g{}'.format(item))
        huff_storage_path = os.path.join(emb_storage_path, f'huff_{args.epoch}')
        args_dict_m = torch.load(os.path.join(huff_storage_path, 'args.pth'))
        embed_s = decode_huffman(huff_storage_path, 's')
        embed_t = decode_huffman(huff_storage_path, 't')
        embed_s = embed_s.reshape(args_dict_m['s_size']).to(device)
        embed_t = embed_t.reshape(args_dict_m['t_size']).to(device)
        dict_embed_s = {'quant': embed_s, 'scale': args_dict_m['s_scale'], 'min': args_dict_m['s_min']}
        dict_embed_t = {'quant': embed_t, 'scale': args_dict_m['t_scale'], 'min': args_dict_m['t_min']}
        print(args_dict_m['s_scale'].shape)
        print(args_dict_m['t_scale'].shape)
        vid_embed_s = dequant_tensor(dict_embed_s)
        vid_embed_t = dequant_tensor(dict_embed_t)
        coord_list = create_overlap_patch_info_test(
            args_dict_m['patch_x'], args_dict_m['x'], args_dict_m['patch_t'], args_dict_m['t'],
            args_dict_m['y'] if 'y' in args_dict_m.keys() else args_dict_m['x']
        )
        np_res = np.zeros(shape=(args_dict_m['t'], args_dict_m['x'],
                                 args_dict_m['y'] if 'y' in args_dict_m.keys() else args_dict_m['x']), dtype=np.float_)
        # np_res = np.zeros(shape=(args_dict_m['t'], args_dict_m['x'], 1792), dtype=np.float_)
        mode_res = np.zeros_like(np_res)
        print('vid_embed_s.shape: ', vid_embed_s.shape)
        print('vid_embed_t.shape: ', vid_embed_t.shape)
        print('mode_res.shape: ', mode_res.shape)
        start_time = datetime.now()

        # Select frame indices and reconstruct them
        for patch_idx in range(len(vid_embed_s)):
            # print('patch_idx: ', patch_idx)
            patch_out = img_decoder(
                torch.unsqueeze(vid_embed_s[patch_idx], 0),
                torch.unsqueeze(vid_embed_t[patch_idx], 0)
            ).cpu()
            patch_fill_in = torch.squeeze(patch_out.detach()).numpy()
            # print('patch_fill_in.shape: ', patch_fill_in.shape)
            cur_coord = coord_list[patch_idx]
            start_t, end_t = get_interp_coord(cur_coord[0], args_dict_m['t'], args_dict_m['patch_t'], args_dict_m['interp_size_t'])
            start_x, end_x = get_interp_coord(cur_coord[1], args_dict_m['x'], args_dict_m['patch_x'], args_dict_m['interp_size_x'])
            start_y, end_y = get_interp_coord(cur_coord[2], args_dict_m['x'], args_dict_m['patch_x'], args_dict_m['interp_size_x'])
            overlap_mode = np.ones_like(patch_fill_in)
            mode_res[start_t: end_t, start_x: end_x, start_y: end_y] += overlap_mode
            np_res[start_t: end_t, start_x: end_x, start_y: end_y] += patch_fill_in

        mode_res = np.reciprocal(mode_res)
        np_res = np_res * mode_res

        decoding_time = (datetime.now() - start_time).total_seconds()
        print('Processing completes in {}'.format(str(decoding_time)))
        print('Decoding PPS: {}'.format(len(vid_embed_s) / decoding_time))
        print(np.max(np_res), np.min(np_res))
        print('Current mean and max_minus_mean: {}, {}'.format(args_dict_m['tif_mean'], args_dict_m['tif_max_minus_mean']))
        # tif_res = np_res.astype(np.float32)
        tif_res = ndarray2tif_mean_max_clip(np_res, tif_mean=args_dict_m['tif_mean'], tif_max_minus_mean=args_dict_m['tif_max_minus_mean'])
        out_vid = os.path.join(emb_storage_path, args.name + '.tif')
        tif.imwrite(out_vid, tif_res)

        print('Reconstruction completes in {}'.format(str(datetime.now() - start_time)))
        print()


if __name__ == '__main__':
    main()
