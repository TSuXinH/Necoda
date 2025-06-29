import imageio
# from pygifsicle import optimize
import argparse
import os
import random
import shutil
from torch.autograd import Variable
from datetime import datetime
import numpy as np
import csv
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.optim as optim
import torch.utils.data
# from torch.utils.tensorboard import SummaryWriter
from model_box import NeRPSTPro, NeRPSTProDecoder, Discriminator, cal_discriminator_loss, cal_p_loss, cal_generator_loss
from nerp_utils import *
from torch.utils.data import Subset
from copy import deepcopy
from dahuffman import HuffmanCodec
from auxiliary import DatasetTifPatchTest, DatasetTifPatchTrainWithPadding
from torchvision.utils import save_image
import torch.nn.functional as F
import pandas as pd
from collections import OrderedDict
import json
from einops import rearrange


def main():
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='', help='data path for vid')
    parser.add_argument('--output_path', default='unify', help='folder to output images and model checkpoints')
    parser.add_argument('--shuffle_data', action='store_true', help='randomly shuffle the frame idx')
    parser.add_argument('--remark', type=str, default='', help='remark that leaves behind the resultant file name')
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--patch_x', type=int, default=128)
    parser.add_argument('--patch_t', type=int, default=128)
    parser.add_argument('--gap_x', type=int, default=64)
    parser.add_argument('--gap_t', type=int, default=64)
    parser.add_argument('--interp_size_x', type=int, default=4, help='half of xy padded for overlap.')
    parser.add_argument('--interp_size_t', type=int, default=4, help='half of t padded for overlap.')
    parser.add_argument('--interp_chn', type=int, default=32, help='half of t padded for overlap.')
    parser.add_argument('--interp_method', type=str, default='interp', choices=['interp', 'conv'], help='select methods for interpolating for overlap')
    parser.add_argument('-p', '--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--apply_augmentation', action='store_true', help='add augmentation for raw and target')
    parser.add_argument('--apply_sampling', action='store_true', default=False, help='add hierarchical sampling')
    parser.add_argument('-g', '--generalization_data_path', type=str, nargs='+', default=[],
                        help='import generalization data for testing')
    parser.add_argument('--lam_spatial', type=float, default=0, help='weight for temporal averaging loss')
    parser.add_argument('--lam_temporal', type=float, default=0, help='weight for spatial averaging loss')
    parser.add_argument('--discriminator_lr', type=float, nargs='+', default=1e-6, help='discriminator learning rate')
    parser.add_argument('--discriminator_channel_list', type=int, nargs='+', default=[1, 16, 32, 16, 8],
                        help='channel list for conv nets in GAN discriminator')
    parser.add_argument('--discriminator_pooling_stride_list', type=int, nargs='+', default=[4, 4, 2, 2],
                        help='pooling strides (kernels) for discriminator')
    parser.add_argument('--discriminator_linear_dim_list', type=int, nargs='+', default=[64, 256, 64, 16, 1],
                        help='pooling strides (kernels) for discriminator')
    parser.add_argument('--discriminator_selected_layer', type=int, nargs='+', default=[0, 1],
                        help='pooling strides (kernels) for discriminator')

    # parser.add_argument('--data_split', type=str, default='1_1_1',
    #                     help='Valid_train/total_train/all data split, e.g., 18_19_20 means for every 20 samples, '
    #                          'the first 19 samples is full train set, and the first 18 samples is chose currently')
    parser.add_argument('--resize_list', type=str, default='-1', help='video resize size', )
    parser.add_argument('--img_chns', type=int, default=3,
                        help='the input and output channels of image, e.g. 1 for gray and 3 for rgb')
    parser.add_argument('--pre_norm', type=str, default='mean', choices=['min_max', 'mean', 'mean_std', 'mean_max'],
                        help='determine the pre process of images')

    # NERV architecture parameters
    # Embedding and encoding parameters
    parser.add_argument('--pre_s_rate', type=int, default=2,
                        help='st encoder, decoder spatial shrinking or expanding rate')
    parser.add_argument('--pre_t_rate', type=int, default=2,
                        help='st encoder, decoder temporal shrinking or expanding rate')
    parser.add_argument('--s_emb_dim', type=int, default=8, help='spatial pass way embedding dim')
    parser.add_argument('--t_emb_dim', type=int, default=8, help='temporal pass way embedding dim')
    parser.add_argument('--s_s_rate_list', type=int, nargs='+', default=[],
                        help='spatial pass way spatial sample rate list')
    parser.add_argument('--t_s_rate_list', type=int, nargs='+', default=[],
                        help='temporal pass way spatial sample rate list')
    parser.add_argument('--s_t_rate_list', type=int, nargs='+', default=[],
                        help='spatial pass way temporal sample rate list')
    parser.add_argument('--t_t_rate_list', type=int, nargs='+', default=[],
                        help='temporal pass way temporal sample rate list')
    parser.add_argument('--chns_list', type=int, nargs='+', default=[], help='channels list')
    parser.add_argument('--model_type', type=str, default='raw', choices=['nerv', 'hnerv', 'diff', 'nerp', 'nerp_st'],
                        help='Define the type of the model.')
    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator',
                        choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use',
                        choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])

    # General training setups
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--start_epoch', type=int, default=-1, help='starting epoch')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Epoch number')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--lr_type', type=str, default='cosine_0.1_1_0.1', help='learning rate type, default=cosine')
    parser.add_argument('--loss', type=str, default='L2', help='loss type, default is L2')
    parser.add_argument('--out_bias', default='tanh', type=str, help='using sigmoid/tanh/0.5 for output prediction')

    # evaluation parameters
    parser.add_argument('--eval_freq', type=int, default=10, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--quant_model_bit', type=int, default=8, help='bit length for model quantization')
    parser.add_argument('--quant_embed_bit', type=int, default=6, help='bit length for embedding quantization')
    parser.add_argument('--quant_axis', type=int, default=0, help='quantization axis (-1 means per tensor)')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')
    parser.add_argument('--encoder_file', default='', type=str, help='specify the embedding file')

    # process all the basic settings
    torch.set_printoptions(precision=2)

    args = parser.parse_args()
    args.output_path = os.path.join('output', args.output_path)
    args.chns_list_str = ','.join([str(x) for x in args.chns_list])
    args.quant_str = f'Q_M{args.quant_model_bit}_E{args.quant_embed_bit}'
    exp_id = (
        f'{args.model_type}_B{args.batchSize}_E{args.epochs}'
        f'_chns{args.chns_list_str}_{args.quant_str}_lr{args.lr}_{args.loss}_{args.pre_norm}'
        f'_ts{np.prod(args.t_s_rate_list).item()}tt{np.prod(args.t_t_rate_list).item()}ss{np.prod(args.s_s_rate_list).item()}st{np.prod(args.s_t_rate_list).item()}'
    )
    if args.apply_sampling:
        exp_id += '_sampling'
    if args.apply_augmentation:
        exp_id += '_aug'
    if args.lam_temporal:
        exp_id += '_tlam{}'.format(args.lam_temporal)
    if args.lam_spatial:
        exp_id += '_slam{}'.format(args.lam_spatial)
    args.exp_id = exp_id + '_' + args.remark if args.remark != '' else exp_id
    args.output_path = os.path.join(args.output_path) + '/' + args.exp_id

    if args.overwrite and os.path.isdir(args.output_path):
        print('The existing output dir will be overwritten.')
        shutil.rmtree(args.output_path)
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    port = hash(args.exp_id) % 20000 + 10000
    args.init_method = f'tcp://127.0.0.1:{port}'
    args.ngpus_per_node = torch.cuda.device_count()
    print(f'init_method: {args.init_method}', flush=True)

    train(None, args)


def data_to_gpu(x, device):
    return x.to(device)


def train(local_rank, args):
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    args.metric_names = ['pred_seen_psnr', 'pred_seen_ssim', 'pred_unseen_psnr', 'pred_unseen_ssim',
                         'quant_seen_psnr', 'quant_seen_ssim', 'quant_unseen_psnr', 'quant_unseen_ssim']
    best_metric_list = [torch.tensor(0) for _ in range(len(args.metric_names))]

    print('GENERATING TRAINING DATALOADER -->')
    train_dataset = DatasetTifPatchTrainWithPadding(
        args.data_path,
        patch_x=args.patch_x,
        patch_t=args.patch_t,
        gap_x=args.gap_x,
        gap_t=args.gap_t,
        apply_aug=args.apply_augmentation,
        apply_sampling=args.apply_sampling,
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize,
                                                   shuffle=True, num_workers=args.workers, pin_memory=True,
                                                   drop_last=True, worker_init_fn=worker_init_fn)
    print('GENERATING TESTING DATALOADER -->')
    test_dataset = DatasetTifPatchTest(
        args.data_path,
        patch_x=args.patch_x,
        patch_t=args.patch_t,
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize,
                                                  shuffle=False, num_workers=args.workers, pin_memory=True,
                                                  drop_last=False, worker_init_fn=worker_init_fn)
    args.patch_y = args.patch_x
    args.num_x = args.num_y = test_dataset.num_x
    args.num_t = test_dataset.num_t
    args.x = test_dataset.x
    args.y = test_dataset.y
    args.t = test_dataset.t
    args.interp_size_y = args.interp_size_x

    print('GENERATING GENERALIZATION DATALOADER LIST -->')

    gen_dataloader_list = []
    for gen_dataset_path in args.generalization_data_path:
        gen_dataset = DatasetTifPatchTest(gen_dataset_path)
        gen_dataloader = torch.utils.data.DataLoader(gen_dataset, batch_size=args.batchSize,
                                                     shuffle=False, num_workers=args.workers, pin_memory=True,
                                                     drop_last=False, worker_init_fn=worker_init_fn)
        gen_dataloader_list.append(gen_dataloader)
    # Building model
    model = NeRPSTPro(
        raw_size_x=args.patch_x,
        raw_size_t=args.patch_t,
        interp_size_x=args.interp_size_x * 2 + args.patch_x,
        interp_size_t=args.interp_size_t * 2 + args.patch_t,
        interp_chn=args.interp_chn,
        pre_s_rate=args.pre_s_rate,
        pre_t_rate=args.pre_t_rate,
        s_embedding_dim=args.s_emb_dim,
        t_embedding_dim=args.t_emb_dim,
        s_s_rate_list=args.s_s_rate_list,
        s_t_rate_list=args.s_t_rate_list,
        t_s_rate_list=args.t_s_rate_list,
        t_t_rate_list=args.t_t_rate_list,
        chns_list=args.chns_list,
        act=nn.GELU
    )
    discriminator = Discriminator(
        args.discriminator_channel_list,
        args.discriminator_pooling_stride_list,
        args.discriminator_linear_dim_list,
        args.discriminator_selected_layer
    )

    encoder_param = ((sum([p.data.nelement() for p in model.st_encoder.parameters()]) / 1e6)
                     + (sum([p.data.nelement() for p in model.t_pass_way.encoder.parameters()]) / 1e6)
                     + (sum([p.data.nelement() for p in model.s_pass_way.encoder.parameters()]) / 1e6)
                     + (sum([p.data.nelement() for p in model.interp_encoder.parameters()]) / 1e6))
    decoder_param = ((sum([p.data.nelement() for p in model.st_decoder.parameters()]) / 1e6)
                     + (sum([p.data.nelement() for p in model.t_pass_way.decoder.parameters()]) / 1e6)
                     + (sum([p.data.nelement() for p in model.s_pass_way.decoder.parameters()]) / 1e6)
                     + (sum([p.data.nelement() for p in model.interp_decoder.parameters()]) / 1e6))
    # total_param = decoder_param + embed_param / 1e6
    # args.encoder_param, args.decoder_param, args.total_param = encoder_param, decoder_param, total_param
    args.encoder_param, args.decoder_param = encoder_param, decoder_param
    # param_str = f'Encoder_{round(encoder_param, 3)}M_Decoder_{round(decoder_param, 3)}
    # M_Emb_{round(embed_param / 1e6, 3)}M_Total_{round(total_param, 3)}M'
    param_str = f'Encoder_{round(encoder_param, 3)}M_Decoder_{round(decoder_param, 3)}M'
    print(f'{args}\n {model}\n {param_str}', flush=True)
    print(discriminator, flush=True)
    with open('{}/rank0.txt'.format(args.output_path), 'a') as f:
        f.write(str(model) + '\n' + f'{param_str}\n')

    print("Use GPU: {} for training".format(local_rank))
    if args.ngpus_per_node > 1:
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)
    elif torch.cuda.is_available():
        model = model.cuda()
        discriminator = discriminator.cuda()
    if args.start_epoch < 0:
        args.start_epoch = max(args.start_epoch, 0)

    optimizer = optim.Adam(model.parameters(), weight_decay=0.)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.discriminator_lr, weight_decay=1e-5)

    # Training
    start = datetime.now()

    psnr_list = []
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        epoch_start_time = datetime.now()
        pred_psnr_list = []
        l2_loss_list = []
        p_loss_list = []
        g_loss_list = []
        d_loss_list = []
        device = next(model.parameters()).device
        for i, sample in enumerate(train_dataloader):
            patch_data = data_to_gpu(sample['patch'], device)
            patch_gt = data_to_gpu(sample['target'], device)
            # print('patch_data.shape: ', patch_data.shape)
            # print('patch_gt.shape: ', patch_gt.shape)
            patch_data = Variable(patch_data)
            cur_epoch = (epoch + float(i) / len(train_dataloader)) / args.epochs
            lr = adjust_lr(optimizer, cur_epoch, args)
            patch_out, _, _ = model(patch_data)
            l2_loss = F.smooth_l1_loss(patch_out, patch_gt)
            loss_p = cal_p_loss(discriminator, patch_out, patch_gt, F.smooth_l1_loss)
            loss_g = cal_generator_loss(discriminator, patch_out, F.binary_cross_entropy)

            if args.lam_spatial:
                temporal_averaging_loss = F.smooth_l1_loss(torch.mean(patch_out, dim=2), torch.mean(patch_gt, dim=2))
                l2_loss += args.lam_spatial * temporal_averaging_loss
            if args.lam_temporal:
                spatial_averaging_loss = F.smooth_l1_loss(
                    torch.mean(patch_out, dim=[3, 4]),
                    torch.mean(patch_gt, dim=[3, 4])
                )
                l2_loss += args.lam_temporal * spatial_averaging_loss

            total_g_loss = l2_loss + 1e-4 * loss_p + 1e-4 * loss_g
            optimizer.zero_grad()
            total_g_loss.backward(retain_graph=True)
            optimizer.step()

            loss_d = cal_discriminator_loss(discriminator, patch_gt, F.binary_cross_entropy)
            optimizer_d.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizer_d.step()
            
            l2_loss_list.append(l2_loss.detach().item())
            p_loss_list.append(loss_p.detach().item())
            g_loss_list.append(loss_g.detach().item())
            d_loss_list.append(loss_d.detach().item())

            pred_psnr_list.append(psnr_fn_patch(patch_out.detach(), patch_gt))
            if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                pred_psnr = torch.cat(pred_psnr_list).mean()
                print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e}, pred_PSNR: {}, l2_loss: {:.6e}, p_loss: {:.6e}, g_loss: {:.6e}, d_loss: {:.6e}'.format(
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                    local_rank, epoch + 1, args.epochs, i + 1, len(train_dataloader), lr,
                    round_tensor(pred_psnr, 2),
                    np.mean(l2_loss_list),
                    np.mean(p_loss_list),
                    np.mean(g_loss_list),
                    np.mean(d_loss_list),
                )

                print(print_str, flush=True)
                with open('{}/rank0.txt'.format(args.output_path), 'a') as f:
                    f.write(print_str + '\n')

        epoch_end_time = datetime.now()
        print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format(
            (epoch_end_time - epoch_start_time).total_seconds(),
            (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch)))

        if (epoch + 1) % args.eval_freq == 0 or args.epochs - epoch == 1:
            results_list, hw = evaluate(model, test_dataloader, local_rank, args, epoch,
                                        True if epoch == args.epochs - 1 else False)
            for idx, generalization_dataloader in enumerate(gen_dataloader_list):
                _, _ = evaluate(model, generalization_dataloader, local_rank, args, epoch,
                                True if epoch == args.epochs - 1 else False, generalization_id=idx + 2)
            print_str = f'Eval at epoch {epoch + 1} for {hw}: '
            for i, (metric_name, best_metric_value, metric_value) in enumerate(
                    zip(args.metric_names, best_metric_list, results_list)):
                best_metric_value = best_metric_value if best_metric_value > metric_value.max() else metric_value.max()
                if '_seen_' in metric_name:
                    if metric_name == 'pred_seen_psnr':
                        psnr_list.append(metric_value.max())
                    print_str += f'{metric_name}: {round_tensor(metric_value, 2)} | '
                best_metric_list[i] = best_metric_value
            print(print_str, flush=True)
            with open('{}/rank0.txt'.format(args.output_path), 'a') as f:
                f.write(print_str + '\n')

        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),
        }
        if epoch == args.epochs - 1:
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.output_path))
            args.cur_epoch = epoch + 1
            args.train_time = str(datetime.now() - start)
            # Dump2CSV(args, best_metric_list, results_list, psnr_list, f'epoch{epoch + 1}.csv')
            torch.save(save_checkpoint, f'{args.output_path}/epoch{epoch + 1}.pth')
            if best_metric_list[0] == results_list[0]:
                torch.save(save_checkpoint, f'{args.output_path}/model_best.pth')

    print(f"Training completes in: {str(datetime.now() - start)}")


# Writing final results in CSV file
def Dump2CSV(args, best_results_list, results_list, psnr_list, filename='results.csv'):
    result_dict = {'CurEpoch': args.cur_epoch, 'Time': args.train_time,
                   'FPS': args.fps, 'Split': args.data_split, 'Embed': args.embed,
                   'Resize': args.resize_list, 'Lr_type': args.lr_type, 'LR (E-3)': args.lr * 1e3,
                   'Batch': args.batchSize,
                   'Size (M)': f'{round(args.encoder_param, 2)}_{round(args.decoder_param, 2)}_{round(args.total_param, 2)}',
                   'model_size': args.model_size, 'Epoch': args.epochs, 'Loss': args.loss, 'Act': args.act,
                   'Norm': args.norm, 'pre_norm': args.pre_norm,
                   'FC': args.fc_hw, 'Reduce': args.reduce, 'ENC_type': args.conv_type[0],
                   'ENC_strds': args.enc_strd_str, 'KS': args.ks,
                   'enc_dim': args.enc_dim, 'DEC': args.conv_type[1], 'DEC_strds': args.dec_strd_str,
                   'lower_width': args.lower_width,
                   'Quant': args.quant_str, 'bits/param': args.bits_per_param,
                   'bits/param w/ overhead': args.full_bits_per_param,
                   'bits/pixel': args.total_bpp,
                   f'PSNR_list_{args.eval_freq}': ','.join([round_tensor(v, 2) for v in psnr_list]), }
    result_dict.update(
        {f'best_{k}': round_tensor(v, 4 if 'ssim' in k else 2) for k, v in zip(args.metric_names, best_results_list)})
    result_dict.update(
        {f'{k}': round_tensor(v, 4 if 'ssim' in k else 2) for k, v in zip(args.metric_names, results_list) if
         'pred' in k})
    csv_path = os.path.join(args.output_path, filename)
    pd.DataFrame(result_dict, index=[0]).to_csv(csv_path)
    print(f'results dumped to {csv_path}')


@torch.no_grad()
def evaluate(model, test_dataloader, local_rank, args, epoch=1, huffman_coding=False, generalization_id=1):
    print('START TESTING') if generalization_id == 1 \
        else print('START GENERALIZATION DATA TESTING --> {}'.format(generalization_id))
    emb_s_list = []
    emb_t_list = []
    model_list, quant_ckt, quanted_model, quant_min_scale, raw_ckt = quant_model(model, args)
    metric_list = [[] for _ in range(len(args.metric_names))]  # [] numbers: 8
    for model_ind, cur_model in enumerate(model_list):  # model_index 0: raw model, 1: quantized model
        cur_model.eval()
        device = next(cur_model.parameters()).device
        patch_psnr_list = []
        patch_msssim_list = []
        for i, sample in enumerate(test_dataloader):
            patch_data = data_to_gpu(sample['patch'], device)
            patch_out, emb_s, emb_t = cur_model(
                patch_data,
                dequant_vid_embed_s[i] if model_ind else None,
                dequant_vid_embed_t[i] if model_ind else None
            )

            if model_ind == 0:
                emb_s_list.append(emb_s)
                emb_t_list.append(emb_t)

            pred_psnr, pred_ssim = psnr_fn_patch(patch_out, patch_data), msssim_fn_patch(patch_out, patch_data)
            for idx in range(len(pred_psnr)):
                patch_psnr_list.append(pred_psnr[idx])
                patch_msssim_list.append(pred_ssim[idx])

            # print eval results and add to log txt
            if i % args.print_freq == 0 or i == len(test_dataloader) - 1:
                print_str = '[{}] Rank:{}, Eval at Step [{}/{}] '.format(
                    datetime.now().strftime("%Y/%m/%d %H:%M:%S"), local_rank, i + 1, len(test_dataloader))
                metric_name1 = ('quant' if model_ind else 'pred') + '_seen_psnr'
                metric_name2 = ('quant' if model_ind else 'pred') + '_seen_ssim'
                cur_psnr_mean = np.mean(patch_psnr_list)
                cur_msssim_mean = np.mean(patch_msssim_list)
                print_str += f'{metric_name1}: {round_tensor(cur_psnr_mean, 2)} | '
                print_str += f'{metric_name2}: {round_tensor(cur_msssim_mean, 2)}'
                print_str += f' | generalization dataset id: {generalization_id}'
                print(print_str, flush=True)
                with open('{}/rank0.txt'.format(args.output_path), 'a') as f:
                    f.write(print_str + '\n')

        # embedding quantization
        if model_ind == 0:
            vid_emb_s = torch.cat(emb_s_list, 0)  # concatenate all the encoded embedding.
            vid_emb_t = torch.cat(emb_t_list, 0)  # concatenate all the encoded embedding.
            quant_emb_s, dequant_emved_s = quant_tensor(vid_emb_s, args.quant_embed_bit)
            quant_emb_t, dequant_emved_t = quant_tensor(vid_emb_t, args.quant_embed_bit)
            dequant_vid_embed_s = dequant_emved_s.split(args.batchSize, dim=0)
            dequant_vid_embed_t = dequant_emved_t.split(args.batchSize, dim=0)

        results_list = [torch.stack(v_list, dim=1).mean(1).cpu() if len(v_list) else torch.zeros(1) for v_list in
                        metric_list]
        h, w = patch_data.shape[-2:]

    # dump quantized checkpoint, and decoder
    if quant_ckt is not None:
        # raw_path = f'{args.output_path}/old'
        # huff_storage = f'{args.output_path}/huff'
        # ### those important dataset parameters should be saved individually
        cur_gen_dir = f'{args.output_path}/g{generalization_id}'
        if not os.path.isdir(cur_gen_dir):
            os.makedirs(cur_gen_dir)
        args.x, args.y = test_dataloader.dataset.x, test_dataloader.dataset.y
        args.tif_mean, args.tif_max_minus_mean = test_dataloader.dataset.tif_mean, test_dataloader.dataset.tif_max_minus_mean

        # if not os.path.isdir(raw_path):
        #     os.makedirs(raw_path)
        # if not os.path.isdir(huff_storage):
        #     os.makedirs(huff_storage)
        # torch.save(quant_min_scale, f'{args.output_path}/old/quant_min_scale_{epoch}.pth')
        # torch.save(quant_embed, f'{args.output_path}/old/quant_embed_{epoch}.pth')
        # torch.jit.save(torch.jit.trace(NeRPDecoder(quanted_model), (vid_embed[:2])),
        #                f'{args.output_path}/old/img_dec_{epoch}.pth')
        after_quant_remark = '{}'.format(epoch + 1)
        quant_all = {'quant_ckpt': quant_ckt, 'quant_embed_s': quant_emb_s, 'quant_embed_t': quant_emb_t,
                     'm_args': args}
        raw_all = {'raw_all': raw_ckt, 'raw_embed_s': vid_emb_s, 'raw_embed_t': vid_emb_t, 'm_args': args}
        torch.save(quant_all, f'{cur_gen_dir}/quant_all_{after_quant_remark}.pth')
        torch.save(raw_all, f'{cur_gen_dir}/raw_all_{after_quant_remark}.pth')

        # huffman coding
        # if huffman_coding:
        #     size_dict_all = OrderedDict()
        #     size_dict_all['embed'] = quant_embed['quant'].shape
        #     quant_v_list = quant_embed['quant'].flatten().tolist()
        #     tmin_scale_len = quant_embed['min'].nelement() + quant_embed['scale'].nelement()
        #     for k, layer_wt in quant_ckt.items():
        #         size_dict_all[k] = layer_wt['quant'].shape
        #         quant_v_list.extend(layer_wt['quant'].flatten().tolist())
        #         tmin_scale_len += layer_wt['min'].nelement() + layer_wt['scale'].nelement()
        #
        #     # get the element name and its frequency
        #     unique, counts = np.unique(quant_v_list, return_counts=True)
        #     num_freq = dict(zip(unique, counts))
        #
        #     # generating HuffmanCoding table
        #     codec = HuffmanCodec.from_data(quant_v_list)
        #     sym_bit_dict = {}
        #     for k, v in codec.get_code_table().items():
        #         sym_bit_dict[k] = v[0]
        #     encoded_quant = codec.encode(quant_v_list)
        #
        #     # total bits for quantized embed + model weights
        #     total_bits = 0
        #     for num, freq in num_freq.items():
        #         total_bits += freq * sym_bit_dict[num]
        #     args.bits_per_param = total_bits / len(quant_v_list)
        #
        #     # including the overhead for min and scale storage,
        #     total_bits += tmin_scale_len * 16  # (16bits for float16)
        #     args.full_bits_per_param = total_bits / len(quant_v_list)
        #
        #     # bits per pixel
        #     args.total_bpp = total_bits / args.final_size / args.full_data_length
        #     print_str = f'After quantization and encoding: \nBits per parameter: {round(args.full_bits_per_param, 2)}, bits per pixel: {round(args.total_bpp, 4)}'
        #     print(print_str)
        #     with open('{}/rank0.txt'.format(args.output_path), 'a') as f:
        #         f.write(print_str + '\n')
        #
        #     # huffman storage
        #     with open('{}/encode.bin'.format(huff_storage, epoch), 'wb') as f:
        #         f.write(encoded_quant)
        #     codec.save('{}/codec'.format(huff_storage, epoch))
        #     embed_min_scale = {'min': quant_embed['min'], 'scale': quant_embed['scale']}
        #     torch_quant = {'embed_min_scale': embed_min_scale, 'model_min_scale': quant_min_scale,
        #                    'size_all': size_dict_all, 'm_args': args}
        #     torch.save(torch_quant, '{}/torch_quant.pth'.format(huff_storage, epoch))
    return results_list, (h, w)


def quant_model(model, args):
    device = next(model.parameters()).device
    model_list = [deepcopy(model)]
    if args.quant_model_bit == -1:
        return model_list, None
    else:
        cur_model = deepcopy(model)
        quanted_model = deepcopy(model)
        raw_ckt, quant_ckt, cur_ckt, quant_para, quant_min_scale = [cur_model.state_dict() for _ in
                                                                    range(5)]  # keys encoder are here.
        encoder_k_list = []
        for k, v in cur_ckt.items():
            if 'encoder' in k:
                encoder_k_list.append(k)
            else:
                quant_v, new_v = quant_tensor(v, args.quant_model_bit)
                quant_ckt[k] = quant_v
                cur_ckt[k] = new_v.to(device)
                quant_para[k] = quant_v['quant']
                quant_min_scale[k] = {'min': quant_v['min'], 'scale': quant_v['scale']}
        cur_model.load_state_dict(cur_ckt)
        model_list.append(cur_model)
        quanted_model.load_state_dict(quant_para)
        for encoder_k in encoder_k_list:
            del quant_ckt[encoder_k]
            del quant_para[encoder_k]
            del quant_min_scale[encoder_k]
            del raw_ckt[encoder_k]
        return model_list, quant_ckt, quanted_model, quant_min_scale, raw_ckt


if __name__ == '__main__':
    main()
