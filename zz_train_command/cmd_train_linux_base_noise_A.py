import os


data_path = '/mnt/nas00/YZ_personal_storage/DeepCompress/Database/NAOMi/NAOMi_2p_7/NA_0.80_Hz_30_D_0_pow_40/1/Fsim.tiff'
gen_data_path_all = '/mnt/nas00/YZ_personal_storage/DeepCompress/Database/NAOMi/NAOMi_2p_7/NA_0.80_Hz_30_D_0_pow_40/2/Fsim.tiff '
gen_data_path_all += '/mnt/nas00/YZ_personal_storage/DeepCompress/Database/NAOMi/NAOMi_2p_6/NA_0.80_Hz_30_D_0_pow_40/1/Fsim.tiff '
result_root = '/mnt/nas00/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_base/noise/s_gen/group7_12_group6_1'

model_type = 'nerp_base'
pre_s_rate = 2
pre_t_rate = 2
emb_dim = 3
lam_perceptual = .0
lam_temporal = .0
lam_spatial = .0
selected_perceptual_layer = '3 8'
t_rate = '2 2 1'
s_rate = '2 2 1'
chns_list = '32 32 32'

for e in [100]:
    for b in [2]:
        for ef in [10]:
            for lr in [2e-4]:
                os.system("CUDA_VISIBLE_DEVICES=1 python train_base_new.py --pre_norm mean_std --output_path {} --data_path {} \
                --act gelu --norm none --pre_s_rate {} --pre_t_rate {} --emb_dim {}  \
                --s_rate_list {} --t_rate_list {} \
                --model_type {} -e {} --eval_freq {} -b {} --lr {} --overwrite \
                --chns_list {} -g {} --lam_perceptual {}  --selected_perceptual_layer {} \
                --lam_temporal {}  --lam_spatial {} --quant_embed_bit 4 \
                ".format(result_root, data_path, pre_s_rate, pre_t_rate, emb_dim,
                         s_rate, t_rate, model_type, e, ef, b, lr, chns_list, gen_data_path_all,
                         lam_perceptual, selected_perceptual_layer, lam_perceptual, lam_temporal, lam_spatial))