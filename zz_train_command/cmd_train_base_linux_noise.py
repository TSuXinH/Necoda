import os


data_path = '/mnt/nas/YZ_personal_storage/DeepCompress/simulation_generalization/1/Fsim.tiff'
result_root = '/mnt/nas/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_base/noise/s_gen'
gen_data_path_all = '/mnt/nas/YZ_personal_storage/DeepCompress/simulation_generalization/2/Fsim.tiff '
gen_data_path_all += '/mnt/nas/YZ_personal_storage/xxh/proj/proj_dc/data/fsim1/Fsim.tiff '

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
                os.system("python train_base_new.py --pre_norm min_max --output_path {} --data_path {} \
                --act gelu --norm none --pre_s_rate {} --pre_t_rate {} --emb_dim {}  \
                --s_rate_list {} --t_rate_list {} \
                --model_type {} -e {} --eval_freq {} -b {} --lr {} --overwrite \
                --chns_list {} -g {} --lam_perceptual {}  --selected_perceptual_layer {} \
                --lam_temporal {}  --lam_spatial {} --quant_embed_bit 4 \
                ".format(result_root, data_path, pre_s_rate, pre_t_rate, emb_dim,
                         s_rate, t_rate, model_type, e, ef, b, lr, chns_list, gen_data_path_all,
                         lam_perceptual, selected_perceptual_layer, lam_perceptual, lam_temporal, lam_spatial))