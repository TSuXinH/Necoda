import os
import sys
import numpy as np

base_path = '/slfm/xxh/others/proj_dc/data/Zebrafish'
result_root = '/slfm/xxh/others/proj_dc/exp/nerp_st_huff/zebrafish'
gen_data_path_all = ''
dir_list = os.listdir(base_path)
dir_list = [item for item in dir_list if '2019' not in item]
dir_list.remove('zebrafish_Habenula.tif')
dir_list = sorted(dir_list)
print(dir_list)
data_path = os.path.join(base_path, dir_list[5])
for item in dir_list[:-1]:
    gen_data_path_all += os.path.join(base_path, item)
    gen_data_path_all += ' '

# print(data_path)
# print(gen_data_path_all)
# sys.exit()

model_type = 'nerp_st'
pre_s_rate = 2
pre_t_rate = 2
s_emb_dim = 2
t_emb_dim = 2
lam_perceptual = .0
lam_temporal = .0
lam_spatial = .0
selected_perceptual_layer = '3 8'
t_s = '4 4 4'
t_t = '1 1 1'
s_s = '1 1 1'
s_t = '4 4 4'
chns_list = '32 32 32'


for loss in ['L2']:
    for e in [100]:
        for b in [2]:
            for ef in [10]:
                for lr in [2e-4]:
                    os.system("python train_huff.py --pre_norm mean_max --output_path {} --data_path {} \
                    --act gelu --norm none --pre_s_rate {} --pre_t_rate {} --s_emb_dim {} --t_emb_dim {} \
                    --s_s_rate_list {} --t_s_rate_list {} --s_t_rate_list {} --t_t_rate_list {} \
                    --loss {} --model_type {} -e {} --eval_freq {} -b {} --lr {} --overwrite \
                    --chns_list {} --lam_perceptual {}  --selected_perceptual_layer {} \
                    --lam_temporal {}  --lam_spatial {} --quant_embed_bit 4 --interp_size_x 8 --interp_size_t 8 -g {} --remark interp8_stemb4_Mfirst_test \
                    ".format(result_root, data_path, pre_s_rate, pre_t_rate, s_emb_dim, t_emb_dim,
                             s_s, t_s, s_t, t_t, loss, model_type, e, ef, b, lr, chns_list,
                             lam_perceptual, selected_perceptual_layer, lam_temporal, lam_spatial, gen_data_path_all))
