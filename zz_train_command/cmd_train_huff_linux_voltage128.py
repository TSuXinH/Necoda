import os
import sys

base_data_path = '/slfm/xxh/others/proj_dc/data/voltage/Voltron2_part1'
result_root = '/slfm/xxh/others/proj_dc/exp/nerp_st_huff/voltage/v2p1'
dir_list = os.listdir(base_data_path)
abs_path_list = []
for item in dir_list:
    abs_path_list.append(os.path.join(base_data_path, item, 'dat_crop128.tif'))
sorted_abs_path_list = sorted(abs_path_list)
data_path = sorted_abs_path_list[2]
gen_data_path_all = sorted_abs_path_list[0] + ' ' + sorted_abs_path_list[1]
for idx in range(3, len(sorted_abs_path_list)):
    gen_data_path_all += ' '
    gen_data_path_all += sorted_abs_path_list[idx]


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
                    --chns_list {} -g {} --lam_perceptual {}  --selected_perceptual_layer {} \
                    --lam_temporal {}  --lam_spatial {} --quant_embed_bit 4 --interp_size_x 0 --interp_size_t 6 --remark c3 \
                    ".format(result_root, data_path, pre_s_rate, pre_t_rate, s_emb_dim, t_emb_dim,
                             s_s, t_s, s_t, t_t, loss, model_type, e, ef, b, lr, chns_list, gen_data_path_all,
                             lam_perceptual, selected_perceptual_layer, lam_temporal, lam_spatial))


# --patch_x 32 --patch_t 128 --gap_x 32 --gap_t 64 --remark px64pt128gx32gt64interp6start1 \
