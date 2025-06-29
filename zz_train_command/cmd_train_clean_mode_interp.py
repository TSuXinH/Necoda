import os

data_path = r'C:\Users\BBNC\Desktop\tmp_xxh\others\proj_dc\data\s_gen\1\Fsim_clean.tiff'
result_root = r'C:\Users\BBNC\Desktop\tmp_xxh\others\proj_dc\exp\nerp_st_mode_interp\s_gen\clean'
gen_data_path1 = r'C:\Users\BBNC\Desktop\tmp_xxh\others\proj_dc\data\s_gen\2\Fsim_clean.tiff'
gen_data_path2 = r'C:\Users\BBNC\Desktop\tmp_xxh\others\proj_dc\data\s5_raw\Fsim_clean.tiff'
gen_data_path_all = gen_data_path1 + ' ' + gen_data_path2
model_type = 'nerp_st'
pre_s_rate = 2
pre_t_rate = 2
s_emb_dim = 1
t_emb_dim = 1
lam_perceptual = .0
lam_temporal = .0
lam_spatial = .0
selected_perceptual_layer = '3 8'
t_s = '4 4 2'
t_t = '2 2 1'
s_s = '2 2 1'
s_t = '4 4 2'
chns_list = '16 16 16'

for e in [1]:
    for b in [2]:
        for ef in [1]:
            for lr in [2e-4]:
                os.system("python train_mode_interp.py --pre_norm mean_max --output_path {} --data_path {} \
                --act gelu --norm none --pre_s_rate {} --pre_t_rate {} --s_emb_dim {} --t_emb_dim {} \
                --s_s_rate_list {} --t_s_rate_list {} --s_t_rate_list {} --t_t_rate_list {} \
                --model_type {} -e {} --eval_freq {} -b {} --lr {} --overwrite \
                --chns_list {} -g {} --lam_perceptual {}  --selected_perceptual_layer {} --interp_size_x 2 --interp_size_t 2 --interp_chn 16 \
                --lam_temporal {}  --lam_spatial {}  --interp_method conv --remark stemb1_interpxt2 \
                ".format(result_root, data_path, pre_s_rate, pre_t_rate, s_emb_dim, t_emb_dim,
                         s_s, t_s, s_t, t_t, model_type, e, ef, b, lr, chns_list, gen_data_path_all,
                         lam_perceptual, selected_perceptual_layer, lam_perceptual, lam_temporal, lam_spatial))

