import os

data_path = r'C:\Users\BBNC\Desktop\tmp_xxh\others\proj_dc\data\s_gen\1\Fsim_clean.tiff'
result_root = r'C:\Users\BBNC\Desktop\tmp_xxh\others\proj_dc\exp\nerp_base\s_gen\clean'
gen_data_path1 = r'C:\Users\BBNC\Desktop\tmp_xxh\others\proj_dc\data\s_gen\2\Fsim_clean.tiff'
gen_data_path2 = r'C:\Users\BBNC\Desktop\tmp_xxh\others\proj_dc\data\s5_raw\Fsim_clean.tiff'
gen_data_path_all = gen_data_path1 + ' ' + gen_data_path2
model_type = 'nerp_base'
pre_s_rate = 2
pre_t_rate = 2
emb_dim = 4
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
                os.system("python train_base.py --pre_norm mean_max --output_path {} --data_path {} \
                --act gelu --norm none --pre_s_rate {} --pre_t_rate {} --emb_dim {}  \
                --s_rate_list {} --t_rate_list {} \
                --model_type {} -e {} --eval_freq {} -b {} --lr {} --overwrite \
                --chns_list {} -g {} --lam_perceptual {}  --selected_perceptual_layer {} \
                --lam_temporal {}  --lam_spatial {}  \
                ".format(result_root, data_path, pre_s_rate, pre_t_rate, emb_dim,
                         s_rate, t_rate, model_type, e, ef, b, lr, chns_list, gen_data_path_all,
                         lam_perceptual, selected_perceptual_layer, lam_perceptual, lam_temporal, lam_spatial))

