import os

data_path = '/slfm/xxh/others/proj_dc/data/s_gen/1/Fsim.tiff'
result_root = '/slfm/xxh/others/proj_dc/exp/nerp_st_pro/s_gen/noise'
gen_data_path1 = '/slfm/xxh/others/proj_dc/data/s_gen/2/Fsim.tiff'
gen_data_path2 = '/slfm/xxh/others/proj_dc/data/fsim1/Fsim.tiff'
gen_data_path_all = gen_data_path1 + ' ' + gen_data_path2
model_type = 'nerp_st'
pre_s_rate = 2
pre_t_rate = 2
s_emb_dim = 4
t_emb_dim = 4
lam_perceptual = .0
lam_temporal = .0
lam_spatial = .0
selected_perceptual_layer = '3 8'
t_s = '4 4 2'
t_t = '2 2 1'
s_s = '2 1 1'
s_t = '4 4 2'
chns_list = '32 32 32'


for loss in ['L2']:
    for e in [100]:
        for b in [2]:
            for ef in [10]:
                for lr in [3e-4]:
                    os.system("python train_pro.py --pre_norm mean_max --output_path {} --data_path {} \
                    --act gelu --norm none --pre_s_rate {} --pre_t_rate {} --s_emb_dim {} --t_emb_dim {} \
                    --s_s_rate_list {} --t_s_rate_list {} --s_t_rate_list {} --t_t_rate_list {} \
                    --loss {} --model_type {} -e {} --eval_freq {} -b {} --lr {} --overwrite \
                    --chns_list {} -g {} --lam_perceptual {}  --selected_perceptual_layer {} \
                    --lam_temporal {}  --lam_spatial {} \
                    ".format(result_root, data_path, pre_s_rate, pre_t_rate, s_emb_dim, t_emb_dim,
                             s_s, t_s, s_t, t_t, loss, model_type, e, ef, b, lr, chns_list, gen_data_path_all,
                             lam_perceptual, selected_perceptual_layer, lam_temporal, lam_spatial))
