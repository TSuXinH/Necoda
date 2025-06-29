import os

data_path = '/slfm/xxh/others/proj_dc/data/s_gen/1/Fsim_clean.tiff'
result_root = '/slfm/xxh/others/proj_dc/exp/nerp_st_huff/s_gen/clean_ds'
gen_data_path1 = '/slfm/xxh/others/proj_dc/data/s_gen/2/Fsim_clean.tiff'
gen_data_path2 = '/slfm/xxh/others/proj_dc/data/fsim1/Fsim_clean.tiff'
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
t_t = '1 1 1'
s_s = '1 1 1'
s_t = '4 4 2'
chns_list = '32 32 32'

for loss in ['L2']:
    for e in [100]:
        for b in [2]:
            for ef in [10]:
                for lr in [2e-4]:
                    os.system("python train_3stage.py --pre_norm mean_max --output_path {} --data_path {} \
                    --act gelu --norm none --pre_s_rate {} --pre_t_rate {} --s_emb_dim {} --t_emb_dim {} \
                    --s_s_rate_list {} --t_s_rate_list {} --s_t_rate_list {} --t_t_rate_list {} \
                    --loss {} --model_type {} -e {} --eval_freq {} -b {} --lr {} --overwrite \
                    --chns_list {} -g {} --lam_perceptual {}  --selected_perceptual_layer {} \
                    --lam_temporal {}  --lam_spatial {} --quant_embed_bit 4 --remark interp8 --interp_size_x 8 \
                    --interp_size_t 8 --epoch_stage1 40 --epoch_stage2 40 --remark s1_40_s2_40_test_speed \
                    ".format(result_root, data_path, pre_s_rate, pre_t_rate, s_emb_dim, t_emb_dim,
                             s_s, t_s, s_t, t_t, loss, model_type, e, ef, b, lr, chns_list, gen_data_path_all,
                             lam_perceptual, selected_perceptual_layer, lam_temporal, lam_spatial))
