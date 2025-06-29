import os

data_path = '/slfm/xxh/others/proj_dc/data/neurofinder/neurofinder.00.00/movie/00000.tif'
result_root = '/slfm/xxh/others/proj_dc/exp/nerp_st_huff/neurofinder'
gen_data_path_all = ''
# for item in [100, 140, 160, 180, 200]:
#     gen_data_path_all += '/slfm/xxh/others/proj_dc/data/148_hi_low_noise/{}umdepth_0.3power/movie_gt/00000.tif '.format(item)
for idx in range(1, 12):
    gen_data_path_all += '/slfm/xxh/others/proj_dc/data/neurofinder/neurofinder.00.{}/movie/00000.tif '.format(str(idx).zfill(2))
for idx1 in range(1, 3):
    for idx2 in range(2):
        gen_data_path_all += '/slfm/xxh/others/proj_dc/data/neurofinder/neurofinder.{}.{}/movie/00000.tif '.format(str(idx1).zfill(2), str(idx2).zfill(2))
gen_data_path_all += '/slfm/xxh/others/proj_dc/data/neurofinder/neurofinder.03.00/movie/00000.tif '
gen_data_path_all += '/slfm/xxh/others/proj_dc/data/neurofinder/neurofinder.04.00/movie/00000.tif '
gen_data_path_all += '/slfm/xxh/others/proj_dc/data/neurofinder/neurofinder.04.01/movie/00000.tif '

# gen_data_path_all = '/slfm/xxh/others/proj_dc/data/148_hi_low_noise/200umdepth_0.4power/movie_gt/00000.tif '

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
                    --lam_temporal {}  --lam_spatial {} --quant_embed_bit 4 --interp_size_x 8 --interp_size_t 8 --remark interp8 -g {} \
                    ".format(result_root, data_path, pre_s_rate, pre_t_rate, s_emb_dim, t_emb_dim,
                             s_s, t_s, s_t, t_t, loss, model_type, e, ef, b, lr, chns_list,
                             lam_perceptual, selected_perceptual_layer, lam_temporal, lam_spatial, gen_data_path_all))
