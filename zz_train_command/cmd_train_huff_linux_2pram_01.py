import os

data_path = '/slfm/xxh/others/proj_dc/data/2pram/01_0/movie/01_0.tif'
result_root = '/slfm/xxh/others/proj_dc/exp/nerp_st_huff/2pram/01'
gen_data_path_all = ''
for idx in range(1, 4):
    gen_data_path_all += '/slfm/xxh/others/proj_dc/data/2pram/01_{}/movie/01_{}.tif '.format(idx, idx)

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
                    --lam_temporal {}  --lam_spatial {} --quant_embed_bit 4 \
                    ".format(result_root, data_path, pre_s_rate, pre_t_rate, s_emb_dim, t_emb_dim,
                             s_s, t_s, s_t, t_t, loss, model_type, e, ef, b, lr, chns_list, gen_data_path_all,
                             lam_perceptual, selected_perceptual_layer, lam_temporal, lam_spatial))
