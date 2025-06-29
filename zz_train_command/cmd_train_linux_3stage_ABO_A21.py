import os

# id_abo = '503772253'
# data_path = '/mnt/nas00/YZ_personal_storage/DeepCompress/Database/ABO/{}/movie/00000.tif'.format(id_abo)
# result_root = '/mnt/nas00/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_st_3stage/ABO/{}'.format(id_abo)
# gen_data_path_all = ''
# for idx in range(1, 20):
#     idx_str = str(idx).zfill(2)
#     gen_data_path_all += '/mnt/nas00/YZ_personal_storage/DeepCompress/Database/ABO/{}/movie/000{}.tif '.format(id_abo, idx_str)

cuda_id = 7
id_abo = '672214923'
data_path = '/mnt/nas00/YZ_personal_storage/DeepCompress/Database/ABO/ABO_new/{}/movie/{}_tiff_1.tif'.format(id_abo, id_abo)
result_root = '/mnt/nas00/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_st_3stage/ABO/{}'.format(id_abo)
gen_data_path_all = ''
for idx in range(2, 20):
    idx_str = str(idx).zfill(2)
    gen_data_path_all += '/mnt/nas00/YZ_personal_storage/DeepCompress/Database/ABO/ABO_new/{}/movie/{}_tiff_{}.tif '.format(id_abo, id_abo, idx)


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
t_t = '2 1 1'
s_s = '1 1 1'
s_t = '4 4 2'
chns_list = '32 32 32'
pre_norm = 'mean_std'

for loss in ['L2']:
    for e in [100]:
        for b in [2]:
            for ef in [10]:
                for lr in [2e-4]:
                    os.system("CUDA_VISIBLE_DEVICES={} python train_3stage.py --pre_norm {} --output_path {} --data_path {} \
                    --act gelu --norm none --pre_s_rate {} --pre_t_rate {} --s_emb_dim {} --t_emb_dim {} \
                    --s_s_rate_list {} --t_s_rate_list {} --s_t_rate_list {} --t_t_rate_list {} \
                    --loss {} --model_type {} -e {} --eval_freq {} -b {} --lr {} --overwrite \
                    --chns_list {} -g {} --lam_perceptual {}  --selected_perceptual_layer {} \
                    --lam_temporal {}  --lam_spatial {} --quant_embed_bit 4 --remark interp8 --interp_size_x 8 \
                    --interp_size_t 8 --epoch_stage1 30 --epoch_stage2 50 --remark s1_30_s2_50 \
                    ".format(cuda_id, pre_norm, result_root, data_path, pre_s_rate, pre_t_rate, s_emb_dim, t_emb_dim,
                             s_s, t_s, s_t, t_t, loss, model_type, e, ef, b, lr, chns_list, gen_data_path_all,
                             lam_perceptual, selected_perceptual_layer, lam_temporal, lam_spatial))

