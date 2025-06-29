import os

# id_abo = '503772253'
# data_path = '/mnt/nas00/YZ_personal_storage/DeepCompress/Database/ABO/{}/movie/00000.tif'.format(id_abo)
# result_root = '/mnt/nas00/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_st_3stage/ABO/{}'.format(id_abo)
# gen_data_path_all = ''
# for idx in range(1, 20):
#     idx_str = str(idx).zfill(2)
#     gen_data_path_all += '/mnt/nas00/YZ_personal_storage/DeepCompress/Database/ABO/{}/movie/000{}.tif '.format(id_abo, idx_str)

name = 'images_J115_file'
cuda_id = 1
data_path = '/mnt/nas00/YZ_personal_storage/DeepCompress/Database/caiman/{}/000.tif'.format(name)
result_root = '/mnt/nas00/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_st_3stage/caiman/{}'.format(name)
gen_data_path_all = ''
for i in range(1, 15):
    gen_data_path_all += '/mnt/nas00/YZ_personal_storage/DeepCompress/Database/caiman/{}/{}.tif '.format(name, str(i).zfill(3))


model_type = 'nerp_st'
pre_s_rate = 3
pre_t_rate = 3
s_emb_dim = 1
t_emb_dim = 1
lam_perceptual = .0
lam_temporal = .0
lam_spatial = .0
selected_perceptual_layer = '3 8'
t_s = '5 5 1'
t_t = '1 1 1'
s_s = '1 1 1'
s_t = '5 5 1'
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
                    --lam_temporal {}  --lam_spatial {} --quant_embed_bit 4 --interp_size_x 5 \
                    --patch_x 150 --patch_t 150 --gap_x 75 --gap_t 75 --do_resize True --x_resize 450 --y_resize 450 \
                    --interp_size_t 5 --epoch_stage1 30 --epoch_stage2 50 --remark s1_30_s2_50 \
                    ".format(cuda_id, pre_norm, result_root, data_path, pre_s_rate, pre_t_rate, s_emb_dim, t_emb_dim,
                             s_s, t_s, s_t, t_t, loss, model_type, e, ef, b, lr, chns_list, gen_data_path_all,
                             lam_perceptual, selected_perceptual_layer, lam_temporal, lam_spatial))

