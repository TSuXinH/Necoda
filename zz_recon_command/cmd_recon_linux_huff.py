import os

for cur_epoch in [60]:
    # gen_id_list = '1 2 3 4 5 6'
    # gen_id_list = '1 2 3 4 5 6'
    # gen_id_list = '1 2'
    gen_id_list = '1 2 3'
    base_path = '/mnt/nas/YZ_personal_storage/DeepCompress/xxh/proj/proj_dc/exp/nerp_st_huff/neurofinder/nerp_st_B2_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_max_ts32tt1ss1st32_s1_20_s2_100'
    # base_path = '/slfm/xxh/others/proj_dc/exp/nerp_st_huff/neurofinder/nerp_st_B2_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_max_ts64tt1ss1st64_interp8'
    os.system('python recon_nerp_st_huff.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))


    # base_path = '/slfm/xxh/others/proj_dc/exp/nerp_st_huff/voltage/v2p1/nerp_st_B2_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_max_ts64tt1ss1st64_c3'
    # os.system('python recon_nerp_st_huff.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))
