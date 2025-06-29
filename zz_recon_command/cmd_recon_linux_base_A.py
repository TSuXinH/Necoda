import os

for cur_epoch in [60]:
    # gen_id_list = '1 2 3 4 5 6'
    gen_id_list = '1 2 3'
    # gen_id_list = '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19'
    base_path = '/mnt/nas00/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_base/noise/s_gen/group7_12_group6_1/nerp_base_B2_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_std_s4t4'
    # base_path = '/slfm/xxh/others/proj_dc/exp/nerp_st_huff/neurofinder/nerp_st_B2_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_max_ts64tt1ss1st64_interp8'
    os.system('CUDA_VISIBLE_DEVICES=4 python recon_nerp_base.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))


    # base_path = '/slfm/xxh/others/proj_dc/exp/nerp_st_huff/voltage/v2p1/nerp_st_B2_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_max_ts64tt1ss1st64_c3'
    # os.system('python recon_nerp_st_huff.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))
