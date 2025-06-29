import os

x_resize = 472
y_resize = 463
# x_resize = 477
# y_resize = 458
for cur_epoch in [60]:
    gen_id_list = '1'
    # gen_id_list = '7 8 9 10 11 12 13 14 15'
    # gen_id_list = '1 2 3'
    # gen_id_list = '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21'
    # base_path = '/mnt/nas00/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_st_3stage/ABO/ABO_prev1/709948912/nerp_st_B2_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_std_ts32tt2ss1st32_s1_30_s2_50'
    # os.system('CUDA_VISIBLE_DEVICES=2 python recon_nerp_st_huff.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))

    base_path = '/mnt/nas00/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_st_3stage/motion/trial_test4_ras/nerp_st_B2_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_std_ts32tt2ss1st32_s1_30_s2_50'
    os.system('CUDA_VISIBLE_DEVICES=2 python recon_nerp_st_huff.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))


    # base_path = '/mnt/nas00/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_st_3stage/ephys/GCaMP6s_9cells_Chen2013/20120417_cell3/nerp_st_B2_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_std_ts16tt2ss2st16_s1_30_s2_50_s1t2'
    # os.system('CUDA_VISIBLE_DEVICES=3 python recon_nerp_st_huff.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))

    # base_path = '/mnt/nas00/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_st_3stage/zebrafish/OT/nerp_st_B2_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_std_ts32tt2ss1st32_s1_30_s2_50'
    # os.system('CUDA_VISIBLE_DEVICES=0 python recon_nerp_st_huff.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))

    # base_path = '/mnt/nas00/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_st_3stage/spine/02/nerp_st_B2_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_std_ts32tt1ss1st32_s1_30_s2_50'
    # os.system('CUDA_VISIBLE_DEVICES=0 python recon_nerp_st_huff.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))

    # base_path = '/mnt/nas00/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_st_3stage/fly/nerp_st_B2_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_std_ts25tt1ss1st25_s1_30_s2_50'
    # os.system('CUDA_VISIBLE_DEVICES=0 python recon_nerp_st_huff.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))

    # base_path = '/mnt/nas00/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_st_3stage/caiman/images_J115_file/nerp_st_B2_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_std_ts25tt1ss1st25_s1_30_s2_50'
    # os.system('CUDA_VISIBLE_DEVICES=3 python recon_nerp_st_huff.py -d {} -e {} --name recon_{} -g {} --do_resize --x_resize {} --y_resize {} '.
    #           format(base_path, cur_epoch, cur_epoch, gen_id_list, x_resize, y_resize))

    # base_path = '/mnt/nas00/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_st_huff/s_gen/group8_12_group6_2/nerp_st_B2_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_std_ts64tt1ss1st64_interp4_in'
    # os.system('CUDA_VISIBLE_DEVICES=4 python recon_nerp_st_huff.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))

    # base_path = '/mnt/nas00/YZ_personal_storage/xxh/proj/proj_dc/exp/nerp_base/noise/s_gen/group7_12_group6_1/nerp_base_B2_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_std_s4t4'
    # os.system('CUDA_VISIBLE_DEVICES=3 python recon_nerp_st_huff.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))
