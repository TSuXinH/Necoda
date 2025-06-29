import os

for cur_epoch in [40]:
    gen_id_list = '1 2 3'
    base_path = r'\\as13000.com\slfm\xxh\others\proj_dc\exp\nerp_st_huff\s_gen\clean\nerp_st_B16_E100_chns32,32,32_Q_M8_E4_lr0.0002_L2_mean_max_ts32tt1ss1st32_p32gap16interp6'
    os.system('python recon_nerp_st_huff.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))


