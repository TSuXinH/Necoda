import os

for cur_epoch in [20]:
    gen_id_list = '1 2 3'
    base_path = r'\\as13000.com\slfm\xxh\others\proj_dc\exp\nerp_st_mode_interp\s_gen\clean\nerp_st_B2_E100_chns32,32,32_Q_M8_E6_lr0.0002_L2_mean_max_ts64tt1ss1st64_interp_chn16'
    os.system('python recon_nerp_st_mode_interp.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))

