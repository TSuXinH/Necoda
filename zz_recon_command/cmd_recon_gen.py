import os

for cur_epoch in [100]:
    gen_id_list = '4'
    base_path = r'Z:\xxh\others\proj_dc\exp\nerp_st\simulation_gen\clean\nerp_st_B2_E100_chns32,32,32_Q_M8_E6_lr0.0003_L2_mean_max_ts32tt2ss32st32'
    os.system('python recon_nerp_st_gen.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))
