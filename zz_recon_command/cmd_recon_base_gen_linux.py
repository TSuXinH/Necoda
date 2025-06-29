import os

for cur_epoch in [70]:
    gen_id_list = '1 2 3'
    base_path = r'C:\Users\BBNC\Desktop\tmp_xxh\others\proj_dc\exp\nerp_st_pro\s_gen\clean\nerp_base_B2_E100_chns32,32,32_Q_M8_E6_lr0.0002_L2_mean_max_s8t8'
    os.system('python recon_nerp_base_gen.py -d {} -e {} --name recon_{} -g {}'.format(base_path, cur_epoch, cur_epoch, gen_id_list))
