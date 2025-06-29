import os

cur_epoch = 10
base_path = r'C:\Users\BBNC\Desktop\tmp_xxh\others\proj_dc\exp\nerp_st\s_gen\1\clean\nerp_st_B2_E100_chns32,32,32_Q_M8_E6_lr0.0005_L2_mean_max_ts32tt4ss4st32_train_w_overlap'
os.system('python recon_nerp_st.py -d {} -e {} --name recon_{}'.format(base_path, cur_epoch, cur_epoch))
