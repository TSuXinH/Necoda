import os

cuda_id = 0
for cur_epoch in [60]:
    gen_id_list = '1 2'
    base_path = './result'
    os.system('CUDA_VISIBLE_DEVICES={} python recon_nerp_st.py -d {} -e {} --name recon_{} -g {}'.format(cuda_id, base_path, cur_epoch, cur_epoch, gen_id_list))
    
