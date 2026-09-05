import argparse
import os
import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--cuda-id", type=int, default=1)
args = parser.parse_args()

code_root = Path(__file__).resolve().parent
cuda_id = args.cuda_id
id_abo = "642877968"
data_path = f"./data/{id_abo}_tiff_1.tif"
result_root = f"./result/{id_abo}"
generalization_data_paths = [f"./data/{id_abo}_tiff_2.tif"]
pre_norm = "robust_min_max"
robust_low = 0.001
robust_high = 99.99
patch_x = 128
patch_t = 128
gap_x = 64
gap_t = 64
interp_size_x = 4
interp_size_t = 4
interp_chn = 32
pre_s_rate = 2
pre_t_rate = 2
emb_dim = 4
s_rate_list = [1, 1, 1]
t_rate_list = [2, 2, 2]
chns_list = [32, 32, 32]
act = "gelu"
checkerboard_kernel = 5
latent_gain_init = 4
latent_gain_min = 0.25
latent_gain_max = 64
branch_level = 1
temporal_channels = 8
temporal_pool_h = 16
temporal_pool_w = 16
spatial_stat_channels = 1
lam = 1.10
rd_metric = "normalized-mse"
lam_temporal = 0.35
distortion_only_epochs = 5
rate_warmup_epochs = 10
lr = 2e-4
aux_lr = 1e-4
epochs = 100
batch_size = 2
workers = 2
eval_freq = 10
print_freq = 50
remark = "ABO_lam1p10_tl0p35"
overwrite = True

command = [
    sys.executable,
    "-u",
    str(code_root / "train.py"),
    "--data_path",
    data_path,
    "--output_path",
    result_root,
    "-g",
    *generalization_data_paths,
    "--pre_norm",
    pre_norm,
    "--robust_low",
    str(robust_low),
    "--robust_high",
    str(robust_high),
    "--patch_x",
    str(patch_x),
    "--patch_t",
    str(patch_t),
    "--gap_x",
    str(gap_x),
    "--gap_t",
    str(gap_t),
    "--interp_size_x",
    str(interp_size_x),
    "--interp_size_t",
    str(interp_size_t),
    "--interp_chn",
    str(interp_chn),
    "--pre_s_rate",
    str(pre_s_rate),
    "--pre_t_rate",
    str(pre_t_rate),
    "--emb_dim",
    str(emb_dim),
    "--s_rate_list",
    *(str(value) for value in s_rate_list),
    "--t_rate_list",
    *(str(value) for value in t_rate_list),
    "--chns_list",
    *(str(value) for value in chns_list),
    "--act",
    act,
    "--checkerboard_kernel",
    str(checkerboard_kernel),
    "--latent_gain_init",
    str(latent_gain_init),
    "--latent_gain_min",
    str(latent_gain_min),
    "--latent_gain_max",
    str(latent_gain_max),
    "--branch_level",
    str(branch_level),
    "--temporal_channels",
    str(temporal_channels),
    "--temporal_pool_h",
    str(temporal_pool_h),
    "--temporal_pool_w",
    str(temporal_pool_w),
    "--spatial_stat_channels",
    str(spatial_stat_channels),
    "--lam",
    str(lam),
    "--rd_metric",
    rd_metric,
    "--lam_temporal",
    str(lam_temporal),
    "--distortion_only_epochs",
    str(distortion_only_epochs),
    "--rate_warmup_epochs",
    str(rate_warmup_epochs),
    "--lr",
    str(lr),
    "--aux_lr",
    str(aux_lr),
    "--epochs",
    str(epochs),
    "--batchSize",
    str(batch_size),
    "--workers",
    str(workers),
    "--eval_freq",
    str(eval_freq),
    "--print_freq",
    str(print_freq),
    "--remark",
    remark,
]
if overwrite:
    command.append("--overwrite")

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
env["PYTHONUNBUFFERED"] = "1"
subprocess.run(command, cwd=code_root, env=env, check=True)
