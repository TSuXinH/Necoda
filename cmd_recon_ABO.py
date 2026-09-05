import argparse
import os
import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--cuda-id", type=int, default=0)
args = parser.parse_args()

code_root = Path(__file__).resolve().parent
cuda_id = args.cuda_id
epoch = 100
generalization_id_list = [1, 2]
experiment_dir = "./result/name"

command = [
    sys.executable,
    "-u",
    str(code_root / "recon.py"),
    "-d",
    str(experiment_dir),
    "-e",
    str(epoch),
    "--name",
    f"recon_{epoch}",
    "-g",
    *(str(value) for value in generalization_id_list),
]

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
env["PYTHONUNBUFFERED"] = "1"
subprocess.run(command, cwd=code_root, env=env, check=True)
