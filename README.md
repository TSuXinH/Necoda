# Necoda: High-Fidelity Functional Neural Data Compression via Neural Representation Enhances Data Sharing

<img src="fig/overview1.png" width="800" align="middle">

## [Project page](https://tsuxinh.github.io/Necoda/)

## Contents

- [Overview](#overview)
- [Directory structure](#directory-structure)
- [Pytorch code](#pytorch-code)
- [Results](#results)
- [Citation](#citation)

## Overview

<img src="fig/overview2.png" width="400" align="right">

In neuroscience, progress hinges on sharing large-scale data. Sharing comprehensive, raw-like imaging data is particularly crucial for transparency, reproducibility, and developing new analysis tools.

The primary obstacle is data size. Functional imaging datasets often reach the terabyte-scale, creating a massive bottleneck for data sharing, storage, and reuse. These logistical hurdles severely slow the pace of discovery and hinder the reproduction of scientific findings.


To solve this, we developed **Necoda, a deep learning method that compresses functional imaging data by over 1,000-fold while maintaining high fidelity**. Our method uses a content-adaptive encoder-decoder network that leverages the inherent spatiotemporal structure of neural recordings. A key innovation is Necoda's ability to generalize across unseen data; we can train it once on a small data subset and then apply it to compress numerous other experiments, making it a practical and efficient tool.

We demonstrated that **Necoda preserves essential scientific information while drastically reducing file size**. We validated our method on diverse datasets, including data from different species, brain regions, and imaging modalities. In our key proof-of-concept, we compressed a 4.82 TB ABO dataset to just 4.81 GB. Using only this compact file, we fully reproduced the published findings of a major neuroscience study in just a few hours. 

By removing the data-sharing bottleneck, we believe Necoda will significantly accelerate discovery and enhance reproducibility in neuroscience.

## Directory structure

## Pytorch code 

### Environment 

* Ubuntu 22.04.5
* Python 3.13.2
* Pytorch 2.6.0 + CUDA 12.4
* NVIDIA A100-SXM4 GPU (40 GB memory) 

### Code setup


* Create a virtual environment and install Pytorch. Please select the correct Pytorch version that matches your CUDA version at https://pytorch.org/get-started/previous-versions/.

```
$ conda create -n necoda python=3.13
$ conda activate necoda
$ pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 pybind11 --extra-index-url https://download.pytorch.org/whl/cu124
```

* Clone the repository here.

```
$ git clone git@github.com:TSuXinH/Necoda.git
$ cd Necoda
```

* Install other necessary dependencies.

```
$ pip install --no-build-isolation -r requirements.txt
```


### Training

For the dataset with a spatial shape of 512x512 and temporal shape of 6000, the following command can be used for standard training setup. This will generate entropy-coded bitstream files named `c_dict_<epoch>.pth`. The detailed meaning of each argument can be found in the training script.

To speed up training, `train_2stages.py` can be used to enable two-stage training strategy.

ABO datasets, as the training data, can be downloaded from https://drive.google.com/drive/folders/1bAsTiy0aMIoUEjw9QJhIm0PCqTSuXoR8.
```
CUDA_VISIBLE_DEVICES=0 python train.py --output_path {} --data_path {} \
                    --pre_norm robust_min_max --rd_metric normalized-mse \
                    --patch_x 128 --patch_t 128 --gap_x 64 --gap_t 64 \
                    --interp_size_x 4 --interp_size_t 4 \
                    --s_rate_list 1 1 1 --t_rate_list 2 2 2 --chns_list 32 32 32 \
                    --lam 2.2 --lam_temporal 0.35 --temporal_channels 8 \
                    -e 100 -b 2 -j 2 --lr 2e-4 --eval_freq 10 --overwrite \
                    -g {} --remark {}
``` 

### Inference
The following command can be used to decompress the bitstreams with the trained network.
```
CUDA_VISIBLE_DEVICES=0 python recon.py -d {} -e {} --name recon_{} -g {}
```

Please refer to the py files with prefix **cmd** (command) for more details or use them directly.

### Optional

The nwb extension for Necoda and 7z compression is further provided in scripts `necoda_nwb_encode_ABO.py` and `necoda_nwb_decode_ABO.py`.

```
python necoda_nwb_encode_ABO.py --name {} --base-path {} --epoch {} \
                               --embedding-number {} --original-data-size {} \
                               --output-dir {}

python necoda_nwb_decode_ABO.py --nwb-path {} --name {} --output-dir {}
```

`--embedding-number` is the number of streams (`g1`, `g2`, ...) to archive, and `--original-data-size` is their total uncompressed size in bytes.

The TensorRT support is provided in scripts `encode_tensorrt.py` and `decode_tensorrt.py` for faster compression and decompression after model deployment.

```
CUDA_VISIBLE_DEVICES=0 python encode_tensorrt.py --experiment {} --epoch {} \
                    --input-tiff {} --stream-output {} --engine-dir ./tensorrt_engines \
                    --precision amp --batch-size 8 --entropy-workers 8 \
                    --loader-workers 8 --overwrite

CUDA_VISIBLE_DEVICES=0 python decode_tensorrt.py --experiment {} --epoch {} \
                    --stream-input {} --output-tiff {} --engine-dir ./tensorrt_engines \
                    --precision amp --batch-size 8 --entropy-workers 8 --overwrite
```

The required TensorRT engines are built and cached in `--engine-dir` on first use. Use the same directory for subsequent compression and decompression.


## Results

### 1. Performance of Necoda across diverse imaging modalities.
Evaluation of Necoda on multiple functional imaging datasets reveals that Necoda effectively compresses neuroimaging collections while preserving necessary physiological information.

<img src="fig/performance1.png" width="800" align="middle">

### 2. Benchmarking Necoda with existing video codecs.
Necoda outperforms other codecs in training/compression efficiency and rate distortion performance on simulation benchmarks.

<img src="fig/benchmark1.png" width="800" align="middle">

### 3. Reproducing analysis with Necoda on ABO datasets.
Reproduction of results from one previous research paper with ABO datasets demonstrates that Necoda is able to function as a tool to facilitate TB-level data sharing and replication.

<img src="fig/performance2.png" width="800" align="middle">



## Citation

Currently paper of this project is not officially online.
