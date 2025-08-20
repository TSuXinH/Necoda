# Necoda: High-Fidelity Functional Neural Data Compression via Neural Representation Enhances Data Sharing

<img src="fig/overview1.png" width="800" align="middle">

<!-- ### :triangular_flag_on_post: [[New version released! ：DeepCAD-RT]](https://github.com/cabooster/DeepCAD-RT) -->

## Contents

- [Overview](#overview)
- [Directory structure](#directory-structure)
- [Pytorch code](#pytorch-code)
- [Results](#results)
- [Citation](#citation)

## Overview

<img src="fig/overview2.png" width="400" align="right">

In neuroscience, progress hinges on sharing large-scale data. Initiatives like the Allen Brain Observatory (ABO) and standards like Neurodata Without Borders (NWB) are vital for enabling broad data reuse and collaboration. Sharing comprehensive, raw-like imaging data is particularly crucial for transparency, reproducibility, and developing new analysis tools.

The primary obstacle is data size. Functional imaging datasets often reach the terabyte-scale, creating a massive bottleneck for data sharing, storage, and reuse. These logistical hurdles severely slow the pace of discovery and hinder the reproduction of scientific findings.


To solve this, we developed **Necoda, a deep learning method that compresses functional imaging data by over 1,000-fold while maintaining high fidelity**. Our method uses a content-adaptive encoder-decoder network that leverages the inherent spatiotemporal structure of neural recordings. A key innovation is Necoda's ability to generalize across unseen data; we can train it once on a small data subset and then apply it to compress numerous other experiments, making it a practical and efficient tool. 

We demonstrated that **Necoda preserves essential scientific information while drastically reducing file size**. We validated our method on diverse datasets, including data from different species, brain regions, and imaging modalities. In our key proof-of-concept, we compressed a 4.82 TB ABO dataset to just 4.81 GB. Using only this compact file, we fully reproduced the published findings of a major neuroscience study in just a few hours. 

By removing the data-sharing bottleneck, we believe Necoda will significantly accelerate discovery and enhance reproducibility in neuroscience.

## Directory structure

## Pytorch code 

### Environment 

* Ubuntu 20.04
* Python 3.10.14
* Pytorch 2.2.1
* NVIDIA GPU (24 GB Memory) + CUDA

### Code setup


* Create a virtual environment and install Pytorch. Please select the correct Pytorch version that matches your CUDA version from https://pytorch.org/get-started/previous-versions/

```
$ conda create -n necoda python=3.10
$ source activate necoda
$ pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

* Clone the environment

```
$ git clone git@github.com:TSuXinH/Necoda.git
$ cd Necoda
```

* Install other dependencies

```
$ pip install -r requirements
```


### Training



### Test



## Results

### 1. The performance of Necoda across diverse imaging modalities.

<img src="fig/performance1.png" width="800" align="middle">

### 2. Benchmarking Necoda with existing video codecs.

<img src="fig/benchmark1.png" width="800" align="middle">

### 3. Reproducing analysis with Necoda on ABO datasets.

<img src="fig/performance2.png" width="800" align="middle">



## Citation

If you use this code please cite the companion paper where the original method appeared: 

