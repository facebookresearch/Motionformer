# Motionformer

This is an official pytorch implementation of paper [Keeping Your Eye on the Ball: Trajectory Attention in Video Transformers](https://arxiv.org/abs/2106.05392). In this repository, we provide PyTorch code for training and testing our proposed Motionformer model. Motionformer use proposed *trajectory attention* to achieve state-of-the-art results on several video action recognition benchmarks such as Kinetics-400 and Something-Something V2.

If you find Motionformer useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@inproceedings{patrick2021keeping,
   title={Keeping Your Eye on the Ball: Trajectory Attention in Video Transformers}, 
   author={Mandela Patrick and Dylan Campbell and Yuki M. Asano and Ishan Misra Florian Metze and Christoph Feichtenhofer and Andrea Vedaldi and Jo\ão F. Henriques},
   year={2021},
   booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
}
```

# Model Zoo

We provide Motionformer models pretrained on Kinetics-400 (K400), Kinetics-600 (K600), Something-Something-V2 (SSv2), and Epic-Kitchens datasets.

| name | dataset | # of frames | spatial crop | acc@1 | acc@5 | url |
| --- | --- | --- | --- | --- | --- | --- |
| Joint | K400 | 16 | 224 | 79.2 | 94.2 | [model](https://dl.fbaipublicfiles.com/motionformer/k400_joint_224_16x4.pyth) |
| Divided | K400 | 16 | 224 | 78.5 | 93.8 | [model](https://dl.fbaipublicfiles.com/motionformer/k400_divided_224_16x4.pyth) |
| Motionformer | K400 | 16 | 224 | 79.7 | 94.2 | [model](https://dl.fbaipublicfiles.com/motionformer/k400_motionformer_224_16x4.pyth) |
| Motionformer-HR | K400 | 16 | 336 | 81.1 | 95.2 | [model](https://dl.fbaipublicfiles.com/motionformer/k400_motionformer_336_16x8.pyth) |
| Motionformer-L | K400 | 32 | 224 | 80.2 | 94.8 | [model](https://dl.fbaipublicfiles.com/motionformer/k400_motionformer_224_32x3.pyth) |

| name | dataset | # of frames | spatial crop | acc@1 | acc@5 | url |
| --- | --- | --- | --- | --- | --- | --- |
| Motionformer | K600 | 16 | 224 | 81.6 | 95.6 | [model](https://dl.fbaipublicfiles.com/motionformer/k600_motionformer_224_16x4.pyth) |
| Motionformer-HR | K600 | 16 | 336 | 82.7 | 96.1 | [model](https://dl.fbaipublicfiles.com/motionformer/k600_motionformer_336_16x8.pyth) |
| Motionformer-L | K600 | 32 | 224 | 82.2 | 96.0 | [model](https://dl.fbaipublicfiles.com/motionformer/k600_motionformer_224_32x3.pyth) |

| name | dataset | # of frames | spatial crop | acc@1 | acc@5 | url |
| --- | --- | --- | --- | --- | --- | --- |
| Joint | SSv2 | 16 | 224 | 64.0 | 88.4 | [model](https://dl.fbaipublicfiles.com/motionformer/ssv2_joint_224_16x4.pyth) |
| Divided | SSv2 | 16 | 224 | 64.2 | 88.6 | [model](https://dl.fbaipublicfiles.com/motionformer/ssv2_divided_224_16x4.pyth) |
| Motionformer | SSv2 | 16 | 224 | 66.5 | 90.1 | [model](https://dl.fbaipublicfiles.com/motionformer/ssv2_motionformer_224_16x4.pyth) |
| Motionformer-HR | SSv2 | 16 | 336 | 67.1 | 90.6 | [model](https://dl.fbaipublicfiles.com/motionformer/ssv2_motionformer_336_16x4.pyth) |
| Motionformer-L | SSv2 | 32 | 224 | 68.1 | 91.2 | [model](https://dl.fbaipublicfiles.com/motionformer/ssv2_motionformer_224_32x3.pyth) |

| name | dataset | # of frames | spatial crop | A acc | N acc | url |
| --- | --- | --- | --- | --- | --- | --- |
| Motionformer | EK | 16 | 224 | 43.1 | 56.5 | [model](https://dl.fbaipublicfiles.com/motionformer/ek_motionformer_224_16x4.pyth) |
| Motionformer-HR | EK | 16 | 336 | 44.5 | 58.5 | [model](https://dl.fbaipublicfiles.com/motionformer/ek_motionformer_336_16x4.pyth) |
| Motionformer-L | EK | 32 | 224 | 44.1 | 57.6 | [model](https://dl.fbaipublicfiles.com/motionformer/ek_motionformer_224_32x3.pyth) |

# Installation

First, create a conda virtual environment and activate it:
```
conda create -n motionformer python=3.8.5 -y
source activate motionformer
```

Then, install the following packages:

- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- simplejson: `pip install simplejson`
- einops: `pip install einops`
- timm: `pip install timm`
- PyAV: `conda install av -c conda-forge`
- psutil: `pip install psutil`
- scikit-learn: `pip install scikit-learn`
- OpenCV: `pip install opencv-python`
- tensorboard: `pip install tensorboard`
- matplotlib: `pip install matplotlib`
- pandas: `pip install pandas`
- ffmeg: `pip install ffmpeg-python`

OR:

simply create conda environment with all packages just from yaml file:

`conda env create -f environment.yml`

Lastly, build the Motionformer codebase by running:
```
git clone https://github.com/facebookresearch/Motionformer
cd Motionformer
python setup.py build develop
```

# Usage

## Dataset Preparation

Please use the dataset preparation instructions provided in [DATASET.md](slowfast/datasets/DATASET.md).

## Training the Default Motionformer

Training the default Motionformer that uses trajectory attention, and operates on 16-frame clips cropped at 224x224 spatial resolution, can be done using the following command:

```
python tools/run_net.py \
  --cfg configs/K400/motionformer_224_16x4.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```
You may need to pass location of your dataset in the command line by adding `DATA.PATH_TO_DATA_DIR path_to_your_dataset`, or you can simply modify

```
DATA:
  PATH_TO_DATA_DIR: path_to_your_dataset
```

We improved the trajectory attention from original code, and you can set the `VIT.USE_ORIGINAL_TRAJ_ATTN_CODE` flag to `False` to use it:
```
VIT:
  USE_ORIGINAL_TRAJ_ATTN_CODE: False
```

To the yaml configs file, then you do not need to pass it to the command line every time.

## Using a Different Number of GPUs

If you want to use a smaller number of GPUs, you need to modify .yaml configuration files in [`configs/`](configs/). Specifically, you need to modify the NUM_GPUS, TRAIN.BATCH_SIZE, TEST.BATCH_SIZE, DATA_LOADER.NUM_WORKERS entries in each configuration file. The BATCH_SIZE entry should be the same or higher as the NUM_GPUS entry.

## Using Different Self-Attention Schemes

If you want to experiment with different space-time self-attention schemes, e.g., joint space-time attention or divided space-time attention, use the following commands:


```
python tools/run_net.py \
  --cfg configs/K400/joint_224_16x4.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

and

```
python tools/run_net.py \
  --cfg configs/K400/divided_224_16x4.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

## Training Different Motionformer Variants

If you want to train more powerful Motionformer variants, e.g., Motionformer-HR (operating on 16-frame clips sampled at 336x336 spatial resolution), and Motionformer-L (operating on 32-frame clips sampled at 224x224 spatial resolution), use the following commands:

```
python tools/run_net.py \
  --cfg configs/K400/motionformer_336_16x8.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

and

```
python tools/run_net.py \
  --cfg configs/K400/motionformer_224_32x3.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \
```

Note that for these models you will need a set of GPUs with ~32GB of memory.

## Inference

Use `TRAIN.ENABLE` and `TEST.ENABLE` to control whether training or testing is required for a given run. When testing, you also have to provide the path to the checkpoint model via TEST.CHECKPOINT_FILE_PATH.
```
python tools/run_net.py \
  --cfg configs/K400/motionformer_224_16x4.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False \
```

Alterantively, you can modify provided SLURM script and run following:

```
sbatch slurm_scripts/test.sh configs/K400/motionformer_224_16x4.yaml path_to_your_checkpoint
```

## Single-Node Training via Slurm

To train Motionformer via Slurm, please check out our single node Slurm training script [`slurm_scripts/run_single_node_job.sh`](slurm_scripts/run_single_node_job.sh).

```
sbatch slurm_scripts/run_single_node_job.sh configs/K400/motionformer_224_16x4.yaml /your/job/dir/${JOB_NAME}/
```

## Multi-Node Training via Submitit

Distributed training is available via Slurm and submitit

```
pip install submitit
```

To train Motionformer model on Kinetics using 8 nodes with 8 gpus each use the following command:
```
python run_with_submitit.py --cfg configs/K400/motionformer_224_16x4.yaml --job_dir  /your/job/dir/${JOB_NAME}/ --partition $PARTITION --num_shards 8 --use_volta32
```

We provide a script for launching slurm jobs in [`slurm_scripts/run_multi_node_job.sh`](slurm_scripts/run_multi_node_job.sh).

```
sbatch slurm_scripts/run_multi_node_job.sh configs/K400/motionformer_224_16x4.yaml /your/job/dir/${JOB_NAME}/
```

Please note that hyper-parameters in configs were used with 8 nodes with 8 gpus (32 GB). Please scale batch-size, and learning-rate appropriately for your cluster configuration.

## Finetuning

To finetune from an existing PyTorch checkpoint add the following line in the command line, or you can also add it in the YAML config:

```
TRAIN.CHECKPOINT_EPOCH_RESET: True
TRAIN.CHECKPOINT_FILE_PATH path_to_your_PyTorch_checkpoint
```

# Environment

The code was developed using python 3.8.5 on Ubuntu 20.04. For training, we used eight GPU compute nodes each node containing 8 Tesla V100 GPUs (32 GPUs in total). Other platforms or GPU cards have not been fully tested.

# License

The majority of this work is licensed under [CC-NC 4.0 International license](LICENSE). However, portions of the project are available under separate license terms: [SlowFast](https://github.com/facebookresearch/SlowFast) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) are licensed under the Apache 2.0 license.

# Contributing

We actively welcome your pull requests. Please see [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for more info.

# Acknowledgements

Motionformer is built on top of [PySlowFast](https://github.com/facebookresearch/SlowFast), [Timesformer](https://github.com/facebookresearch/TimeSformer) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman). We thank the authors for releasing their code. If you use our model, please consider citing these works as well:

```BibTeX
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```

```BibTeX
@inproceedings{gberta_2021_ICML,
    author  = {Gedas Bertasius and Heng Wang and Lorenzo Torresani},
    title = {Is Space-Time Attention All You Need for Video Understanding?},
    booktitle   = {Proceedings of the International Conference on Machine Learning (ICML)}, 
    month = {July},
    year = {2021}
}
```

```BibTeX
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```
