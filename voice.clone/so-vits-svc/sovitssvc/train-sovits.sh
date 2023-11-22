#!/bin/bash

#SBATCH -p gpu5     # 在 gpu1 分区运行（不写默认为 cpu1）
#SBATCH -N 1          # 只在一个节点上运行任务
#SBATCH -c 4          # 申请 CPU 核心：1个
#SBATCH --mem 20G     # 申请内存：500 MB
#SBATCH --gres gpu:1  # 分配1个GPU（纯 CPU 任务不用写）
# #SBATCH --nodelist wmc-slave-g12
# #SBATCH -o log/%j-train-30.out
#SBATCH -o log/%j-train-30-50000ep.out
# #SBATCH --nodelist wmc-slave-g14

date
echo "job begin"

# python resample.py;
# python preprocess_flist_config.py;
# python preprocess_hubert_f0.py;
python train.py -c configs/config.json -m 44k


echo "job end"
date
