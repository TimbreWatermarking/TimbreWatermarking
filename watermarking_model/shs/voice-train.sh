#!/bin/bash

#SBATCH -p gpu5
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem 40G
#SBATCH --gres gpu:1
#SBATCH -o log/%j-train.out
date
echo "job begin"
python train.py -p config/process.yaml -m config/model.yaml -t config/train.yaml
echo "job end"
date
