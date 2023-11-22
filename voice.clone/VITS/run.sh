#!/bin/bash

#SBATCH -p gpu3       
#SBATCH -N 1          
#SBATCH -c 2          
#SBATCH --mem 35G     
#SBATCH --gres gpu:4 
#SBATCH -o log/%j-vits-wm0.out
date
echo "job begin"
python train.py -c configs/ljs_base.json -m ljs_base
python save_syn.py
echo "job end"
date
