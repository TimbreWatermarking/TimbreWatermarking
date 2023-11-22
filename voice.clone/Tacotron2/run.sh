#!/bin/bash

#SBATCH -p gpu5      
#SBATCH -N 1          
#SBATCH -c 2          
#SBATCH --mem 8G     
#SBATCH --gres gpu:1 
#SBATCH -o log/%j-save-wm0-tuned_hifi.out
date
echo "job begin"
# python save_syn_gl.py
# python save_syn_hifi.py
python save_syn_tuned_hifi.py
echo "job end"
date
