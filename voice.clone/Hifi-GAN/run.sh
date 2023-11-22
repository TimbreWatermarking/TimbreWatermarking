#!/bin/bash

#SBATCH -p gpu5      
#SBATCH -N 1          
#SBATCH -c 2          
#SBATCH --mem 20G     
#SBATCH --gres gpu:4 
#SBATCH -o log/%j-train-or.out
date
echo "job begin"
python train.py --input_wavs_dir "/public/user/data/ljspeech/LJSpeech-1.1/wavs" --input_training_file "/public/user/experiment/voice-clone/fs-wm0/preprocessed_data/LJSpeech/train.txt" --checkpoint_path "ckpt_or";
echo "job end"
date
