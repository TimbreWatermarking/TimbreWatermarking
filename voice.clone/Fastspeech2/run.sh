#!/bin/bash

#SBATCH -p gpu5      
#SBATCH -N 1          
#SBATCH -c 2          
#SBATCH --mem 8G     
#SBATCH --gres gpu:1 
#SBATCH -o log/%j-save-wm0-tuned_hifi.out
date
echo "job begin"
# python prepare_align.py config/LJSpeech/preprocess.yaml;
# date;
# python preprocess.py config/LJSpeech/preprocess.yaml;
# python save_syn_gl.py
# python save_syn2.py
python save_syn_tuned_hifi.py --restore_step 900000 -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml;
date;
# python train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
# python save_syn_gl.py --restore_step 900000 -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
echo "job end"
date

