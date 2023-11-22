# train watermarking model
python train.py -p config/process.yaml -m config/model.yaml -t config/train.yaml



# embed
python embed_and_save.py -o "data/ljspeech/LJSpeech-1.1/wavs/" \
                         -s "results/wm_speech/ljspeech/" --wm 0 \
                         -mp "results/ckpt/pth/"



# extract
python extract.py --wm 0



# test common processing
python common_test.py -n "experiments_results"