# Acoustic model: Fastspeech 2
We imported the Fastspeech2 code from [https://github.com/ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) and made some adaptations as needed. Here, we provide the complete code used in our voice cloning experiments.

## Setup
Install python requirements
```
cd Fastspeech2
pip install -r requirements.txt
```

## Training
```
python prepare_align.py config/LJSpeech/preprocess.yaml;
python preprocess.py config/LJSpeech/preprocess.yaml;
python train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

## Synthesizing
In [this link](https://drive.google.com/drive/folders/1S9-F7I-Wr9AuSSoLvKGZd_W7SbkTMFBg?usp=drive_link), we provide a model parameter file trained using a dataset embedded with watermark-0 (the first watermark in `watermark_model/results/wmpool.txt`). When this model file is used for voice cloning, the resulting cloned voice will include watermark-0.

1. Synthesizing with pre-trained Hifi-GAN:
```
python save_syn2.py -sd "./syned/universal-hifi" \
                    -vd "path/to/hifigan/pretrained/g_02420000" \
                    --restore_step 900000 -p config/LJSpeech/preprocess.yaml \
                    -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```
2. Synthesizing with Griffin-Lim:
```
python save_syn_gl.py   -sd "./syned/griffinlim_syned" \
                        --restore_step 900000 -p config/LJSpeech/preprocess.yaml \
                        -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```
3. Synthesizing with tuned watermarked Hifi-GAN:
```
python save_syn2.py -sd "./syned/tuned-hifi_wm0" \
                    -vd "path/to/hifigan/ckpt_wm0/g_02424000" \
                    --restore_step 900000 -p config/LJSpeech/preprocess.yaml \
                    -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```