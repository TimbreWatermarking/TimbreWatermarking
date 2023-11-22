# Vocoder: Hifi-GAN
We imported the Hifi-GAN code from [https://github.com/jik876/hifi-gan](https://github.com/jik876/hifi-gan) and made some adaptations as needed. Here, we provide the complete code used in our voice cloning experiments.

## Setup
Install python requirements
```
cd Hifi-GAN
pip install -r requirements.txt
```

## Download pretrained Model
Download our pre-trained model, which was trained on a clean ljspeech, from [here](https://drive.google.com/drive/folders/1ZtapjTZoP6ADz_QoQnI8lvuRdYKCLhvB?usp=drive_link) and place it in the designated path for pre-placed model files:
```
mkdir ckpt_wm0
```
The model file fine-tuned with voice containing watermark-0 (the first watermark in `watermarking_model/results/wmpool.txt`) is also available at the aforementioned link.


## Finetune the model with 1k watermarked wavs
```
python train.py --input_wavs_dir "path/to/wm_speech/wmed-0/wavs" --input_training_file "LJSpeech-1.1/training_small.txt" --checkpoint_path "ckpt_wm0" --training_epochs 3200 --checkpoint_interval 3000;
```

The dataset used here is a subset for training acoustic models.