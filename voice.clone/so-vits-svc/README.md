# Voice Conversion model: so-vits-svc
Here, we provide the complete source code for conducting experiments with [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc). When using it, you may need to set up your Python environment according to 'my_requirements.txt'.

Pre_trained model is downloaded from [https://huggingface.co/lqlklu/so-vits-svc-4.0-danxiao](https://huggingface.co/lqlklu/so-vits-svc-4.0-danxiao). Based on this, we use a small amount of singing voice audio of the target speaker to fine-tune the model and implement voice cloning.

## Model files
- We provide [here](https://drive.google.com/drive/folders/1KFgFCsSVb3KVQCRrGKCkLzx5KQHHADrq?usp=drive_link) the pre-training model files used and the speech cloning model files obtained by fine-tuning using audio data containing watermark-0 (the first watermark in `watermarking_model/results/wmpool.txt`).

- Additionally, following the official instructions of so-vits-svc, the model parameter files for hubert need to be downloaded and placed in the `hubert` directory. We provide the necessary files [here](https://drive.google.com/file/d/1uj0crMdw66Dkl9fkjsQ0mWBDul_2IsHb/view?usp=drive_link).


## Dataset
We use 30 voice samples from [OpenCpop](https://wenet.org.cn/opencpop/) as our dataset. Before training, we embed watermarks into these audio files. The audio is placed in the  `dataset_raw` directory. The directory structure is as follows, showing the specific audio dataset that we used.
```
dataset_raw/
├── opencpop
│   ├── 2001000001.wav
│   ├── 2001000002.wav
│   ├── 2001000003.wav
│   ├── 2001000004.wav
│   ├── 2001000005.wav
│   ├── 2001000006.wav
│   ├── 2001000007.wav
│   ├── 2001000008.wav
│   ├── 2001000009.wav
│   ├── 2001000010.wav
│   ├── 2001000011.wav
│   ├── 2001000012.wav
│   ├── 2001000013.wav
│   ├── 2001000014.wav
│   ├── 2001000015.wav
│   ├── 2001000016.wav
│   ├── 2001000017.wav
│   ├── 2001000018.wav
│   ├── 2001000019.wav
│   ├── 2001000020.wav
│   ├── 2001000021.wav
│   ├── 2001000022.wav
│   ├── 2001000023.wav
│   ├── 2001000024.wav
│   ├── 2001000025.wav
│   ├── 2001000026.wav
│   ├── 2001000027.wav
│   ├── 2001000028.wav
│   ├── 2001000029.wav
│   └── 2001000030.wav
└── wav_structure.txt
```


## Train
```
python resample.py;
python preprocess_flist_config.py;
python preprocess_hubert_f0.py;
python train.py -c configs/config.json -m 44k
```

## Inference
We utilized the [Ultimate Vocal Remover](https://ultimatevocalremover.com/) to strip the vocals and background music from the audio we intended to transform. The extracted vocal audio is stored in the `./raw` directory. Following that, we use a timbre conversion model embedded with watermarks to modify the timbre of the vocals.
```
for i in {0..31}
do
   python inference_main.py -m "logs/44k_from_danxiao/G_60000.pth" -c "configs/config.json" -s "opencpop" -n "chunk_${i}.wav" -t 0
done
```