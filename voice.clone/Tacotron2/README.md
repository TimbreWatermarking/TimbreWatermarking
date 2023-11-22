# Acoustic model: Tacotron 2
We imported the Tacotron2 code from [https://github.com/NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2) and made some adaptations as needed. Here, we provide the complete code used in our voice cloning experiments.

## Setup
Install python requirements
```
cd Tacotron2
pip install -r requirements.txt
```
Update .wav paths: 
```
sed -i -- 's,DUMMY,ljs_dataset_folder/wavs,g' filelists/*.txt
```

## Training
```
python train.py --output_directory="results_wm0/result" --log_directory="results/log" --n_gpus=1
```

## Synthesizing
In [this link](https://drive.google.com/drive/folders/1wH3cCJi1HVC1LwNh7yerYqNzy_PX17zG?usp=drive_link), we provide a model parameter file trained using a dataset embedded with watermark-0 (the first watermark in `watermark_model/results/wmpool.txt`). When this model file is used for voice cloning, the resulting cloned voice will include watermark-0.

1. Synthesizing with pre-trained Hifi-GAN:
```
python save_syn_hifi.py --checkpoint_path "path/to/checkpoint" --out_dir "path/to/output/directory" --vocoder_dir "path/to/pretrained/vocoder"
```
2. Synthesizing with Griffin-Lim:
```
python save_syn_gl.py --checkpoint_path "path/to/checkpoint" --out_dir "path/to/output/directory"
```
3. Synthesizing with tuned watermarked Hifi-GAN:
```
python save_syn_hifi.py --checkpoint_path "path/to/checkpoint" --out_dir "path/to/output/directory" --vocoder_dir "path/to/watermarked/vocoder"
```