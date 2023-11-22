# Voice Cloning model: VITS
We imported the source code of VITS from [https://github.com/jaywalnut310/vits](https://github.com/jaywalnut310/vits) and made some adaptations as needed. Here, we provide the complete code used in our voice cloning experiments.

## Setup
Install python requirements
```
cd VITS
pip install -r requirements.txt
```
Update .wav paths: 
```
sed -i -- 's,DUMMY,ljs_dataset_folder/wavs,g' filelists/*.txt
```

## Training
```
python train.py -c configs/ljs_base.json -m ljs_base
```

## Synthesizing
In [this link](https://drive.google.com/drive/folders/1Fdf02xD31IgOAo7HPzRAxvUyG1wFxzTw?usp=drive_link), we provide a model parameter file trained using a dataset embedded with watermark-0 (the first watermark in `watermark_model/results/wmpool.txt`). When this model file is used for voice cloning, the resulting cloned voice will include watermark-0.
```
python save_syn.py "path/to/G_300000.pth"
```

