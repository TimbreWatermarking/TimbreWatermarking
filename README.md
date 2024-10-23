# :rabbit: [Detecting Voice Cloning Attacks via Timbre Watermarking](https://github.com/TimbreWatermarking/TimbreWatermarking)

Source code for [paper](https://www.ndss-symposium.org/wp-content/uploads/2024-200-paper.pdf) “Detecting Voice Cloning Attacks via Timbre Watermarking” 

by _Chang Liu, Jie Zhang, Tianwei Zhang, Xi Yang, Weiming Zhang, and Nenghai Yu_ 
In [Network and Distributed System Security Symposium (NDSS) 2024](https://www.ndss-symposium.org/ndss2024/).

Visit our [website](https://timbrewatermarking.github.io/samples.html) for audio samples.

## Introduction

:rabbit2: In this repository, we provide the complete code for training and testing the watermarking model. Additionally, we include the source code used for voice cloning experiments under various scenarios, along with corresponding README files. `Please visit the respective directories to access detailed READMEs`

- [watermarking_model](https://github.com/TimbreWatermarking/TimbreWatermarking/tree/main/watermarking_model): Code of the watermarking model
- [voice.clone](https://github.com/TimbreWatermarking/TimbreWatermarking/tree/main/voice.clone): Code and details of the voice cloning part


## Model files
All the parameter files for the voice cloning model used in our work are available at [this link](https://drive.google.com/drive/folders/1tRbEneN1VsSCZ0HPxG3DSoJdxDRZ_NUJ?usp=drive_link).


## Acknowledgments

Part of our experiments were based on code from several open-source repositories, including [VITS](https://github.com/jaywalnut310/vits), [Tacotron2](https://github.com/NVIDIA/tacotron2), [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc), [Hifi-GAN](https://github.com/jik876/hifi-gan), and [FastSpeech2](https://github.com/ming024/FastSpeech2). Their code served as a foundation for portions of our experiments.



## Citation
If you find this work useful, please consider citing our paper:
```
@inproceedings{timbrewatermarking-ndss2024,
  title = {Detecting Voice Cloning Attacks via Timbre Watermarking},
  author = {Liu, Chang and Zhang, Jie and Zhang, Tianwei and Yang, Xi and Zhang, Weiming and Yu, Nenghai},
  booktitle = {Network and Distributed System Security Symposium},
  year = {2024},
  doi = {10.14722/ndss.2024.24200},
}
```
