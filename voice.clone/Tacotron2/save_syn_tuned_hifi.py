import matplotlib
import matplotlib.pylab as plt

# import IPython.display as ipd
import os
import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
# from train import load_model
from text import text_to_sequence
# from denoiser import Denoiser
import json
import hifigan
from scipy.io import wavfile


def load_model(hparams):
    model = Tacotron2(hparams)
    return model

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='upper', 
                       interpolation='none')


hparams = create_hparams()
hparams.sampling_rate = 22050


checkpoint_path = "results_wm0/result/checkpoint_90000"

model = load_model(hparams)
# model.load_state_dict(torch.load(checkpoint_path,map_location=torch.device('cpu'))['state_dict'])
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_vocoder(device):
    with open("hifigan/config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)

    ckpt = torch.load("../Hifi-GAN/ckpt_wm0/g_02424000")
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder
vocoder = get_vocoder(device)


import os
import soundfile
# txt = os.listdir("txt/p225")
# out_dir = "./syned/universal-hifi_mine_gf"
# if not os.path.exists(out_dir): os.makedirs(out_dir)

# for f in txt:
#     t = open(os.path.join("txt/p225", f), "r")
#     text = t.readline()[:-1]
#     sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
#     sequence = torch.autograd.Variable(
#         torch.from_numpy(sequence)).cuda().long()
#     mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
#     with torch.no_grad():
#         mel_predictions = mel_outputs_postnet
#         wavs = (vocoder(mel_predictions).squeeze(1).cpu().numpy() * hparams.max_wav_value).astype("int16")
#         wav_path = os.path.join(out_dir, text[:-1] + ".wav")
#         wavfile.write(wav_path, hparams.sampling_rate, wavs[0])
#     print(wav_path)


txtf = open("../ljs_audio_text_test_filelist.txt", 'r')
txt = txtf.readlines()
out_dir = "./syned/tuned-hifi_wm0"
if not os.path.exists(out_dir): os.makedirs(out_dir)
count = 0
for text in txt:
    count += 1
    text = text.split("|")[-1]
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    with torch.no_grad():
        mel_predictions = mel_outputs_postnet
        wavs = (vocoder(mel_predictions).squeeze(1).cpu().numpy() * hparams.max_wav_value).astype("int16")
        wav_path = os.path.join(out_dir, str(count) + ".wav")
        wavfile.write(wav_path, hparams.sampling_rate, wavs[0])
    print(wav_path)