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
from denoiser import Denoiser


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

# checkpoint_path = "pth/tacotron2_statedict.pt"
model_p = "pth"
checkpoint_path = os.path.join(model_p, os.listdir(model_p)[-1])
checkpoint_path = "pth/tacotron2_statedict.pt"
checkpoint_path = "pth/checkpoint_38000"
model = load_model(hparams)
# model.load_state_dict(torch.load(checkpoint_path,map_location=torch.device('cpu'))['state_dict'])
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()


waveglow_path = 'wav_pth/waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)


import os
import soundfile
txt = os.listdir("txt/p225")
out_dir = "./syned/wm_syn_wav"
if not os.path.exists(out_dir): os.makedirs(out_dir)

for f in txt:
    t = open(os.path.join("txt/p225", f), "r")
    text = t.readline()[:-1]
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        # audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    wav_path = os.path.join(out_dir, text[:-1] + ".wav")
    # soundfile.write(wav_path, audio[0].data.cpu().numpy(), samplerate=hparams.sampling_rate)
    soundfile.write(wav_path, audio[0].data.cpu().numpy().astype(np.float32), samplerate=hparams.sampling_rate)
    # soundfile.write(wav_path, audio_denoised[0].data.cpu().numpy(), samplerate=hparams.sampling_rate)
    print(wav_path)