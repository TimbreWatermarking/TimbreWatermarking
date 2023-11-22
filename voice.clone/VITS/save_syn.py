# import matplotlib.pyplot as plt
# import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import soundfile

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import numpy as np

from scipy.io.wavfile import write


hps = utils.get_hparams_from_file("./configs/ljs_base.json")


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

import sys
def main():
    if len(sys.argv) > 1:
        argument = sys.argv[1]
        print(f"Received argument: {argument}")
    else:
        print("No argument received.")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        # **hps.model).cuda()
        **hps.model)
    _ = net_g.eval()

    # _ = utils.load_checkpoint("logs/ljs_base/G_300000.pth", net_g, None)
    _ = utils.load_checkpoint(argument, net_g, None)


    txtf = open("../ljs_audio_text_test_filelist.txt", 'r')
    txt = txtf.readlines()
    out_dir = "./syned/syn_wav"
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    count = 0
    for text in txt:
        count += 1
        text = text.split("|")[-1]
        stn_tst = get_text(text, hps)
        with torch.no_grad():
            # x_tst = stn_tst.cuda().unsqueeze(0)
            # x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
            audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        # import pdb
        # pdb.set_trace()
        # ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
        wav_path = os.path.join(out_dir, str(count) + ".wav")
        soundfile.write(wav_path, audio.astype(np.float32), samplerate=hps.data.sampling_rate)
        print(wav_path)

if __name__ == "__main__":
    main()