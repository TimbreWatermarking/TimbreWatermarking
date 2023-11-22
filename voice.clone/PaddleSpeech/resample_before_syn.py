import torch
import torchaudio
import os
import julius
from pydub import AudioSegment
import soundfile

root = "wmed"
aim = "wmed-24000"
SAMPLE_RATE = 24000

def main():
    print("resample")
    aus = os.listdir(root)
    for d in aus:
        if not os.path.exists(os.path.join(aim, d)): os.makedirs(os.path.join(aim, d))
        wavs = os.listdir(os.path.join(root, d, 'wmed-0', "wavs"))
        for wav in wavs:
            if ".wav" in wav:
                print("resample")
                a, sr = torchaudio.load(os.path.join(root, d, 'wmed-0', "wavs", wav))
                resampler = julius.ResampleFrac(sr, SAMPLE_RATE)
                a = resampler(a)
                aud = a[0].numpy()
                soundfile.write(os.path.join(os.path.join(aim, d, wav)), aud, SAMPLE_RATE)


if __name__ == "__main__":
    main()