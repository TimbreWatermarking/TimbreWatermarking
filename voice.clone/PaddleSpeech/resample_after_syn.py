import torch
import torchaudio
import os
import julius
from pydub import AudioSegment
import soundfile

root = "paddlepaddle-syned"
aim = "paddlepaddle-syned-22050"
SAMPLE_RATE = 22050

def main():
    print("resample")
    aus = os.listdir(root)
    for d in aus:
        if not os.path.exists(os.path.join(aim, d)): os.makedirs(os.path.join(aim, d))
        wavs = os.listdir(os.path.join(root, d))
        for wav in wavs:
            if ".wav" in wav:
                print("resample")
                a, sr = torchaudio.load(os.path.join(root, d, wav))
                resampler = julius.ResampleFrac(sr, SAMPLE_RATE)
                a = resampler(a)
                aud = a[0].numpy()
                soundfile.write(os.path.join(os.path.join(aim, d, wav)), aud, SAMPLE_RATE)


if __name__ == "__main__":
    main()