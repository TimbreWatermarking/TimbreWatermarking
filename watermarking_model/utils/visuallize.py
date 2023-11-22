import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt


def save_spectrum(spect, phase, flag='linear'):
    root = "draw_figure/test_stft"
    os.makedirs(root, exist_ok=True)
    spec = librosa.amplitude_to_db(spect.squeeze(0).cpu().numpy(), ref=np.max)
    img=librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='log', y_coords=None);
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_amplitude_spectrogram.png'), bbox_inches='tight', pad_inches=0.0)
    spec = librosa.amplitude_to_db(phase.squeeze(0).cpu().numpy(), ref=np.max)
    img=librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='log');
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_phase_spectrogram.png'), bbox_inches='tight', pad_inches=0.0)