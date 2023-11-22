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


def load_model(hparams):
    model = Tacotron2(hparams)
    return model

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='upper', 
                       interpolation='none')







import os
import soundfile
# txt = os.listdir("txt/p225")
# out_dir = "./syned/gl_syned"
# if not os.path.exists(out_dir): os.makedirs(out_dir)




# add
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.align_tts_config import AlignTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.config import BaseAudioConfig
output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(formatter="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "/public/user/experiment/voice-watermarking/results/wm_speech/wmed_7/"))
audio_config = BaseAudioConfig(
                sample_rate=22050,
                do_trim_silence=True,
                trim_db=60.0,
                signal_norm=False,
                mel_fmin=0.0,
                mel_fmax=8000,
                spec_gain=1.0,
                log_func="np.log",
                ref_level_db=20,
                preemphasis=0.0,
            )
config = AlignTTSConfig(
        audio=audio_config,
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="english_cleaners",
        use_phonemes=False,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=True,
        mixed_precision=False,
        output_path=output_path,
        datasets=[dataset_config],
    )
ap = AudioProcessor.init_from_config(config)


import argparse
def main(checkpoint_path, out_dir):
    hparams = create_hparams()
    hparams.sampling_rate = 22050

    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval()

    txtf = open("../ljs_audio_text_test_filelist.txt", 'r')
    txt = txtf.readlines()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    count = 0
    for text in txt:
        count += 1
        text = text.split("|")[-1]
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

        with torch.no_grad():
            wav_predictions = ap.inv_melspectrogram(mel_outputs_postnet[0].cpu().numpy())

        wav_path = os.path.join(out_dir, str(count) + ".wav")
        ap.save_wav(wav_predictions, wav_path, hparams.sampling_rate)
        print(wav_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="results_wm0/result/checkpoint_90000",
                        help="Path to the model checkpoint")
    parser.add_argument("--out_dir", type=str, default="./syned/griffinlim_syned",
                        help="Directory to save the synthesized audio files")
    args = parser.parse_args()

    main(args.checkpoint_path, args.out_dir)
