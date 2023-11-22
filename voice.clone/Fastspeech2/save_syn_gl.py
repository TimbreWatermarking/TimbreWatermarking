import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples, my_synth_samples, my_synth_samples2
from dataset import TextDataset
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

'''
def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                "output/result/LJSpeech_original",
            )
'''

def mysynthesize(model, step, configs, vocoder, batchs, control_values, args):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    import os
    import soundfile
    # txt = os.listdir("txt/p225")
    # out_dir = "./syned/wm_syn_wav"
    # if not os.path.exists(out_dir): os.makedirs(out_dir)
    # for f in txt:
    #     t = open(os.path.join("txt/p225", f), "r")  
    #     text = t.readline()[:-1]
    txtf = open("../ljs_audio_text_test_filelist.txt", 'r')
    txt = txtf.readlines()
    out_dir = args.save_dir
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    count = 0
    for text in txt:
        count += 1
        text = text.split("|")[-1]
        # ids = raw_texts = [text[:100]]
        ids = [str(count)]
        raw_texts = [text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
    
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(
                    *(batch[2:]),
                    p_control=pitch_control,
                    e_control=energy_control,
                    d_control=duration_control
                )
                # add here
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
                my_synth_samples2(
                    batch,
                    output,
                    vocoder,
                    model_config,
                    preprocess_config,
                    # "output/result/LJSpeech_test_1_1",
                    # "output/result/gl_or",
                    out_dir,
                    ap,
                )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["batch", "single"],
        # required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    
    parser.add_argument(
        "-myt",
        "--my_text_dir",
        type=str,
        default="./txt/p225",
        help="path to text.txt",
    )
    parser.add_argument(
        "-sd",
        "--save_dir",
        type=str,
        default="./syned/griffinlim_syned",
        help="path to save syned wav",
    )
    parser.add_argument(
        "-vd",
        "--vocoder_dir",
        type=str,
        default="../Hifi-GAN/ckpt_or/g_02420000",
        help="path to save syned wav",
    )
    args = parser.parse_args()

    # Check source texts
    # if args.mode == "batch":
    #     assert args.source is not None and args.text is None
    # if args.mode == "single":
    #     assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(args, model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    
    
    # if args.mode == "single":
    #     ids = raw_texts = [args.text[:100]]
    #     speakers = np.array([args.speaker_id])
    #     if preprocess_config["preprocessing"]["text"]["language"] == "en":
    #         texts = np.array([preprocess_english(args.text, preprocess_config)])
    #     elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
    #         texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
    #     text_lens = np.array([len(texts[0])])
    #     batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    
    txt = args.my_text_dir
    mysynthesize(model, args.restore_step, configs, vocoder, txt, control_values, args)
