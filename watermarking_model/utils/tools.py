import os
import datetime
import torch

def save(path, epoch, encoder, decoder, en_de_op):
    now = datetime.datetime.now()
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "en_de_op": en_de_op._optimizer.state_dict(),
        },
        os.path.join(path,"none-"+ encoder.name + "_ep_{}_{}.pth.tar".format(epoch, now.strftime("%Y-%m-%d_%H_%M_%S"))),
    )


def save_op(path, epoch, encoder, decoder, en_de_op):
    if not os.path.exists(path): os.makedirs(path)
    now = datetime.datetime.now()
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "en_de_op": en_de_op.state_dict(),
        },
        os.path.join(path,"none-"+ encoder.name + "_ep_{}_{}.pth.tar".format(epoch, now.strftime("%Y-%m-%d_%H_%M_%S"))),
    )


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/msg", losses[0], step)
        logger.add_scalar("Loss/wav_loss", losses[1], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


from torch.nn.functional import mse_loss
from pypesq import pesq as pesq_val
from resemblyzer import preprocess_wav, VoiceEncoder
import numpy as np
import soundfile
import pdb
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import torchaudio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resamber_encoder = VoiceEncoder(device="cuda")
pesq_resampler = torchaudio.transforms.Resample(22050, 16000).to(device)


def cosine_similarity(x, y):
    x = np.array(x).reshape(-1)
    y = np.array(y).reshape(-1)
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

def fidelity(a, b, sr):
    if a.shape[2] > b.shape[2]:
        distored_b = torch.cat([b.detach(), torch.zeros(b.shape[0],b.shape[1],a.shape[2]-b.shape[2]).to(device)], dim=2)
        snr_score = 10 * torch.log10(mse_loss(a.detach().cpu(), torch.zeros(a.shape)) / mse_loss(a.detach().cpu(), distored_b.detach().cpu()))
        pesq_score = pesq_val(pesq_resampler(a).detach().cpu().squeeze(0).squeeze(0).numpy(), pesq_resampler(distored_b).detach().cpu().squeeze(0).squeeze(0).numpy(), fs=16000)
    elif a.shape[2] < b.shape[2]:
        distored_b = b[:,:,:a.shape[2]]
        snr_score = 10 * torch.log10(mse_loss(a.detach().cpu(), torch.zeros(a.shape)) / mse_loss(a.detach().cpu(), distored_b.detach().cpu()))
        pesq_score = pesq_val(pesq_resampler(a).detach().cpu().squeeze(0).squeeze(0).numpy(), pesq_resampler(distored_b).detach().cpu().squeeze(0).squeeze(0).numpy(), fs=16000)
    else:
        snr_score = 10 * torch.log10(mse_loss(a.detach().cpu(), torch.zeros(a.shape)) / mse_loss(a.detach().cpu(), b.detach().cpu()))
        pesq_score = pesq_val(pesq_resampler(a).detach().cpu().squeeze(0).squeeze(0).numpy(), pesq_resampler(b).detach().cpu().squeeze(0).squeeze(0).numpy(), fs=16000)

    wav_a = preprocess_wav(a.cpu().squeeze(0).squeeze(0).detach().numpy(), source_sr=sr)
    wav_b = preprocess_wav(b.cpu().squeeze(0).squeeze(0).detach().numpy(), source_sr=sr)
    embeds_a = resamber_encoder.embed_utterance(wav_a)
    embeds_b = resamber_encoder.embed_utterance(wav_b)
    utterance_sim_matrix = cosine_similarity(embeds_a, embeds_b)
    spk_sim_matrix = utterance_sim_matrix
    # pdb.set_trace()
    return snr_score, pesq_score, utterance_sim_matrix, spk_sim_matrix
    


import math
def get_score(wav_matrix, decoder, msg):
    seg_num = 5
    segs_acc_true = []
    len_per_seg = math.floor(wav_matrix.shape[2]/seg_num)
    for i in range(seg_num):
        this_decoded = decoder.test_forward(wav_matrix[:, :, i*len_per_seg:(i+1)*len_per_seg])
        this_acc = (this_decoded >= 0).eq(msg >= 0).sum().float() / msg.numel()
        segs_acc_true.append(round(this_acc.cpu().numpy() * 100) / 100)
    return segs_acc_true

def get_score_2(wav_matrix, decoder, msg):
    seg_num = 5
    segs_acc_true = []
    len_per_seg = math.floor(wav_matrix.shape[2]/seg_num)
    for i in range(seg_num):
        this_decoded = decoder.test_forward(wav_matrix[:, :, i*len_per_seg:(i+1)*len_per_seg])
        this_bit = (this_decoded >= 0).eq(msg >= 0)
        segs_acc_true.append(this_bit.reshape(-1))
        # pdb.set_trace()
    def count_common_bits(list1, list2, list3, list4):
        if len(list1) != 10 or len(list2) != 10 or len(list3) != 10 or len(list4) != 10:
            return "All lists must have a length of 10."

        common_count = 0
        
        for i in range(10):
            if list1[i] == list2[i] == list3[i] == list4[i]:
                common_count += 1
                
        return common_count
    return [count_common_bits(segs_acc_true[0], segs_acc_true[1], segs_acc_true[2], segs_acc_true[3])]