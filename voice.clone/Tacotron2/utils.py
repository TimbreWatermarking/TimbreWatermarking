import numpy as np
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    # filepaths_and_text = []
    # count = 0
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    #     line = f.readline()
    #     while line:
    #         count += 1
    #         try:
    #             line = f.readline()
    #             filepaths_and_text.append(line.strip().split(split))
    #         except Exception as e:
    #             print("#"*100)
    #             print(f"line:{count}")
            
        # for line in f.readlines():
        #     count += 1
        #     try:
        #         filepaths_and_text.append(line.strip().split(split))
        #     except Exception as e:
        #         print(f"line:{count}\ncontent:{line}")
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
