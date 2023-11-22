from concurrent.futures import process
import os
import torch
import julius
import torchaudio
from torch.utils.data import Dataset
import random
import librosa


class twod_dataset(Dataset):
    def __init__(self, process_config, train_config):
        self.dataset_name = train_config["dataset"]
        self.dataset_path = train_config["path"]["raw_path"]
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        self.wavs = self.process_meta()
    
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        audio_name = self.wavs[idx]
        wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
        wav = wav[:,:self.max_len]
        patch_num = wav.shape[1] // self.win_len
        pad_num = (patch_num + 1)*self.win_len - wav.shape[1]
        if pad_num == self.win_len:
            pad_num = 0
            wav_matrix = wav.reshape(-1, self.win_len)
        else:
            wav_matrix = torch.cat((wav,torch.zeros(1,pad_num)), dim=1).reshape(-1, self.win_len)
        sample = {
            "matrix": wav_matrix,
            "sample_rate": sr,
            "patch_num": patch_num,
            "pad_num": pad_num,
            "name": audio_name
        }
        return sample

    def process_meta(self):
        wavs = os.listdir(self.dataset_path)
        return wavs


class oned_dataset(Dataset):
    def __init__(self, process_config, train_config):
        self.dataset_name = train_config["dataset"]
        self.dataset_path = train_config["path"]["raw_path"]
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        self.wavs = self.process_meta()
    
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        audio_name = self.wavs[idx]
        wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
        wav = wav[:,:self.max_len]
        sample = {
            "matrix": wav,
            "sample_rate": sr,
            "patch_num": 0,
            "pad_num": 0,
            "name": audio_name
        }
        return sample

    def process_meta(self):
        wavs = os.listdir(self.dataset_path)
        return wavs


# pre-load dataset and resample to 22.05KHz
class mel_dataset(Dataset):
    def __init__(self, process_config, train_config):
        self.dataset_name = train_config["dataset"]
        self.dataset_path = train_config["path"]["raw_path"]
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        self.wavs = self.process_meta()
        # n_fft = process_config["mel"]["n_fft"]
        # hop_length = process_config["mel"]["hop_length"]
        # self.stft = STFT(n_fft, hop_length)

        sr = process_config["audio"]["or_sample_rate"]
        self.resample = torchaudio.transforms.Resample(sr,self.sample_rate)
        self.sample_list = []
        for idx in range(len(self.wavs)):
            audio_name = self.wavs[idx]
            # import pdb
            # pdb.set_trace()
            wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
            if wav.shape[1] > self.max_len:
                cuted_len = random.randint(5*sr, self.max_len)
                wav = wav[:, :cuted_len]
            wav = self.resample(wav[0,:].view(1,-1))
            # wav = wav[:,:self.max_len]
            sample = {
                "matrix": wav,
                "sample_rate": sr,
                "patch_num": 0,
                "pad_num": 0,
                "name": audio_name
            }
            self.sample_list.append(sample)
    
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        return self.sample_list[idx]

    def process_meta(self):
        wavs = os.listdir(self.dataset_path)
        return wavs



class mel_dataset_test(Dataset):
    def __init__(self, process_config, train_config):
        self.dataset_name = train_config["dataset"]
        self.dataset_path = train_config["path"]["raw_path_test"]
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        self.wavs = self.process_meta()
        self.resample = julius.ResampleFrac(22050, 16000)
        # n_fft = process_config["mel"]["n_fft"]
        # hop_length = process_config["mel"]["hop_length"]
        # self.stft = STFT(n_fft, hop_length)
    
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        audio_name = self.wavs[idx]
        wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
        # wav = self.resample(wav)
        # wav = wav[:,:self.max_len]
        # spect, phase = self.stft.transform(wav)
        sample = {
            "matrix": wav,
            "sample_rate": sr,
            "patch_num": 0,
            "pad_num": 0,
            "name": audio_name
        }
        return sample

    def process_meta(self):
        # wavs = os.listdir(self.dataset_path)
        # return wavs
        wav_files = []
        for filename in os.listdir(self.dataset_path):
            if filename.endswith('.wav'):
                wav_files.append(filename)
        return wav_files



class mel_dataset_test_2(Dataset):
    def __init__(self, process_config, train_config):
        self.dataset_name = train_config["dataset"]
        self.dataset_path = train_config["path"]["raw_path_test"]
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        self.wavs = self.process_meta()
    
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        audio_name = self.wavs[idx]
        wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
        wav, sr2 = librosa.load(os.path.join(self.dataset_path, audio_name), sr=self.sample_rate)
        wav = torch.Tensor(wav).unsqueeze(0)
        # wav = self.resample(wav)
        # wav = wav[:,:self.max_len]
        # spect, phase = self.stft.transform(wav)
        sample = {
            "matrix": wav,
            "sample_rate": sr,
            "patch_num": 0,
            "pad_num": 0,
            "name": audio_name
        }
        return sample

    def process_meta(self):
        # wavs = os.listdir(self.dataset_path)
        # return wavs
        wav_files = []
        for filename in os.listdir(self.dataset_path):
            if filename.endswith('.wav'):
                wav_files.append(filename)
        return wav_files


# pre-load dataset and resample to 22.05KHz
class wav_dataset(Dataset):
    def __init__(self, process_config, train_config, flag='train'):
        self.dataset_name = train_config["dataset"]
        raw_dataset_path = train_config["path"]["raw_path"]
        self.dataset_path = os.path.join(raw_dataset_path, flag)
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        # self.wavs = self.process_meta()[:10]
        self.wavs = self.process_meta()
        # n_fft = process_config["mel"]["n_fft"]
        # hop_length = process_config["mel"]["hop_length"]
        # self.stft = STFT(n_fft, hop_length)

        sr = process_config["audio"]["or_sample_rate"]
        self.resample = torchaudio.transforms.Resample(sr,self.sample_rate)
        self.sample_list = []
        for idx in range(len(self.wavs)):
            audio_name = self.wavs[idx]
            wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
            if wav.shape[1] > self.max_len:
                cuted_len = random.randint(5*sr, self.max_len)
                wav = wav[:, :cuted_len]
            wav = self.resample(wav[0,:].view(1,-1))
            # wav = wav[:,:self.max_len]
            sample = {
                "matrix": wav,
                "sample_rate": sr,
                "patch_num": 0,
                "pad_num": 0,
                "name": audio_name
            }
            self.sample_list.append(sample)
    
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        return self.sample_list[idx]

    def process_meta(self):
        wavs = os.listdir(self.dataset_path)
        return wavs



class wav_dataset_wopreload(Dataset):
    def __init__(self, process_config, train_config, flag='train'):
        self.dataset_name = train_config["dataset"]
        raw_dataset_path = train_config["path"]["raw_path"]
        self.dataset_path = os.path.join(raw_dataset_path, flag)
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        # self.wavs = self.process_meta()[:10]
        self.wavs = self.process_meta()
        # n_fft = process_config["mel"]["n_fft"]
        # hop_length = process_config["mel"]["hop_length"]
        # self.stft = STFT(n_fft, hop_length)

        sr = process_config["audio"]["or_sample_rate"]
        self.resample = torchaudio.transforms.Resample(sr,self.sample_rate)
    
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        audio_name = self.wavs[idx]
        # import pdb
        # pdb.set_trace()
        wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
        if wav.shape[1] > self.max_len:
            cuted_len = random.randint(5*sr, self.max_len)
            wav = wav[:, :cuted_len]
        wav = self.resample(wav[0,:].view(1,-1))
        # wav = wav[:,:self.max_len]
        sample = {
            "matrix": wav,
            "sample_rate": sr,
            "patch_num": 0,
            "pad_num": 0,
            "name": audio_name
        }
        return sample

    def process_meta(self):
        wavs = os.listdir(self.dataset_path)
        return wavs
    

class wav_dataset_test(Dataset):
    def __init__(self, process_config, train_config, flag='train'):
        self.dataset_name = train_config["dataset"]
        raw_dataset_path = train_config["path"]["raw_path"]
        self.dataset_path = os.path.join(raw_dataset_path, flag)
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        # self.wavs = self.process_meta()[:2]
        self.wavs = self.process_meta()
        # n_fft = process_config["mel"]["n_fft"]
        # hop_length = process_config["mel"]["hop_length"]
        # self.stft = STFT(n_fft, hop_length)

        sr = process_config["audio"]["or_sample_rate"]
        self.resample = torchaudio.transforms.Resample(sr,self.sample_rate)
        self.sample_list = []
        for idx in range(len(self.wavs)):
            audio_name = self.wavs[idx]
            # import pdb
            # pdb.set_trace()
            wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
            # if wav.shape[1] > self.max_len:
            #     cuted_len = random.randint(5*sr, self.max_len)
            #     wav = wav[:, :cuted_len]
            wav = self.resample(wav[0,:].view(1,-1))
            # wav = wav[:,:self.max_len]
            sample = {
                "matrix": wav,
                "sample_rate": sr,
                "patch_num": 0,
                "pad_num": 0,
                "name": audio_name
            }
            self.sample_list.append(sample)
    
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        return self.sample_list[idx]

    def process_meta(self):
        wavs = os.listdir(self.dataset_path)
        return wavs




class wav_dataset_librosa(Dataset):
    def __init__(self, process_config, train_config, flag='train'):
        self.dataset_name = train_config["dataset"]
        raw_dataset_path = train_config["path"]["raw_path"]
        self.dataset_path = os.path.join(raw_dataset_path, flag)
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        # self.wavs = self.process_meta()[:10]
        self.wavs = self.process_meta()
        # n_fft = process_config["mel"]["n_fft"]
        # hop_length = process_config["mel"]["hop_length"]
        # self.stft = STFT(n_fft, hop_length)

        sr = process_config["audio"]["or_sample_rate"]
        # self.resample = torchaudio.transforms.Resample(sr,self.sample_rate)
        self.sample_list = []
        for idx in range(len(self.wavs)):
            audio_name = self.wavs[idx]
            # import pdb
            # pdb.set_trace()
            # wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
            wav, sr2 = librosa.load(os.path.join(self.dataset_path, audio_name), sr=self.sample_rate)
            wav = torch.Tensor(wav).unsqueeze(0)
            if wav.shape[1] > self.max_len:
                cuted_len = random.randint(5*sr, self.max_len)
                wav = wav[:, :cuted_len]
            # wav = self.resample(wav[0,:].view(1,-1))
            # wav = wav[:,:self.max_len]
            sample = {
                "matrix": wav,
                "sample_rate": sr,
                "patch_num": 0,
                "pad_num": 0,
                "name": audio_name
            }
            self.sample_list.append(sample)
    
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        return self.sample_list[idx]

    def process_meta(self):
        wavs = os.listdir(self.dataset_path)
        return wavs
