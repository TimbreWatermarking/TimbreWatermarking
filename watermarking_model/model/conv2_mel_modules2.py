from base64 import encode
import torch
import torch.nn as nn
from torch.nn import LeakyReLU, Tanh
from .blocks import FCBlock, PositionalEncoding, Mish, Conv1DBlock, Conv2Encoder, WatermarkEmbedder, WatermarkExtracter, ReluBlock
from distortions.frequency import TacotronSTFT, fixed_STFT, tacotron_mel
from distortions.dl import distortion
import pdb
import hifigan
import json
import torchaudio


def save_spectrum(spect, phase, flag='linear'):
    import numpy as np
    import os
    import librosa
    import librosa.display
    root = "draw_figure"
    import matplotlib.pyplot as plt
    spect = spect/torch.max(torch.abs(spect))
    spec = librosa.amplitude_to_db(spect.squeeze(0).cpu().numpy(), ref=np.max, amin=1e-5)
    img=librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='log', y_coords=None);
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_amplitude_spectrogram.png'), bbox_inches='tight', pad_inches=0.0)
    phase = phase/torch.max(torch.abs(phase))
    spec = librosa.amplitude_to_db(phase.squeeze(0).cpu().numpy(), ref=np.max, amin=1e-5)
    img=librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='log', y_coords=None);
    plt.clim(-40, 40)
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_phase_spectrogram.png'), bbox_inches='tight', pad_inches=0.0)

def save_feature_map(feature_maps):
    import os
    import matplotlib.pyplot as plt
    import librosa
    import numpy as np
    import librosa.display
    feature_maps = feature_maps.cpu().numpy()
    root = "draw_figure"
    output_folder = os.path.join(root,"feature_map_or")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    n_channels = feature_maps.shape[0]
    for channel_idx in range(n_channels):
        fig, ax = plt.subplots()
        ax.imshow(feature_maps[channel_idx, :, :], cmap='gray')
        ax.axis('off')
        output_file = os.path.join(output_folder, f'feature_map_channel_{channel_idx + 1}.png')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)

def save_waveform(a_tensor, flag='original'):
    import os
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile
    root = "draw_figure"
    y = a_tensor.cpu().numpy()
    soundfile.write(os.path.join(root, flag + "_waveform.wav"), y, samplerate=22050)
    D = librosa.stft(y)
    spectrogram = np.abs(D)
    img=librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=22050, x_axis='time', y_axis='log', y_coords=None);
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_amplitude_spectrogram_from_waveform.png'), bbox_inches='tight', pad_inches=0.0)



def get_vocoder(device):
    with open("hifigan/config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load("./hifigan/model/VCTK_V1/generator_v1")
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    freeze_model_and_submodules(vocoder)
    return vocoder

def freeze_model_and_submodules(model):
    for param in model.parameters():
        param.requires_grad = False

    for module in model.children():
        if isinstance(module, nn.Module):
            freeze_model_and_submodules(module)


class Encoder(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=6, transformer_drop=0.1, attention_heads=8):
        super(Encoder, self).__init__()
        self.name = "conv2"
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.add_carrier_noise = False
        self.block = model_config["conv2"]["block"]
        self.layers_CE = model_config["conv2"]["layers_CE"]
        self.EM_input_dim = model_config["conv2"]["hidden_dim"] + 1
        self.layers_EM = model_config["conv2"]["layers_EM"]

        self.vocoder_step = model_config["structure"]["vocoder_step"]
        #MLP for the input wm
        self.msg_linear_in = FCBlock(msg_length, win_dim, activation=LeakyReLU(inplace=True))

        #stft transform
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])

        self.ENc = Conv2Encoder(input_channel=1, hidden_dim = model_config["conv2"]["hidden_dim"], block=self.block, n_layers=self.layers_CE)

        self.EM = WatermarkEmbedder(input_channel=self.EM_input_dim, hidden_dim = model_config["conv2"]["hidden_dim"], block=self.block, n_layers=self.layers_EM)

    def forward(self, x, msg, global_step):
        num_samples = x.shape[2]
        spect, phase = self.stft.transform(x)
        
        carrier_encoded = self.ENc(spect.unsqueeze(1)) 
        watermark_encoded = self.msg_linear_in(msg).transpose(1,2).unsqueeze(1).repeat(1,1,1,carrier_encoded.shape[3])
        concatenated_feature = torch.cat((carrier_encoded, watermark_encoded), dim=1)  
        carrier_wateramrked = self.EM(concatenated_feature)  

        
        self.stft.num_samples = num_samples
        y = self.stft.inverse(carrier_wateramrked.squeeze(1), phase.squeeze(1))
        return y, carrier_wateramrked
    
    def test_forward(self, x, msg, strength_factor=1.0):
        num_samples = x.shape[2]
        spect, phase = self.stft.transform(x)
        
        carrier_encoded = self.ENc(spect.unsqueeze(1)) 
        watermark_encoded = self.msg_linear_in(msg).transpose(1,2).unsqueeze(1).repeat(1,1,1,carrier_encoded.shape[3])
        concatenated_feature = torch.cat((carrier_encoded, watermark_encoded*strength_factor), dim=1)  
        carrier_wateramrked = self.EM(concatenated_feature)  
        
        self.stft.num_samples = num_samples
        y = self.stft.inverse(carrier_wateramrked.squeeze(1), phase.squeeze(1))
        return y, carrier_wateramrked



class Decoder(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=6, transformer_drop=0.1, attention_heads=8):
        super(Decoder, self).__init__()
        self.robust = model_config["robust"]
        if self.robust:
            self.dl = distortion()

        self.mel_transform = TacotronSTFT(filter_length=process_config["mel"]["n_fft"], hop_length=process_config["mel"]["hop_length"], win_length=process_config["mel"]["win_length"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocoder_step = model_config["structure"]["vocoder_step"]

        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.block = model_config["conv2"]["block"]
        self.EX = WatermarkExtracter(input_channel=1, hidden_dim=model_config["conv2"]["hidden_dim"], block=self.block)
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])
        self.msg_linear_out = FCBlock(win_dim, msg_length)

    def forward(self, y, global_step):
        y_identity = y.clone()
        # pdb.set_trace()
        if global_step > self.vocoder_step:
            y_mel = self.mel_transform.mel_spectrogram(y.squeeze(1))
            # y = self.vocoder(y_mel)
            y_d = (self.mel_transform.griffin_lim(magnitudes=y_mel)).unsqueeze(1)
        else:
            y_d = y
        if self.robust:
            y_d_d = self.dl(y_d, self.robust)
        else:
            y_d_d = y_d
        spect, phase = self.stft.transform(y_d_d)

        extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
        msg = torch.mean(extracted_wm,dim=2, keepdim=True).transpose(1,2)
        msg = self.msg_linear_out(msg)

        spect_identity, phase_identity = self.stft.transform(y_identity)
        extracted_wm_identity = self.EX(spect_identity.unsqueeze(1)).squeeze(1)
        msg_identity = torch.mean(extracted_wm_identity,dim=2, keepdim=True).transpose(1,2)
        msg_identity = self.msg_linear_out(msg_identity)
        return msg, msg_identity
    
    def test_forward(self, y):
        spect, phase = self.stft.transform(y)
        extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
        msg = torch.mean(extracted_wm,dim=2, keepdim=True).transpose(1,2)
        msg = self.msg_linear_out(msg)
        return msg
    
    def save_forward(self, y):
        # save mel_spectrum
        y_mel = self.mel_transform.mel_spectrogram(y.squeeze(1))
        save_spectrum(y_mel, y_mel, 'mel')
        y, reconstruct_spec = self.mel_transform.griffin_lim(magnitudes=y_mel)
        y = y.unsqueeze(1)
        save_waveform(y.squeeze().squeeze(), 'distored')
        save_spectrum(reconstruct_spec, reconstruct_spec, 'recon')
        # y = (self.mel_transform.griffin_lim(magnitudes=y_mel)).unsqueeze(1)
        spect, phase = self.stft.transform(y)
        save_spectrum(spect, spect, 'distored')
        pdb.set_trace()
        extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
        msg = torch.mean(extracted_wm,dim=2, keepdim=True).transpose(1,2)
        msg = self.msg_linear_out(msg)
        return msg
    
    def mel_test_forward(self, spect):
        extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
        msg = torch.mean(extracted_wm,dim=2, keepdim=True).transpose(1,2)
        msg = self.msg_linear_out(msg)
        return msg
        


class Discriminator(nn.Module):
    def __init__(self, process_config):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
                ReluBlock(1,16,3,1,1),
                ReluBlock(16,32,3,1,1),
                ReluBlock(32,64,3,1,1),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
                )
        self.linear = nn.Linear(64,1)
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])

    def forward(self, x):
        spect, phase = self.stft.transform(x)
        x = spect.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(2).squeeze(2)
        x = self.linear(x)
        return x


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param