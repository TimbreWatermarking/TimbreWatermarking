from base64 import encode
import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from .blocks import FCBlock, PositionalEncoding, Mish, Conv1DBlock
from distortions.mel_transform import STFT
import pdb


class Encoder(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=6, transformer_drop=0.1, attention_heads=8):
        super(Encoder, self).__init__()
        self.name="mel_trans"
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        embedding_dim = win_dim
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=attention_heads, dropout=transformer_drop) 
        self.dec_encoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=attention_heads, dropout=transformer_drop) 

        self.encoder = nn.TransformerEncoder(self.encoder_layer, nlayers_encoder)
        self.decoder = nn.TransformerDecoder(self.dec_encoder_layer, nlayers_encoder)	

        #MLP for the input audio waveform
        self.wav_linear_in = FCBlock(win_dim, embedding_dim, activation=LeakyReLU(inplace=True))
        self.wav_linear_out = FCBlock(embedding_dim, win_dim)

        #MLP for the input wm
        self.msg_linear_in = FCBlock(msg_length, embedding_dim, activation=LeakyReLU(inplace=True))

        #position encoding
        self.pos_encoder = PositionalEncoding(d_model=embedding_dim, dropout=transformer_drop)

        #stft transform
        self.stft = STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"])

        # #Conv for wm
        # self.wm_conv = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             Conv1DBlock(embedding_dim, 2 * embedding_dim, model_config["wm"]["kernel_size"], activation=LeakyReLU(inplace=True)),
        #             nn.GLU(),
        #         )
        #         for _ in range(model_config["wm"]["n_temporal_layer"])
        #     ]
        # )
        # #Conv for wav
        # self.wav_conv = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             Conv1DBlock(embedding_dim, 2 * embedding_dim, model_config["audio"]["kernel_size"], activation=LeakyReLU(inplace=True)),
        #             nn.GLU(),
        #         )
        #         for _ in range(model_config["audio"]["n_temporal_layer"])
        #     ]
        # )

    def forward_encode_msg(self, x, w):
        num_samples = x.shape[2]
        spect, phase = self.stft.transform(x)
        spect_t = spect.transpose(1,2)
        x_embedding = self.wav_linear_in(spect_t)
        # for _, layer in enumerate(self.wav_conv):
        #     residual = x_embedding
        #     x_embedding = layer(x_embedding)
        #     x_embedding = residual + x_embedding
        # p_x = self.pos_encoder(x_embedding)
        p_x = x_embedding
        encoder_out = self.encoder(p_x.transpose(0,1)).transpose(0,1)   # tgt_len, bsz, embed_dim = query.size()
        # Temporal Average Pooling
        wav_feature = torch.mean(encoder_out, dim=1, keepdim=True) # [B, 1, H]
        msg_feature = self.msg_linear_in(w)
        # for _, layer in enumerate(self.wm_conv):
        #     residual = msg_feature
        #     msg_feature = layer(msg_feature)
        #     msg_feature = msg_feature + residual
        encoded_msg = wav_feature.add(msg_feature)
        return encoded_msg, encoder_out, p_x, phase, num_samples, spect
    
    def forward_decode_wav(self, encoded_msg, encoder_out, p_x):
        # B, _, D = encoded_msg.shape
        encode_msg_repeat = encoded_msg.repeat(1, p_x.size(1), 1)
        embeded = self.decoder((encode_msg_repeat + p_x).transpose(0,1), memory=encoder_out.transpose(0,1)).transpose(0,1)
        wav_out = self.wav_linear_out(embeded)
        return wav_out

    def forward(self, x, w):
        encoded_msg, encoder_out, p_x, phase, num_samples, spect = self.forward_encode_msg(x, w)
        wav_out = self.forward_decode_wav(encoded_msg, encoder_out, p_x)
        self.stft.num_samples = num_samples
        noised_wav = self.stft.inverse(wav_out.transpose(1,2), phase)
        # noised_wav = self.stft.inverse(wav_out.transpose(1,2)+spect, phase)
        return noised_wav



class Decoder(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=6, transformer_drop=0.1, attention_heads=8):
        super(Decoder, self).__init__()
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        embedding_dim = win_dim
        self.msg_decoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=attention_heads, dropout=transformer_drop) 
        self.msg_decoder = nn.TransformerEncoder(self.msg_decoder_layer, nlayers_decoder)
        self.msg_linear_out = FCBlock(embedding_dim, msg_length)
        #MLP for the input audio waveform
        self.wav_linear_in = FCBlock(win_dim, embedding_dim, activation=LeakyReLU(inplace=True))
        #position encoding
        self.pos_encoder = PositionalEncoding(d_model=embedding_dim, dropout=transformer_drop)

        #stft transform
        self.stft = STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"])
        # #Conv for wm
        # self.wm_conv = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             Conv1DBlock(embedding_dim, 2 * embedding_dim, model_config["wm"]["kernel_size"], activation=LeakyReLU(inplace=True)),
        #             nn.GLU(),
        #         )
        #         for _ in range(model_config["wm"]["n_temporal_layer"])
        #     ]
        # )
        # #Conv for wav
        # self.wav_conv = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             Conv1DBlock(embedding_dim, 2 * embedding_dim, model_config["audio"]["kernel_size"], activation=LeakyReLU(inplace=True)),
        #             nn.GLU(),
        #         )
        #         for _ in range(model_config["audio"]["n_temporal_layer"])
        #     ]
        # )

    def forward(self, x):
        spect, phase = self.stft.transform(x)
        spect = spect.transpose(1,2)
        x_embedding = self.wav_linear_in(spect)
        # for _, layer in enumerate(self.wav_conv):
        #     residual = x_embedding
        #     x_embedding = layer(x_embedding)
        #     x_embedding = residual + x_embedding
        p_x = self.pos_encoder(x_embedding)
        encoder_out = self.msg_decoder(p_x.transpose(0,1)).transpose(0,1)
        # Temporal Average Pooling
        wav_feature = torch.mean(encoder_out, dim=1, keepdim=True) # [B, 1, H]
        # for _, layer in enumerate(self.wm_conv):
        #     residual = wav_feature
        #     wav_feature = layer(wav_feature)
        #     wav_feature = wav_feature + residual
        out_msg = self.msg_linear_out(wav_feature)
        return out_msg


class Discriminator(nn.Module):
    def __init__(self, msg_length, win_dim, embedding_dim, nlayers_decoder=6, transformer_drop=0.1, attention_heads=8):
        super(Decoder, self).__init__()
        self.msg_decoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=attention_heads, dropout=transformer_drop) 
        self.msg_decoder = nn.TransformerEncoder(self.msg_decoder_layer, nlayers_decoder)
        self.msg_linear_out = FCBlock(embedding_dim, msg_length)
        #MLP for the input audio waveform
        self.wav_linear_in = FCBlock(win_dim, embedding_dim, activation=Mish())
        #position encoding
        self.pos_encoder = PositionalEncoding(d_model=embedding_dim, dropout=transformer_drop)

    def forward(self, x):
        x_embedding = self.wav_linear_in(x)
        p_x = self.pos_encoder(x_embedding)
        encoder_out = self.msg_decoder(p_x)
        # Temporal Average Pooling
        wav_feature = torch.mean(encoder_out, dim=1, keepdim=True) # [B, 1, H]
        out_msg = self.msg_linear_out(wav_feature)
        return torch.mean(out_msg)


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param