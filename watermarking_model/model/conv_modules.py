from base64 import encode
from distutils.command.config import config
import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from .blocks import FCBlock, PositionalEncoding, Mish, Conv1DBlock


class Encoder(nn.Module):
    def __init__(self, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=6, transformer_drop=0.1, attention_heads=8):
        super(Encoder, self).__init__()
        self.name = "conv1"
        self.hidden_size = model_config["conv_module"]["hidden_size"]
        self.wav_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1DBlock(1, self.hidden_size, model_config["conv_module"]["kernel_size"], activation=LeakyReLU(inplace=True)),
                    nn.InstanceNorm1d(num_features=self.hidden_size, affine=True),
                )
            ] +[
                nn.Sequential(
                    Conv1DBlock(self.hidden_size, self.hidden_size, model_config["conv_module"]["kernel_size"], activation=LeakyReLU(inplace=True)),
                    nn.InstanceNorm1d(num_features=self.hidden_size, affine=True),
                )
                for _ in range(model_config["conv_module"]["n_temporal_layer"])
            ] + [nn.Sequential(
                    Conv1DBlock(self.hidden_size, msg_length, model_config["conv_module"]["kernel_size"], activation=LeakyReLU(inplace=True)),
                    nn.InstanceNorm1d(num_features=msg_length, affine=True),
                )
            ]
        )
        self.embedder = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1DBlock(msg_length, self.hidden_size, model_config["conv_module"]["kernel_size"], activation=LeakyReLU(inplace=True)),
                    nn.InstanceNorm1d(num_features=self.hidden_size, affine=True),
                )
            ] +[
                nn.Sequential(
                    Conv1DBlock(self.hidden_size, self.hidden_size, model_config["conv_module"]["kernel_size"], activation=LeakyReLU(inplace=True)),
                    nn.InstanceNorm1d(num_features=self.hidden_size, affine=True),
                )
                for _ in range(model_config["conv_module"]["n_temporal_layer"])
            ] + [nn.Sequential(
                    Conv1DBlock(self.hidden_size, 1, model_config["conv_module"]["kernel_size"], activation=LeakyReLU(inplace=True)),
                    nn.InstanceNorm1d(num_features=1, affine=True),
                )
            ]
        )

    def forward(self, x, w):
        wav_feature = self.wav_encoder[0](x)
        for _, layer in enumerate(self.wav_encoder):
            if _ != 0 and _ != len(self.wav_encoder)-1:
                residual = wav_feature
                wav_feature = layer(wav_feature)
                wav_feature = residual + wav_feature
        wav_feature = self.wav_encoder[-1](wav_feature)
        add_feature = wav_feature + w.transpose(1,2)
        # add_feature = w.transpose(1,2).repeat(1,1,wav_feature.shape[2])
        add_feature = self.embedder[0](add_feature)
        for _, layer in enumerate(self.embedder):
            if _ != 0 and _ != len(self.embedder)-1:
                residual = add_feature
                add_feature = layer(add_feature)
                add_feature = residual + add_feature
        # out = self.embedder[-1](add_feature)
        out = self.embedder[-1](add_feature) + x
        return out
        



class Decoder(nn.Module):
    def __init__(self, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=6, transformer_drop=0.1, attention_heads=8):
        super(Decoder, self).__init__()
        embedding_dim = model_config["conv_module"]["hidden_dim"]
        self.hidden_size = model_config["conv_module"]["hidden_size"]
        self.wav_encoder = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1DBlock(1, self.hidden_size, model_config["conv_module"]["kernel_size"], activation=LeakyReLU(inplace=True)),
                    nn.InstanceNorm1d(num_features=self.hidden_size, affine=True),
                )
            ] + [
                nn.Sequential(
                    Conv1DBlock(self.hidden_size, self.hidden_size, model_config["conv_module"]["kernel_size"], activation=LeakyReLU(inplace=True)),
                    nn.InstanceNorm1d(num_features=self.hidden_size, affine=True),
                )
                for _ in range(model_config["conv_module"]["n_temporal_layer"]*2)
            ] + [
                nn.Sequential(
                    Conv1DBlock(self.hidden_size, msg_length, model_config["conv_module"]["kernel_size"], activation=LeakyReLU(inplace=True)),
                    nn.InstanceNorm1d(num_features=msg_length, affine=True),
                )
            ]
        )
        self.msg_linear = nn.ModuleList(
            [
                FCBlock(msg_length, embedding_dim, activation=LeakyReLU(inplace=True))
            ] +
            [
                FCBlock(embedding_dim, embedding_dim, activation=LeakyReLU(inplace=True)) for _ in range(model_config["conv_module"]["n_linear_layer"])
            ] +
            [
                FCBlock(embedding_dim, msg_length, activation=LeakyReLU(inplace=True))
            ]
        )
        

    def forward(self, x):
        wav_feature = self.wav_encoder[0](x)
        for _, layer in enumerate(self.wav_encoder):
            if _ != 0 and _ != len(self.wav_encoder)-1:
                residual = wav_feature
                wav_feature = layer(wav_feature)
                wav_feature = residual + wav_feature
        wav_feature = self.wav_encoder[-1](wav_feature)
        msg_feature = torch.mean(wav_feature, dim=2, keepdim=True).transpose(1,2)
        msg_feature = self.msg_linear[0](msg_feature)
        for _, layer in enumerate(self.msg_linear):
            if _ != 0 and _ != len(self.msg_linear)-1:
                residual = msg_feature
                msg_feature = layer(msg_feature)
                msg_feature = residual + msg_feature
        out = self.msg_linear[-1](msg_feature)
        return out


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