import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from hparams import HParams

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, residual=False):
        super(ConvNorm, self).__init__()
        self.residual = residual
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        pre_x = x

        x = self.conv(x)
        x = self.batch_norm(x)

        if self.residual:
            x += pre_x

        x = F.relu(x)

        return x

class Encoder(nn.Module):
    hparams: HParams

    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.hparams = hparams
        # todo: make this parameterizable
        self.convolutions = nn.Sequential(
            ConvNorm(3, 6, 5, (1,2,2), (2,1,1)),
            ConvNorm(6, 6, 3, (1,1,1), (1,1,1), residual=True),
            ConvNorm(6, 6, 3, (1,1,1), (1,1,1), residual=True),

            ConvNorm(6, 12, 3, (1,2,2), (1,0,0)),
            ConvNorm(12, 12, 3, (1,1,1), (1,1,1), residual=True),
            ConvNorm(12, 12, 3, (1,1,1), (1,1,1), residual=True),

            ConvNorm(12, 24, 3, (1,2,2), (1,0,0)),
            ConvNorm(24, 24, 3, (1,1,1), (1,1,1), residual=True),
            ConvNorm(24, 24, 3, (1,1,1), (1,1,1), residual=True),

            ConvNorm(24, 24, 3, (1,3,3), (1,0,0)),
        )
        self.lstm = nn.LSTM(24, hparams.encoder_hidden_size, hparams.encoder_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.convolutions(x)
        x = reduce(x, 'b c t 1 1 -> b t c', 'mean')
        x, hidden = self.lstm(x)
        return x, hidden

class Decoder(nn.Module):
    hparams: HParams

    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.hparams = hparams

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=hparams.decoder_hidden_size, nhead=8, batch_first=True),
            num_layers=4)
        self.pos_enc = PositionalEncoding(hparams.decoder_hidden_size, dropout=0.1, max_len=hparams.temporal_dim)
        # self.attention = BahdanauAttention(hparams.decoder_hidden_size, hparams.encoder_lip_embedding_size)
        # self.lstm = nn.LSTM(input_size, hparams.decoder_hidden_size, hparams.decoder_layers, batch_first=True)
        self.projection = nn.Linear(hparams.decoder_hidden_size, hparams.n_mels)

    def forward(self, x):
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.projection(x)
        return x

class MerkelNet(nn.Module):
    hparams: HParams

    def __init__(self, hparams):
        super(MerkelNet, self).__init__()
        self.hparams = hparams
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)

    def forward(self, x):
        x, _ = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    e = Decoder(HParams())
    x = torch.randn(1, 50, 256)
    y = e(x)
    print(y.shape)