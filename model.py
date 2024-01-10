import torch
import torch.nn as nn
from einops import reduce
from hparams import HParams

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, residual=False):
        super(ConvNorm, self).__init__()
        self.residual = residual
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        pre_x = x

        x = self.conv(x)
        x = self.batch_norm(x)

        if self.residual:
            x += pre_x

        x = nn.functional.relu(x)

        return x

def calculate_padding(kernel_size, stride):
    return [
            (kernel_size - stride[0]) // 2,  # Padding for depth dimension
            (kernel_size - stride[1]) // 2,  # Padding for height dimension
            (kernel_size - stride[2]) // 2   # Padding for width dimension
        ]

class Encoder(nn.Module):
    hparams: HParams

    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.hparams = hparams
        self.convolutions = nn.Sequential(
            ConvNorm(3, 6, 5, (1,2,2), (2,1,1)),
            #ConvNorm(6, 6, 3, (1,1,1), (1,1,1), residual=True),
            #ConvNorm(6, 6, 3, (1,1,1), (1,1,1), residual=True),

            ConvNorm(6, 12, 3, (1,2,2), (1,0,0)),
            #ConvNorm(12, 12, 3, (1,1,1), (1,1,1), residual=True),
            #ConvNorm(12, 12, 3, (1,1,1), (1,1,1), residual=True),

            ConvNorm(12, 24, 3, (1,2,2), (1,0,0)),
            #ConvNorm(24, 24, 3, (1,1,1), (1,1,1), residual=True),
            #ConvNorm(24, 24, 3, (1,1,1), (1,1,1), residual=True),

            ConvNorm(24, 24, 3, (1,3,3), (1,0,0)),
        )
        self.lstm = nn.LSTM(24, 32, 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.convolutions(x)
        x = reduce(x, 'b c t 1 1 -> b t c', 'mean')
        x, _ = self.lstm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

class MerkelNet(nn.Module):
    hparams: HParams

    def __init__(self, hparams):
        super(MerkelNet, self).__init__()
        self.hparams = hparams
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(64, 64, 2, self.hparams.n_mels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    e = Encoder(HParams())
    x = torch.randn(1, 3, 50, 48, 48)
    print(e(x))
