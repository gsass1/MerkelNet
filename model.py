import torch.nn as nn
from hparams import HParams

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, residual=False):
        super(ConvBlock, self).__init__()
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
        self.blocks = []
        NUM_BLOCKS = 4
        c = 3
        for i in range(NUM_BLOCKS):
            kernel_size = (5 if i == 0 else 3)
            stride = (1, 2, 2)  # Stride
            padding = calculate_padding(kernel_size, stride)
            block = ConvBlock((c if i == 0 else c//2), c, kernel_size, stride, padding)
            self.blocks.append(block)

            for _ in range(2):
                kernel_size = 3
                stride = (1, 1, 1)
                padding = calculate_padding(kernel_size, stride)

                block1 = ConvBlock(c, c, kernel_size, (1, 1, 1), padding, residual=True)
                block2 = ConvBlock(c, c, kernel_size, (1, 1, 1), padding, residual=True)

                self.blocks.append(block1)
                self.blocks.append(block2)

            if i == NUM_BLOCKS - 1:
                stride = (1, 3, 3)
                padding = calculate_padding(kernel_size, stride)

                last_block = ConvBlock(c, c, kernel_size=(1, 2, 2), stride=1, padding=0)
                self.blocks.append(last_block)

            c *= 2
        self.seq = nn.Sequential(*self.blocks)
        self.lstm = nn.LSTM(24, 64, 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.seq(x)

        #print('x', x.shape)
        x = x.squeeze(-1).squeeze(-1)
        #print('x after squeeze', x.shape)
        x = x.view(-1, self.hparams.temporal_dim, 24)
        #print(x.shape)
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
        self.decoder = Decoder(128, 64, 4, 128)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


