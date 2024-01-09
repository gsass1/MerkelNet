import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os.path
from torchinfo import summary
from argparse import ArgumentParser

T = 50
SR = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FPS = 25
F_MAX = 8000
W, H = 48, 48
DATASET_BATCH_SIZE = 32

DATA_DIR = 'data'

parser = ArgumentParser(
                    prog='train',
                    description='Train the network')

parser.add_argument('data_dir', default=DATA_DIR)
args = parser.parse_args()

if not os.path.isdir(args.data_dir):
    print('Data directory does not exist')
    exit(1)

class MerkelDataset(Dataset):
    def __init__(self, data_dir):
        self.filepaths = []
        for f in os.listdir(data_dir):
            if f.endswith('.npz'):
                self.filepaths.append(os.path.join(data_dir, f))

    def __len__(self):
        return len(self.filepaths) * DATASET_BATCH_SIZE 


    def __getitem__(self, idx):
        file_idx = idx // DATASET_BATCH_SIZE 
        data_idx = idx % DATASET_BATCH_SIZE 

        data = np.load(self.filepaths[file_idx])
        X = torch.from_numpy(data['X'][data_idx])
        Y = torch.from_numpy(data['Y'][data_idx])

        # oops, forgot to do this in make_dataset.py

        # T, H, W, C
        # to
        # C, T, H, W

        X = X.permute(3, 0, 1, 2).to(torch.float32)

        return X, Y


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

class LipEncoder(nn.Module):
    def __init__(self):
        super(LipEncoder, self).__init__()
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

    def forward(self, x):
        x = self.seq(x)

        #print('x', x.shape)
        x = x.squeeze(-1).squeeze(-1)
        #print('x after squeeze', x.shape)
        x = x.view(-1, T, 24)
        #print(x.shape)
        return x

class LipDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(LipDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bi-directional LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Fully connected layer to transform the output dimension
        self.fc = nn.Linear(hidden_size * 2, output_dim)  # Multiply hidden_size by 2 for bi-directional

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)  # 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size * 2)

        # Transform to output dimension at each time step
        out = self.fc(out)

        return out

class MerkelNet(nn.Module):
    def __init__(self):
        super(MerkelNet, self).__init__()
        self.encoder = LipEncoder()
        self.decoder = LipDecoder(24, 64, 2, 128)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Training requirements
model = MerkelNet()
print(summary(model, input_size=(1, 3, T, 48, 48)))

dataset = MerkelDataset(args.data_dir)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(50):
    for batch_idx, (X, Y) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, batch {batch_idx}, loss {loss.item()}')
