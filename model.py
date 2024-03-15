import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from hparams import HParams

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout, skip_connection=False):
        super(ConvNorm, self).__init__()
        self.skip_connection = skip_connection 
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.dropout = dropout
  
    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = self.batch_norm(out)

        if self.skip_connection:
            out += identity

        out = F.relu(out)

        out = F.dropout(out, self.dropout, self.training)

        return out

class Prenet(nn.Module):
    hparams: HParams

    def __init__(self, hparams: HParams):
        super(Prenet, self).__init__()
        self.hparams = hparams
        self.fc = nn.Linear(hparams.n_mels, hparams.prenet_dim)

    # x - previous mel spectrogram: (batch, n_mels)
    def forward(self, x):
        x = F.dropout(F.relu(self.fc(x)), self.hparams.dropout, self.training) # (batch, prenet_dim)
        return x

def conv1d(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=int((kernel_size-1)/2)),
        nn.BatchNorm1d(out_channels))

class Postnet(nn.Module):
    hparams: HParams

    def __init__(self, hparams: HParams):
        super(Postnet, self).__init__()
        self.hparams = hparams
        self.convs = nn.ModuleList()

        self.convs.append(conv1d(hparams.n_mels, hparams.postnet_dim, hparams.postnet_kernel_size, 1))
        for _ in range(hparams.postnet_n_convs-1):
            self.convs.append(conv1d(hparams.postnet_dim, hparams.postnet_dim, hparams.postnet_kernel_size, 1))
        self.convs.append(conv1d(hparams.postnet_dim, hparams.n_mels, hparams.postnet_kernel_size, 1))

    def forward(self, x):
        """
        x: mel spectogram (batch, time, n_mels)
        """
        x = rearrange(x, 'b t c -> b c t')

        for conv in self.convs[:-1]:
            x = conv(x)
            x = F.dropout(F.tanh(x), self.hparams.postnet_dropout, self.training)
        x = F.dropout(self.convs[-1](x), self.hparams.postnet_dropout, self.training)
        return rearrange(x, 'b c t -> b t c')

class Attention(nn.Module):
    hparams: HParams

    def __init__(self, hparams: HParams):
        super(Attention, self).__init__()
        self.hparams = hparams

        # Memory FC
        self.M = nn.Linear(hparams.encoder_hidden_size, hparams.attn_dim, bias=False)

        # Query FC
        self.Q = nn.Linear(hparams.attn_hidden_size, hparams.attn_dim, bias=False)

        # Weight FC
        self.W = nn.Linear(hparams.attn_dim, 1, bias=False)

        # Location FC
        self.L = nn.Linear(hparams.attn_n_filters, hparams.attn_dim, bias=False)
        self.location_conv = nn.Conv1d(2, hparams.attn_n_filters, hparams.attn_kernel_size, padding=int((hparams.attn_kernel_size-1)/2), bias=False, stride=1)

    # query - previous decoder output: (batch, n_mels)
    # encoder_output: (batch, time, encoder_hidden_size)
    # prev_attn: cumulative and prev attn weights: (batch, 2, time)
    def forward(self, query, encoder_output, processed_encoder_output, prev_attn):
        """
        query: previous decoder output (batch, n_mels)
        encoder_output: (batch, time, encoder_hidden_size)
        prev_attn: cumulative and prev attn weights concatenated together: (batch, 2, time)
        """

        # Location-aware attention
        prev_attn = self.location_conv(prev_attn)
        prev_attn = rearrange(prev_attn, 'b c t -> b t c')
        prev_attn = self.L(prev_attn) # (batch, time, attn_dim)

        q = self.Q(query.unsqueeze(1)) # (batch, 1, attn_dim)

        #memory = self.M(encoder_output) # (batch, time, attn_dime)

        energies = self.W(torch.tanh(q + processed_encoder_output + prev_attn))
        energies = energies.squeeze(-1) # (batch, time)

        attn_weights = F.softmax(energies, dim=1)

        attn_context = torch.bmm(attn_weights.unsqueeze(1), encoder_output)
        attn_context = attn_context.squeeze(1) # (batch, encoder_hidden_size)

        return attn_context, attn_weights

class Encoder(nn.Module):
    hparams: HParams

    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.hparams = hparams

        self.convolutions = nn.Sequential(
            nn.Conv3d(3, 32, (5, 3, 3), (1, 2, 2), (2, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2), 0),
            nn.Dropout(hparams.dropout),

            nn.Conv3d(32, 64, (5, 3, 3), (1, 2, 2), (2, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2), 0),
            nn.Dropout(hparams.dropout),

            nn.Conv3d(64, 128, (5, 3, 3), (1, 2, 2), (2, 0, 0)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2), 0),
            nn.Dropout(hparams.dropout),
        )
        self.lstm = nn.LSTM(128, hparams.encoder_hidden_size//2, hparams.encoder_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        out = self.convolutions(x)
        out = reduce(out, 'b c t 1 1 -> b t c', 'mean')
        out, _ = self.lstm(out)
        return out

class Decoder(nn.Module):
    hparams: HParams

    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.hparams = hparams

        self.prenet = Prenet(hparams)

        self.attn = Attention(hparams)
        self.attn_lstm = nn.LSTMCell(hparams.encoder_hidden_size + hparams.prenet_dim, hparams.attn_hidden_size)

        self.lstm = nn.LSTMCell(hparams.attn_hidden_size + hparams.encoder_hidden_size, hparams.decoder_hidden_size)
        self.proj = nn.Linear(hparams.decoder_hidden_size + hparams.encoder_hidden_size, hparams.n_mels)

    def setup_states(self, encoder_output):
        """
        encoder_outputs: (batch, time, encoder_hidden_size)
        """

        device = encoder_output.device

        # Hidden state of the attention RNN
        self.attn_hidden = torch.zeros(encoder_output.size(0), self.hparams.attn_hidden_size).to(device)
        self.attn_cell = torch.zeros(encoder_output.size(0), self.hparams.attn_hidden_size).to(device)

        # Hidden states of the decoder RNN
        self.decoder_hidden = torch.zeros(encoder_output.size(0), self.hparams.decoder_hidden_size).to(device)
        self.decoder_cell = torch.zeros(encoder_output.size(0), self.hparams.decoder_hidden_size).to(device)

        # Attention states
        self.attn_weights = torch.zeros(encoder_output.size(0), encoder_output.size(1)).to(device)
        self.attn_weights_sum = torch.zeros(encoder_output.size(0), encoder_output.size(1)).to(device)
        self.attn_context = torch.zeros(encoder_output.size(0), self.hparams.encoder_hidden_size).to(device)

        # Calculate memory only once
        self.processed_encoder_output = self.attn.M(encoder_output) # (batch, 1, attn_dim)


    def forward(self, encoder_output, targets):
        """
        encoder_outputs: (batch, time, encoder_hidden_size)
        targets: ground truth mel spectograms (batch, time, n_mels)
        """

        device = encoder_output.device
        self.setup_states(encoder_output)

        # First decoder input is always just an empty vector
        first_decoder_input = torch.zeros(encoder_output.size(0), 1, self.hparams.n_mels).to(device)
        decoder_inputs = torch.cat((first_decoder_input, targets), dim=1)
        decoder_inputs = self.prenet(decoder_inputs)

        mel_outputs, alignments = [], []
        while len(mel_outputs) < decoder_inputs.size(1)-1:
            decoder_input = decoder_inputs[:, len(mel_outputs)]
            mel_output = self.decode(encoder_output, decoder_input)
            mel_outputs += [mel_output]
            alignments += [self.attn_weights.detach()]
        mel_outputs = torch.stack(mel_outputs, dim=1) # (batch, time, n_mels)
        alignments = torch.stack(alignments, dim=1) # (batch, decoder_time, encoder_time)
        return mel_outputs, alignments

    def decode(self, encoder_output, prenet_output):
        """
        encoder_outputs: (batch, time, encoder_hidden_size)
        decoder_input: previous processed mel spectogram (batch, prenet_dim)
        """

        # Attention LSTM forward pass by concatenating the previous decoder input and the prenet output
        cell_input = torch.cat((prenet_output, self.attn_context), dim=1)
        self.attn_hidden, self.attn_cell = self.attn_lstm(cell_input, (self.attn_hidden, self.attn_cell))
        self.attn_hidden = F.dropout(self.attn_hidden, self.hparams.dropout, self.training)

        # Perform location-aware attention
        attn_weights_cat = torch.cat((self.attn_weights.unsqueeze(1), self.attn_weights_sum.unsqueeze(1)), dim=1)
        self.attn_context, self.attn_weights = self.attn(self.attn_hidden, encoder_output, self.processed_encoder_output, attn_weights_cat)
        self.attn_weights_sum += self.attn_weights

        # Decoder LSTM forward pass by concatenating the attention context and the attention hidden state
        cell_input = torch.cat((self.attn_context, self.attn_hidden), dim=1)
        self.decoder_hidden, self.decoder_cell = self.lstm(cell_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.hparams.dropout, self.training)

        # Project the decoder hidden state to the mel spectogram
        proj_input = torch.cat((self.decoder_hidden, self.attn_context), dim=1)
        output = self.proj(proj_input)

        return output

    def inference(self, encoder_output):
        self.setup_states(encoder_output)
        decoder_input = torch.zeros(encoder_output.size(0), 1, self.hparams.n_mels).to(encoder_output.device)
        
        mel_outputs, alignments = [], []
        while len(mel_outputs) < encoder_output.size(1):
            decoder_input = self.prenet(decoder_input).squeeze(1)
            mel_output = self.decode(encoder_output, decoder_input)

            mel_outputs += [mel_output]
            alignments += [self.attn_weights.detach()]

            decoder_input = mel_output

        mel_outputs = torch.stack(mel_outputs, dim=1) # (batch, time, n_mels)
        alignments = torch.stack(alignments, dim=1) # (batch, decoder_time, encoder_time)
        return mel_outputs, alignments

class MerkelNet(nn.Module):
    hparams: HParams

    def __init__(self, hparams):
        super(MerkelNet, self).__init__()
        self.hparams = hparams
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def forward(self, frames, targets):
        """
        frames: (B, C, T, H ,W)
        targets: (B, T, n_mels)
        """

        encoder_output = self.encoder(frames) # (batch, time, encoder_hidden_size)

        decoder_output, alignments = self.decoder(encoder_output, targets) # (batch, time, n_mels)
        postnet_output = self.postnet(decoder_output) + decoder_output # (batch, time, n_mels)

        return decoder_output, postnet_output, alignments

    def inference(self, frames):
        """
        frames: (B, C, T, H ,W)
        targets: (B, T, n_mels)
        """
        encoder_output = self.encoder(frames) # (batch, time, encoder_hidden_size)
        decoder_output, alignments = self.decoder.inference(encoder_output) # (batch, time, n_mels)

        postnet_output = self.postnet(decoder_output) + decoder_output # (batch, time, n_mels)

        return postnet_output, alignments