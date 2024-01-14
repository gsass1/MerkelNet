import random
import torch
import torch.nn as nn
from einops import rearrange, reduce
from hparams import HParams

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, feature_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(feature_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    # https://machinelearningmastery.com/the-bahdanau-attention-mechanism/
    def forward(self, query, keys):
        # query previous hidden decoder state s_{t-1}
        # keys encoder outputs h_1, ..., h_T

        # calculate attention scores
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        # apply softmax to obtain corresponding weight values
        weights = nn.functional.softmax(scores, dim=-1)

        # The application of the softmax function essentially normalizes the
        # annotation values to a range between 0 and 1; hence, the resulting
        # weights can be considered probability values. Each probability (or weight)
        # value reflects how important h_i and s_{t-1} are in generating the next state,
        # s_t, and the next output, y_t.

        # calculate context vector as weighted sum of the annotations
        context = torch.bmm(weights, keys)

        return context, weights

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

    def __init__(self, hparams, input_size):
        super(Decoder, self).__init__()
        self.hparams = hparams
        self.attention = BahdanauAttention(hparams.decoder_hidden_size, hparams.encoder_lip_embedding_size)
        self.lstm = nn.LSTM(input_size, hparams.decoder_hidden_size, hparams.decoder_layers, batch_first=True)
        self.projection = nn.Linear(hparams.decoder_hidden_size, hparams.n_mels)

    # encoder_output: (batch_size, seq_len, encoder_lip_embedding_size)
    # targets: (batch_size, seq_len, n_mels)
    def forward(self, encoder_output, targets=None):
        batch_length = encoder_output.size(0)
        
        # initialize empty hidden and cell state
        hidden = torch.zeros(2, batch_length, self.hparams.decoder_hidden_size, dtype=torch.float32, device=encoder_output.device)
        cell = torch.zeros(2, batch_length, self.hparams.decoder_hidden_size, dtype=torch.float32, device=encoder_output.device)

        # start of sequence? this is just an "empty" mel spectogram
        input = torch.zeros(batch_length, 1, self.hparams.n_mels, dtype=torch.float32, device=encoder_output.device)

        # autoregressively decode the next mel spectrogram
        outputs = []
        for i in range(self.hparams.temporal_dim):
            output, hidden, cell, _ = self.decode_step(input, hidden, cell, encoder_output)
            outputs.append(output)

            use_teacher_forcing = random.random() < self.hparams.teacher_forcing_ratio

            if use_teacher_forcing and targets is not None:
                # if we use teacher forcing in this step, use the ground truth mel spectrogram
                input = targets[:, i, :].unsqueeze(1)
            else:
                # else use the predicted mel spectrogram from the previous step
                input = output.detach()

        outputs = torch.cat(outputs, dim=1)
        return outputs

    # input (prev mel spectogram): (batch_size, n_mels)
    # hidden (prev hidden state): (decoder_layers, batch_size, decoder_hidden_size)
    # cell (prev cell state): (decoder_layers, batch_size, decoder_hidden_size)
    # encoder_output: (batch_size, seq_len, encoder_lip_embedding_size)
    def decode_step(self, input, hidden, cell, encoder_output):
        # rearrange because batch should be first
        query = rearrange(hidden, 'l b h -> b l h') # (batch_size, num_layers, hidden_size)
        
        # query is (batch_size, num_layers, hidden_size) but we want (batch_size, seq_len, hidden_size)
        # i dunno if we should do it like this or 
        query = reduce(query, 'b l h -> b 1 h', 'mean').repeat(1, self.hparams.temporal_dim, 1)

        # attend to the current encoder output
        context, attn_weights = self.attention(query, encoder_output) # context is (batch_size, 1, hidden_size*2)

        input_lstm = torch.cat((input, context), dim=2) # (batch_size, 1, hidden_size*2+n_mels)
        lstm_output, (hidden, cell) = self.lstm(input_lstm, (hidden, cell))

        # project to mel spectrogram
        output = self.projection(lstm_output) # (batch_size, 1, n_mels)

        return output, hidden, cell, attn_weights

class MerkelNet(nn.Module):
    hparams: HParams

    def __init__(self, hparams):
        super(MerkelNet, self).__init__()
        self.hparams = hparams
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams, hparams.encoder_hidden_size*2+hparams.n_mels)

    def forward(self, x, targets=None):
        x, _ = self.encoder(x)
        x = self.decoder(x, targets)
        return x

if __name__ == "__main__":
    e = Encoder(HParams())
    x = torch.randn(1, 3, 50, 48, 48)
    print(e(x))
