import argparse
import logging
import os
import os.path
import sys

import librosa
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import MerkelDataset, denormalize
from hparams import do_arg_parse_with_hparams
from model import MerkelNet

from pystoi import stoi

def melspectrogram_to_audio(hparams, S, n_iter=64):
    # Convert to STFT
    linear_segment = librosa.feature.inverse.mel_to_stft(S, sr=hparams.sr, n_fft=hparams.n_fft)

    # Griffin-Lim reconstruction
    audio = librosa.griffinlim(linear_segment, n_iter=n_iter, hop_length=hparams.hop_length)

    return audio

def calculate_eval_metrics(hparams, output, target):
    wav_output = melspectrogram_to_audio(hparams, denormalize(output.cpu().numpy().transpose(1, 0), hparams))
    target_output = melspectrogram_to_audio(hparams, denormalize(target.cpu().numpy().transpose(1, 0), hparams))

    eval_stoi = stoi(target_output, wav_output, hparams.sr, extended=False)
    eval_estoi = stoi(target_output, wav_output, hparams.sr, extended=True)

    return eval_stoi, eval_estoi

def main():
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filemode='w',
                    stream=sys.stdout)

    parser = argparse.ArgumentParser(
                    prog='eval',
                    description='Calculate evaluation metrics on a random validation dataset')
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument('--size', type=float, default=0.1, help='Size of the validation dataset as a fraction of the total dataset')
    args, hparams = do_arg_parse_with_hparams(parser)

    if not os.path.exists(args.checkpoint):
        logging.error('Missing checkpoint')
        exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MerkelNet(hparams).to(device)

    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    dataset = MerkelDataset(hparams)
    total_len = len(dataset)
    eval_size = int(args.size * total_len)
    rest_size = total_len - eval_size

    eval_dataset, _ = random_split(dataset, [eval_size, rest_size])
    eval_loader = DataLoader(eval_dataset, batch_size=hparams.batch_size, shuffle=False)

    logging.info(f"Evaluating on {device}, batch size: {hparams.batch_size}, total eval size: {eval_size}")
    total_steps = eval_size // hparams.batch_size

    eval_stoi, eval_estoi = [], []

    with torch.no_grad():
        with tqdm(enumerate(eval_loader), unit="step", total=total_steps) as tstep:
            for _, (X, Y) in tstep:
                X = X.to(device)
                Y = Y.to(device)
                Y_pred, _ = model.inference(X)

                for idx in range(Y_pred.shape[0]):
                    e_stoi, e_estoi = calculate_eval_metrics(hparams, Y_pred[idx], Y[idx])
                    eval_stoi.append(e_stoi)
                    eval_estoi.append(e_estoi)

    print(f"STOI: {sum(eval_stoi)/len(eval_stoi)}")
    print(f"ESTOI: {sum(eval_estoi)/len(eval_estoi)}")

if __name__ == "__main__":
    main()