import sys
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import os.path
from argparse import ArgumentParser
from tqdm import tqdm
from torchinfo import summary
import librosa
import logging

import torch

from hparams import HParams, do_arg_parse_with_hparams
from model import MerkelNet
from dataset import MerkelDataset

import wandb

def melspectrogram_to_audio(hparams: HParams, S, n_iter=64):
    S = S.detach().numpy().transpose(1, 0)

    # Convert to STFT
    linear_segment = librosa.feature.inverse.mel_to_stft(S, sr=hparams.sr, n_fft=hparams.n_fft)

    # Griffin-Lim reconstruction
    audio = librosa.griffinlim(linear_segment, n_iter=n_iter, hop_length=hparams.hop_length)

    return audio

def main():
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filemode='w',
                    stream=sys.stdout)

    parser = ArgumentParser(
                        prog='train',
                        description='Train the network')
    parser.add_argument("--summary", action='store_true')
    parser.add_argument("--preload", action='store_true')
    parser.add_argument("--enable-logging", action='store_true')
    
    args, hparams = do_arg_parse_with_hparams(parser)
    use_wandb = args.enable_logging

    if not os.path.isdir(hparams.data_dir):
        print('Data directory does not exist')
        exit(1)

    # -- Training requirements --
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # does not support conv3d :(
    #device = torch.device('mps' if torch.backends.mps.is_available() else device)
    model = MerkelNet(hparams).to(device)

    if args.summary:
        summary(model, input_size=(1, 3, hparams.temporal_dim, hparams.h, hparams.w))
        exit(0)

    # Get data
    dataset = MerkelDataset(hparams)

    if args.preload:
        dataset.preload()

    total_len = int(len(dataset) * hparams.dataset_ratio)
    train_size = int(hparams.train_test_ratio * total_len)
    test_size = total_len - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)

    # Add model architecture to Tensorboard
    #writer.add_graph(model, next(iter(train_loader))[0].to(device))

    # start a new wandb run to track this script
    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="lip2speech",
            
            # track hyperparameters and run metadata
            config=hparams._asdict()
        )

        wandb.watch(model, log="all")

    # Training loop
    logging.info(f"Training on {device}, epochs: {hparams.epochs}, batch size: {hparams.batch_size}, total train size: {train_size}, total test size: {test_size}")
    total_steps = train_size // hparams.batch_size
    for epoch in range(1, hparams.epochs):
        with tqdm(enumerate(train_loader), unit="step", total=total_steps) as tstep:
            model.train()
            tstep.set_description(f"Epoch {epoch}")

            running_loss = 0.0
            for _, (X, Y) in tstep:
                X = X.to(device)
                Y = Y.to(device)

                optimizer.zero_grad()
                output = model(X, Y)
                loss = criterion(output, Y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                tstep.set_postfix(loss=loss.item())

                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/epoch": epoch})

                # if batch_idx % hparams.batch_log == 0 and use_wandb:
                #     frames = [wandb.Image(img.cpu().detach().numpy()*255.0, caption="Input") for img in X[0][0]]
                #     ground_truth = wandb.Audio(melspectrogram_to_audio(hparams, Y[0].cpu()), caption="Ground Truth", sample_rate=hparams.sr)
                #     output_audio = wandb.Audio(melspectrogram_to_audio(hparams, output[0].cpu()), caption="Output", sample_rate=hparams.sr)
                #     wandb.log({"frames": frames, "ground_truth": ground_truth, "output": output_audio})

            #avg_train_loss = running_loss / len(train_loader)
            #print(f'Epoch {epoch}, train loss {train_loss}')
            #writer.add_scalar('loss/train', avg_train_loss, epoch)

            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for _, (X, Y) in enumerate(test_loader):
                    X = X.to(device)
                    Y = Y.to(device)

                    output = model(X)
                    loss = criterion(output, Y)
                    running_loss += loss.item()

                avg_test_loss = running_loss / len(test_loader)
                #print(f'Epoch {epoch}, test loss: {average_test_loss:.4f}')
                #writer.add_scalar('loss/test', avg_test_loss, epoch)
                if use_wandb:
                    wandb.log({"test/avg_loss": avg_test_loss, "test/epoch": epoch})

        if epoch % hparams.save_every == 0:
            filename = f"model_{epoch}.pth"
            logging.info(f'Saving model to {filename}')
            torch.save(model.state_dict(), filename)

    # def on_exit(): 
    #     wandb.finish()

    # atexit.register(on_exit)

# X = X.to(device)
# Y = Y.to(device)

# for epoch in range(1000):
#     optimizer.zero_grad()
#     output = model(X)
#     loss = criterion(output, Y)
#     loss.backward()
#     optimizer.step()

#     print(f'Epoch {epoch}, loss {loss.item()}')

if __name__ == "__main__":
    main()
