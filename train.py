import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import os.path
from argparse import ArgumentParser
import atexit
from tqdm import tqdm
from torchinfo import summary

import torch
from torch.utils.tensorboard import SummaryWriter

from hparams import HParams
from model import MerkelNet
from dataset import MerkelDataset

def main():
    # Default hparams
    default_hparams = HParams()

    # Set up an arg parser that lets the user modify hparams
    parser = ArgumentParser(
                        prog='train',
                        description='Train the network')
    fields = default_hparams._asdict().keys()
    for field in fields:
        parser.add_argument(f"--{field}", default=getattr(default_hparams, field), required=False)
    parser.add_argument("--summary", action='store_true')
    args = parser.parse_args()

    hparams = HParams(*[getattr(args, field) for field in fields])

    if not os.path.isdir(hparams.data_dir):
        print('Data directory does not exist')
        exit(1)

    # -- Training requirements --
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MerkelNet(hparams).to(device)

    if args.summary:
        summary(model, input_size=(1, 3, hparams.temporal_dim, hparams.h, hparams.w))
        exit(0)

    # Get data
    dataset = MerkelDataset(hparams)

    train_size = int(hparams.train_test_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)

    # Add model architecture to Tensorboard
    writer.add_graph(model, next(iter(train_loader))[0].to(device))

    # Training loop
    for epoch in range(1, hparams.epochs):
        with tqdm(enumerate(train_loader), unit="batch", total=len(train_loader)) as tepoch:
            model.train()
            tepoch.set_description(f"Epoch {epoch}")

            running_loss = 0.0
            for _, (X, Y) in tepoch:
                X = X.to(device)
                Y = Y.to(device)

                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, Y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())

            avg_train_loss = running_loss / len(train_loader)
            #print(f'Epoch {epoch}, train loss {train_loss}')
            writer.add_scalar('loss/train', avg_train_loss, epoch)

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
                writer.add_scalar('loss/test', avg_test_loss, epoch)

            torch.save(model.state_dict(), 'model.pth')

    def on_exit(): 
        writer.close()

    atexit.register(on_exit)

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
