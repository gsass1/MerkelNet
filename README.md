# MerkelNet: Training a Self-Supervised Lip-to-Speech Model in German

Repository for my DL4CV project at THM.
Project only tested with Python 3.10.12

## Installation

Create new environment and install dependencies: 
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset Preprocessing

You must clone the [Merkel Podcast Corpus repository](https://github.com/deeplsd/Merkel-Podcast-Corpus) and execute the "download_video.py" script.  This may take a while.

After you're done you can execute the following to preprocess the data for training.
```bash
python ./make_dataset.py --workers N /path/to/corpus
```
where N is the amount of worker threads you wish to use. Preprocessed files will be saved by default into the `data` directory.

## Training the model
Following example command trains with Wandb logging enabled and also preloads the entire dataset into memory.

```bash
python ./train.py --enable-logging --preload
```

## Evaluation
This command runs an evaluation on a random subset of the dataset (controlled by `--size`) and calculates mean STOI and ESTOI metrics.
```bash
python ./eval.py --checkpoint /path/to/checkpoint.pth --size 0.1
```

## Demo

Starts the Gradio demo.
```bash
python ./demo.py --checkpoint /path/to/checkpoint.pth
```
