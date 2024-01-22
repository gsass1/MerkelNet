from argparse import ArgumentParser
from typing import NamedTuple

class HParams(NamedTuple):
    temporal_dim: int = 50
    sr: int = 16000
    n_mels: int = 128
    n_fft: int = 1280
    hop_length: int = 640
    fps: int = 25
    f_max: int = 8000
    w: int = 48
    h: int = 48
    dataset_batch_size: int = 32
    batch_size: int = 32
    data_dir: str = "data"
    epochs: int = 1000
    learning_rate: float = 1e-3
    train_test_ratio: float = 0.9
    batch_log: int = 50
    dataset_ratio: float = 1.0
    save_every: int = 100
    teacher_forcing_ratio: float = 0.5

    encoder_layers: int = 1
    encoder_hidden_size: int = 32
    encoder_lip_embedding_size: int = 128  # = encoder_hidden_size*2

    decoder_transformer_size: int = 128
    decoder_transformer_heads: int = 8
    decoder_transformer_layers: int = 4

    min_level_db: float = -100.
    ref_level_db: float = 20.
    max_abs_value: float = 4.

    dropout: float = 0.1

def do_arg_parse_with_hparams(parser: ArgumentParser):
    default_hparams = HParams()

    fields = default_hparams._asdict().keys()
    for field in fields:
        default_value = getattr(default_hparams, field)
        parser.add_argument(f"--{field}", default=default_value, required=False, type=type(default_value))

    args = parser.parse_args()

    hparams = HParams(*[getattr(args, field) for field in fields])
    return args, hparams
