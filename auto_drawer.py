from pathlib import Path

import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from adaptive_hci.datasets import get_omniglot_moves, MaskedTokensDataset
from adaptive_hci.controllers import (
    LightningAutoregressor,
    AutoregressiveWrapper,
    LanguageModel,
    get_device,
)
from auto_drawer_config import AutoDrawerConfig, AutoDrawerSmokeConfig


def get_dataloaders(
    omniglot_dir,
    canvas_size,
    max_sequence_length,
    batch_size,
    max_initial_value=120,
    eos_token=4,
    pad_token=5,
):
    encoded_moves_data = get_omniglot_moves(
        omniglot_dir, canvas_size=canvas_size, max_initial_value=max_initial_value
    )

    # TODO do I need this?
    encoded_data_with_stop = [
        torch.cat((x, torch.tensor([eos_token]))) for x in encoded_moves_data
    ]

    train_list, val_list = train_test_split(encoded_data_with_stop)

    train_dataset = MaskedTokensDataset(
        train_list, max_token_len=max_sequence_length, pad_token=pad_token
    )
    val_dataset = MaskedTokensDataset(
        val_list, max_token_len=max_sequence_length, pad_token=pad_token
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader


def train(config):
    # TODO
    # - sweep parameters (intergrate with buddy mila cluster)

    pl_logger = WandbLogger(config=config)

    pl.seed_everything(config.seed)

    omniglot_dir = Path("./datasets/omniglot")
    train_dataloader, val_dataloader = get_dataloaders(
        omniglot_dir,
        config.canvas_size,
        config.max_sequence_length,
        config.batch_size,
        config.num_workers,
        eos_token=config.eos_token,
        pad_token=config.pad_token,
    )

    model = AutoregressiveWrapper(
        LanguageModel(
            embedding_dimension=config.embedding_dimension,
            number_of_tokens=config.number_of_tokens,
            number_of_heads=config.number_of_heads,
            number_of_layers=config.number_of_layers,
            dropout_rate=config.dropout_rate,
            max_sequence_length=config.max_sequence_length,
        )
    ).to(get_device())
    pl_model = LightningAutoregressor(model=model, lr=config.lr)
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=1,
        enable_checkpointing=config.save_checkpoints,
        logger=pl_logger,
    )
    trainer.fit(
        pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    return pl_model


def pad_left(sequence, final_length, padding_token):
    return [padding_token] * (final_length - len(sequence)) + sequence


def plot_moves(moves, canvas_size=120, save_img=True):
    canvas = np.zeros((canvas_size, canvas_size))
    last_point = np.array([canvas_size // 2, canvas_size // 2])
    canvas[last_point[0], last_point[1]] = 1

    for move in moves:
        point = last_point + move
        canvas[point[0], point[1]] = 1
        last_point = point
    plt.imshow(canvas)
    if save_img:
        plt.savefig("figures/draw_test")


def plot_encoded_moves(encoded_moves, move_map, canvas_size=120, save_img=True):
    moves = [move_map[x] for x in encoded_moves]
    plot_moves(moves, canvas_size=canvas_size, save_img=save_img)

# TODO add accuracy metric
def generate(
    model, config, start_tokens=None, temperature=1.0, max_tokens_to_generate=100
):
    model = model.to(get_device())
    padding_token = config.pad_token
    eos_token = config.eos_token

    model.eval()

    if start_tokens is None:
        start_tokens = [0]

    input_tensor = torch.tensor(
        pad_left(
            sequence=start_tokens,
            final_length=model.max_sequence_length + 1,
            padding_token=padding_token,
        ),
        dtype=torch.long,
    ).to(get_device())

    num_dims = len(input_tensor.shape)

    if num_dims == 1:
        input_tensor = input_tensor[None, :]

    out = input_tensor
    for _ in range(max_tokens_to_generate):
        x = out[:, -model.max_sequence_length :]

        mask = torch.ones_like(x)
        mask[x == padding_token] = 0

        next_token_probabilities = model.next_token_probabilities(
            x=x, temperature=temperature, mask=mask
        )

        next_token = torch.multinomial(next_token_probabilities, num_samples=1)
        out = torch.cat([out, next_token], dim=1)

        if eos_token is not None and next_token == eos_token:
            break

    generated_tokens = out[0]
    generated_tokens_wo_pad = generated_tokens[generated_tokens != padding_token]
    generated_tokens_wo_end = generated_tokens_wo_pad[
        generated_tokens_wo_pad != eos_token
    ]

    generated_tokens_wo_end = generated_tokens_wo_end.tolist()

    return generated_tokens_wo_end


def main():
    _ = wandb.run(project="adaptive-hci", tags=["auto_drawer"])
    config = AutoDrawerConfig(**wandb.config)
    # config = AutoDrawerSmokeConfig()
    pl_model = train(config)

    torch.save(pl_model.model, "models/drawer_test.pt")

    generated_tokens = generate(pl_model.model, config)
    move_map = [[-1, 0], [0, -1], [1, 0], [0, 1], [0, 0]]
    plot_encoded_moves(generated_tokens, move_map, canvas_size=100)


if __name__ == "__main__":
    main()
