
from pathlib import Path

import wandb
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from adaptive_hci.datasets import get_omniglot_moves, MaskedTokensDataset
from adaptive_hci.controllers import LightningAutoregressor, AutoregressiveWrapper, LanguageModel, get_device
from auto_drawer_config import AutoDrawerConfig


def get_dataloaders(omniglot_dir, canvas_size, max_sequence_length, batch_size, max_initial_value=120, pad_token=5):
    encoded_moves_data = get_omniglot_moves(omniglot_dir, canvas_size=canvas_size, max_initial_value=max_initial_value)

    # TODO do I need this?
    encoded_data_with_stop = [torch.cat((x, torch.tensor([4]))) for x in encoded_moves_data]

    train_list, val_list = train_test_split(encoded_data_with_stop)

    train_dataset = MaskedTokensDataset(train_list, max_token_len=max_sequence_length, pad_token=pad_token)
    val_dataset = MaskedTokensDataset(val_list, max_token_len=max_sequence_length, pad_token=pad_token)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader


def main():
    # TODO 
    # - sweep parameters (intergrate with buddy mila cluster)

    config = AutoDrawerConfig()
    pl_logger = WandbLogger(project='adaptive-hci', tags=['auto_drawer'], config=config)

    pl.seed_everything(config.seed)

    omniglot_dir = Path('./datasets/omniglot')
    train_dataloader, val_dataloader = get_dataloaders(omniglot_dir, config.canvas_size, config.max_sequence_length,
                                                       config.batch_size, config.num_workers)

    model = AutoregressiveWrapper(LanguageModel(
        embedding_dimension=config.embedding_dimension,
        number_of_tokens=config.number_of_tokens,
        number_of_heads=config.number_of_heads,
        number_of_layers=config.number_of_layers,
        dropout_rate=config.dropout_rate,
        max_sequence_length=config.max_sequence_length,
    )).to(get_device())
    pl_model = LightningAutoregressor(model=model, lr=config.lr)
    trainer = pl.Trainer(max_epochs=config.epochs, gradient_clip_val=config.gradient_clip_val, log_every_n_steps=1,
                         enable_checkpointing=config.save_checkpoints, logger=pl_logger)
    trainer.fit(pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


def generator():
    # TODO
    # - sample from dist instead of argmax
    # - use temperature parameter

    # move_map = [[-1, 0],
    #             [0, -1],
    #             [1, 0],
    #             [0, 1],
    #             [0, 0]]
    
    # TODO show result in end?
    # plot_encoded_moves(encoded_moves_data[6], move_map, canvas_size=30)

    pass


if __name__ == '__main__':
    main()
