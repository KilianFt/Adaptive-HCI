from pathlib import Path

import torch
from mingpt.model import GPT
from mingpt.trainer import Trainer

from adaptive_hci.datasets import OmniglotGridDataset

def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

def main():
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-mini'
    model_config.vocab_size = 6 # openai's model vocabulary
    model_config.block_size = 100  # openai's model block_size (i.e. input context length)
    model = GPT(model_config)

    omniglot_dir = Path("./datasets/omniglot")
    train_dataset = OmniglotGridDataset(omniglot_dir, context_len=100, char_idxs=None)

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4 # many possible options, see the file
    train_config.max_iters = 20000
    train_config.batch_size = 32
    trainer = Trainer(train_config, model, train_dataset)

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    torch.save(model.state_dict(), 'models/draw_gpt_state_dict.pt')


if __name__ == '__main__':
    main()