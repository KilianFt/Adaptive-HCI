from pathlib import Path
import torch

from autowriter.mingpt.model import GPT
from autowriter.mingpt.trainer import Trainer
from autowriter.datasets import OmniglotGridDataset

from configs import AutoWriterConfig


def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")


def main(config):
    # TODO shuffle starting positions
    train_dataset = OmniglotGridDataset(config.omniglot_dir,
                                        context_len=config.context_len,
                                        char_idxs=config.character_idxs)

    model_config = GPT.get_default_config()
    model_config.model_type = config.gpt_type
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model = GPT(model_config)

    model_file = Path('models/draw_gpt_state_dict_o_l.pt')
    if model_file.exists():
        print('Loading existing writing model')
        return model.load_state_dict(model_file)

    train_config = Trainer.get_default_config()
    train_config.learning_rate = config.lr
    train_config.max_iters = config.max_iters
    train_config.batch_size = config.batch_size
    trainer = Trainer(train_config, model, train_dataset)

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    torch.save(model.state_dict(), model_file)


if __name__ == '__main__':
    config = AutoWriterConfig()
    main(config)
