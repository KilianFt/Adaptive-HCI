from pathlib import Path
import torch
import wandb

from adaptive_hci import utils
from autowriter.mingpt.model import GPT
from autowriter.mingpt.trainer import Trainer
from autowriter.datasets import OmniglotGridDataset
from deployment.buddy import buddy_setup

from configs import BaseConfig


def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        train_loss = trainer.loss.item()
        val_loss = trainer.validate().item()
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {train_loss:.5f} val loss {val_loss:.5f}")
        wandb.run.log(data={"train_loss": train_loss, "val_loss": val_loss,}, step=trainer.iter_num)


def eval_callback(trainer):
        val_loss = trainer.val_loss.item()
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: val loss {val_loss:.5f}")
        wandb.run.log(data={"val_loss": val_loss,}, step=trainer.iter_num)


def main(config):
    train_dataset = OmniglotGridDataset(config.omniglot_dir,
                                        context_len=config.context_len,
                                        char_idxs=config.character_idxs,
                                        canvas_sizes=config.canvas_sizes,)

    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.n_layer = config.n_layer
    model_config.n_head = config.n_head
    model_config.n_embd =  config.n_embd
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model = GPT(model_config)

    model_file = Path('models/draw_gpt_state_dict_l_o_v7.pt')
    # if model_file.exists():
    #     print('Loading existing writing model')
    #     state_dict = torch.load(model_file)
    #     return model.load_state_dict(state_dict)

    train_config = Trainer.get_default_config()
    train_config.learning_rate = config.lr
    train_config.device = utils.get_accelerator()
    train_config.max_iters = config.max_iters
    train_config.batch_size = config.batch_size

    trainer = Trainer(train_config, model, train_dataset)

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    torch.save(model.state_dict(), model_file)


if __name__ == '__main__':
    experiment_config = BaseConfig()
    torch.manual_seed(experiment_config.seed)

    entity = "kilian"

    logger, experiment_config = buddy_setup(experiment_config, entity=entity)
    config = experiment_config.auto_writer

    main(config)