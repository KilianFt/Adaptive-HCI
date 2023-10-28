import hashlib
import pickle
from typing import Literal, Optional

from pydantic import BaseModel, Field

from common import DataSourceEnum

ConfigType = Literal['base', 'smoke']


class BaseConfig(BaseModel):
    config_type: ConfigType = 'base'

    data_source: DataSourceEnum = DataSourceEnum.MAD
    pretrained: bool = False
    early_stopping: bool = False
    pretraining_epochs: int = 40
    batch_size: int = 32
    lr: float = 0.0007
    window_size: int = 200
    overlap: int = 150
    base_model_class: str = 'ViT'
    patch_size: int = 8
    dim: int = 64
    depth: int = 1
    heads: int = 2
    mlp_dim: int = 128
    dropout: float = 0.177
    emb_dropout: float = 0.277
    channels: int = 1
    random_seed: int = 100
    save_checkpoints: bool = False
    finetune_n_frozen_layers: int = 2
    finetune_num_episodes: int = None
    finetune_epochs: int = 50
    finetune_lr: float = 0.005
    finetune_batch_size: int = 32
    online_num_episodes: int = None
    online_batch_size: int = 16
    online_epochs: int = 9
    online_lr: float = 3.5e-3
    online_num_sessions: Optional[int] = None
    online_n_frozen_layers: int = 2
    online_train_intervals: int = 4
    online_first_training_episode: int = 0
    online_additional_train_episodes: int = 4
    hostname: str = ""
    # hostname: str = "mila"
    # hostname: str = "cc-cedar"
    # sweep_config: str = "sweep.yaml"
    sweep_config: str = ""
    proc_num: int = 1
    loss: str = "MSELoss"
    train_fraction: float = Field(0.8, description="80% of the data for training")
    limit_train_batches: int = 200
    finetune_num_workers: int = 8
    online_adaptation_num_workers: int = 8

    class Config:
        validate_assignment = True

    def __init__(self, **data):
        if 'data_source' in data and isinstance(data['data_source'], str):
            enum_value = data['data_source'].split('.')[-1]
            data['data_source'] = DataSourceEnum[enum_value]

        super().__init__(**data)
        if self.sweep_config:
            self.proc_num = 4

    def __str__(self):
        arg_str = pickle.dumps(self.model_dump())
        self_hash = hashlib.sha256(arg_str).hexdigest()[:15]
        return f"{self.__class__.__name__}({self_hash})"


class SmokeConfig(BaseConfig):
    config_type: ConfigType = 'smoke'
    pretraining_epochs: int = 1
    batch_size: int = 1
    patch_size: int = 8
    depth: int = 1
    heads: int = 1
    mlp_dim: int = 4
    channels: int = 1
    random_seed: int = 100
    hostname: str = ""
    sweep_config: str = ""
    data_source: DataSourceEnum = DataSourceEnum.MiniMAD

    finetune_num_episodes: int = 2
    finetune_epochs: int = 2
    finetune_num_workers: int = 0

    online_num_episodes: int = 2
    online_epochs: int = 1
    online_num_sessions: Optional[int] = 2
    online_train_intervals: int = 1
    online_first_training_episode: int = 0
    online_adaptation_num_workers: int = 0

    limit_train_batches: int = 1
