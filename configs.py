import hashlib
import pickle
from typing import Literal, Optional

import pydantic
from pydantic import Field, Extra

from common import DataSourceEnum

ConfigType = Literal['base', 'smoke']


class BaseModel(pydantic.BaseModel):
    class Config:
        extra = Extra.forbid


class PretrainConfig(BaseModel):
    epochs: int = 40
    batch_size: int = 32
    lr: float = 0.0007
    train_fraction: float = Field(0.8, description="% of the data for training")
    num_workers: int = 8


class FinetuneConfig(BaseModel):
    n_frozen_layers: int = 2
    num_episodes: Optional[int] = None
    epochs: int = 50
    lr: float = 0.005
    batch_size: int = 32
    num_workers: int = 8


class OnlineConfig(BaseModel):
    num_episodes: Optional[int] = None
    batch_size: int = 16
    epochs: int = 9
    lr: float = 3.5e-3
    num_sessions: Optional[int] = None
    n_frozen_layers: int = 2
    train_intervals: int = 4
    first_training_episode: int = 0
    additional_train_episodes: int = 4
    adaptive_training: bool = True
    num_workers: int = 8


class ViTConfig(BaseModel):
    base_model_class: str = 'ViT'
    patch_size: int = 8
    dim: int = 64
    depth: int = 1
    heads: int = 2
    mlp_dim: int = 128
    dropout: float = 0.177
    emb_dropout: float = 0.277
    channels: int = 1


class BaseConfig(BaseModel):
    config_type: str = 'base'
    data_source: DataSourceEnum = DataSourceEnum.MAD
    window_size: int = 200
    overlap: int = 150
    num_classes: int = 5
    random_seed: int = 100
    save_checkpoints: bool = False

    general_model_config: ViTConfig = Field(default_factory=ViTConfig)
    pretrain: PretrainConfig = Field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)

    hostname: str = "mila"
    # hostname: str = ""
    # hostname: str = "cc-cedar"
    sweep_config: str = "sweep.yaml"
    # sweep_config: str = ""
    proc_num: int = 1
    loss: str = "MSELoss"

    class Config:
        validate_assignment = True

    def __init__(self, **data):
        if 'data_source' in data and isinstance(data['data_source'], str):
            enum_value = data['data_source'].split('.')[-1]
            data['data_source'] = DataSourceEnum[enum_value]

        super().__init__(**data)
        if self.sweep_config:
            self.proc_num = 8

    def __str__(self):
        arg_str = pickle.dumps(self.dict())
        self_hash = hashlib.sha256(arg_str).hexdigest()[:15]
        return f"{self.__class__.__name__}({self_hash})"


class SmokeConfig(BaseConfig):
    config_type: str = 'smoke'
    random_seed: int = 100
    # hostname: str = ""
    # sweep_config: str = ""
    data_source: DataSourceEnum = DataSourceEnum.MiniMAD

    general_model_config: ViTConfig = Field(default_factory=lambda: ViTConfig(patch_size=8, depth=1, heads=1, mlp_dim=4))
    pretrain: PretrainConfig = Field(default_factory=lambda: PretrainConfig(epochs=1, batch_size=1, num_workers=0))
    finetune: FinetuneConfig = Field(default_factory=lambda: FinetuneConfig(num_episodes=2, epochs=2, num_workers=0))
    online: OnlineConfig = Field(default_factory=lambda: OnlineConfig(num_episodes=2, epochs=1, num_sessions=2, train_intervals=1,
                                                                      first_training_episode=0, num_workers=0))


def fail_early():
    a = SmokeConfig()
    b = BaseConfig()
