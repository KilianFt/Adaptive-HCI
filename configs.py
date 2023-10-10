import dataclasses
import hashlib
import pickle

from common import DataSourceEnum


@dataclasses.dataclass
class BaseConfig:
    data_source: DataSourceEnum = DataSourceEnum.MAD
    pretrained: bool = False
    early_stopping: bool = False
    epochs: int = 40
    batch_size: int = 32
    lr: float = 0.0007
    window_size: int = 200
    overlap: int = 150
    model_class: str = 'ViT'
    patch_size: int = 8
    dim: int = 64
    depth: int = 1
    heads: int = 2
    mlp_dim: int = 128
    dropout: float = 0.177
    emb_dropout = 0.277
    channels: int = 1
    n_frozen_layers: int = 2
    random_seed: int = 100
    save_checkpoints: bool = False
    hostname: str = ""
    # hostname: str = "mila"
    # hostname: str = "cc-cedar"
    # sweep_config: str = "sweep.yaml"
    sweep_config: str = ""
    proc_num: int = 1
    loss: str = "MSELoss"
    train_fraction: float = 0.8  # 80% of the data for training

    def __post_init__(self):
        # TODO: assert all parameters have a typehint otherwise to_dict doesn't parse them?!?!?
        if self.sweep_config:
            self.proc_num = 4
        if isinstance(self.data_source, str):  # TODO: tihs is bad, either pydantic or python 3.11 could fix it maybe
            for s in DataSourceEnum:
                if str(s) == self.data_source:
                    self.data_source = s
                    break
            else:
                raise ValueError

    def __str__(self):
        arg_str = pickle.dumps(self)
        self_hash = hashlib.sha256(arg_str).hexdigest()[:15]
        return f"{self.__class__.__name__}({self_hash})"


@dataclasses.dataclass
class SmokeConfig(BaseConfig):
    epochs: int = 1
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
