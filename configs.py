import dataclasses


@dataclasses.dataclass
class BaseConfig:
    pretrained: bool = False
    early_stopping: bool = False
    epochs: int = 40
    batch_size: int = 32
    lr: float = 0.0007
    window_size: int = 200
    overlap: int = 150
    model_class = 'ViT'
    patch_size: int = 8
    dim: int = 64
    depth: int = 1
    heads: int = 2
    mlp_dim: int = 128
    dropout: float = 0.177
    emb_dropout = 0.277
    channels: int = 1
    random_seed: int = 100
    save_checkpoints: bool = False
    hostname = "mila"
    # hostname = "cc-cedar"
    sweep_config = "sweep.yaml"
    # sweep_config = ""
    proc_num = 1
    loss = "MSELoss"


    def __post_init__(self):
        # TODO: assert all parameters have a typehint otherwise to_dict doesn't parse them?!?!?
        if self.sweep_config:
            self.proc_num = 4
