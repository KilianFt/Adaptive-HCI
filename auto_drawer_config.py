import hashlib
import pickle

from configs import BaseModel


class AutoDrawerConfig(BaseModel):
    config_type: str = 'base'
    seed: int = 1000
    save_checkpoints: bool = False
    canvas_size: int = 30 # size to reduce stroke to (project to canvas_size x canvas_size grid)
    batch_size: int = 32
    num_workers: int = 8
    epochs: int = 20
    lr: float = 1e-3
    embedding_dimension: int = 256
    max_sequence_length: int = 200
    number_of_tokens: int = 6
    number_of_heads: int = 4
    number_of_layers: int = 3
    dropout_rate: float = 0.1
    gradient_clip_val: float = 0.5
    criterion_key: str = 'crossentropy'
    pad_token: int = 5
    eos_token: int = 4

    hostname: str = "mila"
    # hostname: str = ""
    sweep_config: str = "sweep.yaml"
    # sweep_config: str = ""
    proc_num: int = 1

    class Config:
        validate_assignment = True

    def __init__(self, **data):
        super().__init__(**data)
        if self.sweep_config:
            self.proc_num = 8

    def __str__(self):
        arg_str = pickle.dumps(self.dict())
        self_hash = hashlib.sha256(arg_str).hexdigest()[:15]
        return f"{self.__class__.__name__}({self_hash})"


class AutoDrawerSmokeConfig(AutoDrawerConfig):
    config_type: str = 'smoke'
    # hostname: str = ""
    # sweep_config: str = ""
    num_workers: int = 1
    epochs: int = 1


def fail_early():
    a = AutoDrawerSmokeConfig()
    b = AutoDrawerConfig()
