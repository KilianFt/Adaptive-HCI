import enum
from adaptive_hci.controllers import EMGLSTM, EMGViT

class DataSourceEnum(enum.Enum):
    MAD = "mad"
    MiniMAD = "mini_mad"
    MERGED = "merged"
    NINA_PRO = "nina_pro"

class BaseModelEnum(enum.Enum):
    RNN = "rnn"
    ViT = "vit"


BASE_MODELS = {
    BaseModelEnum.ViT: EMGViT,
    BaseModelEnum.RNN: EMGLSTM,
}
