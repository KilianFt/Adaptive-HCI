import enum


class DataSourceEnum(enum.Enum):
    MAD = "mad"
    MiniMAD = "mini_mad"
    MERGED = "merged"
    NINA_PRO = "nina_pro"

class GeneralModelEnum(enum.Enum):
    TCN = "tcn"
    ViT = "vit"
    MLP = "mlp"
