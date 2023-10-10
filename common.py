import enum


class DataSourceEnum(enum.Enum):  # (enum.StrEnum):
    MAD = "mad"
    MiniMAD = "mini_mad"
    MERGED = "merged"
    NINA_PRO = "nina_pro"
