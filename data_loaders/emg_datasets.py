from torch.utils.data import random_split, DataLoader

import configs
from adaptive_hci import utils
from adaptive_hci.datasets import CombinedDataset, EMGWindowsDataset
from common import DataSourceEnum


def get_dataloaders(config: configs.BaseConfig):
    dataset = get_dataset(config)
    train_fraction = int(config.pretrain.train_fraction * len(dataset))
    test_fraction = int(len(dataset) - train_fraction)
    train_dataset, test_dataset = random_split(dataset, [train_fraction, test_fraction])
    dataloader_args = dict(batch_size=config.pretrain.batch_size, drop_last=False, num_workers=config.pretrain.num_workers)

    train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_args)
    test_dataloader = DataLoader(test_dataset, **dataloader_args)
    return train_dataloader, test_dataloader, dataset.num_unique_labels


@utils.disk_cache
def get_dataset(config: configs.BaseConfig) -> CombinedDataset:
    data_source = config.data_source
    assert data_source not in (DataSourceEnum.NINA_PRO,), "not implemented, use merged to include it"

    dataset = EMGWindowsDataset(
        data_source,
        split="Train",
        window_size=config.window_size,
        overlap=config.overlap
    )
    if data_source == DataSourceEnum.MERGED:
        ninapro5_train_dataset = EMGWindowsDataset(
            data_source,
            split="train",
            window_size=config.window_size,
            overlap=config.overlap
        )
        ninapro5_test_dataset = EMGWindowsDataset(
            data_source,
            split="test",
            window_size=config.window_size,
            overlap=config.overlap
        )
        ninapro = CombinedDataset(ninapro5_test_dataset, ninapro5_train_dataset)
        dataset = CombinedDataset(dataset, ninapro)
    elif data_source in (DataSourceEnum.MiniMAD, DataSourceEnum.MAD):
        pass
    else:
        raise NotImplementedError(f"Unknown datasource {data_source}")
    return dataset
