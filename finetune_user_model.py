import pathlib

import lightning.pytorch as pl
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import configs
from adaptive_hci.datasets import load_emg_writing_data, to_tensor_class_dataset, maybe_download_drive_folder
from adaptive_hci import utils


file_ids = [
'1Mb_HNtnT3CrAjUvNNh3Z98ghfFmkSqqp',
'174sJRa1RCkYAlT-81tGM-YuRy0hhQFq_',
'1dgUmhM-oi0jiPR103h57dbqa_rH1vO2t',
'17CG_Xm0J8YkURTlOFfa21CSX0hoKJjoN',
'15I_UkMkKIrwgPxb9jTFOkBSHAR5p3E3e',
'19ZSRRadPQmrbQ1zQC1U94XNtn7vo-dxU',
'1tvFtO5DfuW5BUkBUsNtBnLfQ9fmcfkGM',
'14P3Qi5iD0kR-5g6PpCHCZ82S5p3R8D9i',
'1U-wNxgGhfrQgRQ-zBZm0vlFbKsVaeu3W',
'1suL8S9GfFaO1XVoXUr_mdWvVgKnw74P2',
'1aa9XUkRbb-4Z577xIvJiWNtgO7Cz_wMZ',
'1zXij6AGD4T7ADzrVUulPVrOa5eLM3_zz',
'1sPwPLvJ-li2AvXRxa6OjFpUyKqgKI-sw',
'1RUeD92mxURz8TVJMUWPLBFzU_oXpJ8cI',
'11haIkD6giBwkur3QZGY5iSwUMg_vswmt',
'1vUxvwNwJT1qPv58Nvs09x2u2qEHUCPT7',
'1MAHyPiRL9NaEqfDmQOUADBJ5Pfln73ki',
'1UPloRikI0ETzz3chSp3x-eYG2WVve4Xz',
'19wBkr3VAfBkZxNLwp7xaz7H3iPUqwdQr',
'1YxteURGdP17wXDiYpxGJ9hs89UBjWsPu',
'1O-isEThwHQ_GVP_eEWCscCmrtePZeZYB',
'1tYlAI5Oya5LJ2MUil3eKDqEzlsXooDER',
'1cNqtqE8Ps0o6MxKphz1QNqLk-8zeOVoY',
'11DCtNsu4peW9qtpcH3Z2Qfh0UkFTZGBC',
'15KJXEzdpIRhkIHpbmWuxPV5GrBk4u6nf',
'15tJDp5fZOwkgKPd19OF_uAn9s5M_OE9t',
'1qTeCcOsqMkLZjpLkjnl0G2e_IAxiMsgJ',
'1p_GLVOR6PHKDei0_bblO4-VCB-UOBI_I',
'1QbQtSGg7zF-V2lhXMZ8U2NFpsM9Ezy1T',
'1pJj0KFtxfOSQY3HIESCHR35FpTVwbHvr',
'1-2X06IK7F9bGtwIzFA5NTq3uYn8rh6Qx',
'1I5AHKyoo9P7_KryV2f5I4Om0SOfBInqz',
'1g_yzZEHbowTOEKGX3t69KWRataiyx2NI',
'1hvkzyHS_uG2E2JV8o3k8_LZd-0acuK4w',
'1P2IeWkisgAqXyNrpzcuS3RPBHX9xE0Rm',
'18U9rgZ8RMrYhOEiCdUdPDqyWAvnDCt7a',
'1ZvawfZHBFq_jqTsACFEx-eyEm07j-ghC',
'16OnpXhN9q2AxDOcZGkM_AKpwzo-57Kce',
'10zZ8wLebKNmtE-gWQX5TL_Zi_aEE7uaG',
'1w5YDB7tJyRKeL30IaGzKAWJXW0bLfjZP',
'1hA0OL4djnuw42rX1msjE-PnSzXY4Qwdp',
'1VNGb0aKM1bG90fE4cf-ohBm_nWRq66-Y',
'1M2PgwFkZoZCEjoo34nhwintTO4JF9Nok',
'1F_VYUAAN7oJvtJGyBLu0FsA_w1iyX4XV',
'1J6gENg-f67GiS_qBT0156IkOUdB1GxXL',
'1NAXTQ7YFLVG47C_0NmWJYYhr4hD4-Zj3',
'1uYMzIniaRaaAQQaRxGIYwOZRQzaikzTw',
'1NslN0ytvS2i7mdKjAQqMELJpDSvsiqE5',
'10SaXk4NELp7itNTLsDRYBh7hYhV4XNNo',
'1SijE_FisantprtdOaUOmWWuOnfU-Irlw',
'1cbjYSiibIdqzQZe5HrJVJkdP4z3IuWVm',
'1Kzoa_l7vH_s4K2cnUbAF03GHaeLtksJU',
'1ELyiNHvKhpGF3HNb1soFfpbLmdhScRKV',
'1Phi36oM9S4pcEENLMrsjot8axrcV6skc',
'19MdSU1a1MV_Q5f-kxhxSIGSGML8dX96x',
'1vgMD95xxqLaZGK4BMKgWw8PBgNQZlSRN',
'1dpnaEpsp3Tadv6ZDckCshabmkzONRhqN',
'1nDLe9CaIMDR2wmn5m798DfjEobfk5GLo',
'1Q04S-EUJ5eKdNp_5AXH0IqrKDtZehB9s',
'17oIN3YeXs_dFZKqZUDHccthtxd_9R-Ef',
'1AcMA-6Vl67FjZ6ICCB2WWwklA08x6slo',
'1FITVLXtiUYKYRt0HaiyCGPS-5K_CUYBO',
'1lZlIcyh_hcnjVAxHpmgqovljqQfGkeoz',
'1V2oAp6XFaHJdHwUfRNXM75obQ7v6lhWJ',
'1XJzexKsDeoNUBwQvjqSOWGxSItAp8Ksy',
'1S3N_U4tjt1CkzkMtLvQQD1PmgooljLOT',
'1cpPGUJjKJZDGZR42Tg2Da2elcpGQIhZw',
'1jM0d8zVtPbaHspKUhisSQAOipybyRvIB',
'1w8EJsizK2Z3GvfHG48sJkicWr-70PkBq',
'1r3L8zpPBbMl_K_W4RM6rcRmwQyRhdifU',
]

def load_finetune_dataloader(config):
    file_path = pathlib.Path(__file__).resolve()
    print(file_path)
    emg_draw_data_dir = file_path.parent / 'datasets' / 'emg_writing_o_l/'
    # emg_writing_ids_file = file_path.parent / 'emg_writing_file_names.txt'
    # with open(emg_writing_ids_file, 'rb') as f:
    #     file_ids = f.readlines()
    # file_ids = [file_id.decode().strip() for file_id in file_ids]
    print(file_ids)
    maybe_download_drive_folder(emg_draw_data_dir, file_ids)

    observations, actions = load_emg_writing_data(emg_draw_data_dir, window_size=config.window_size, overlap=config.overlap)

    train_observations, val_observations, train_optimal_actions, val_optimal_actions = train_test_split(observations, actions, test_size=0.25)

    train_offline_adaption_dataset = to_tensor_class_dataset(train_observations, train_optimal_actions)
    val_offline_adaption_dataset = to_tensor_class_dataset(val_observations, val_optimal_actions)

    dataloader_args = dict(batch_size=config.finetune.batch_size, num_workers=config.finetune.num_workers)
    train_dataloader = DataLoader(train_offline_adaption_dataset, shuffle=True, **dataloader_args)
    val_dataloader = DataLoader(val_offline_adaption_dataset, **dataloader_args)
    return train_dataloader, val_dataloader


def main(model: LightningModule, user_hash, config: configs.BaseConfig) -> LightningModule:
    if not config.finetune.do_finetuning:
        print('Skip finetuning')
        return model

    logger = WandbLogger(project='adaptive_hci', tags=["finetune", user_hash], config=config,
                         name=f"finetune_{config}_{user_hash[:15]}")

    train_dataloader, val_dataloader = load_finetune_dataloader(config)

    model.lr = config.finetune.lr
    model.freeze_layers(config.finetune.n_frozen_layers)
    model.metric_prefix = f'{user_hash}/finetune/'
    model.step_count = 0

    accelerator = utils.get_accelerator(config.config_type)
    trainer = pl.Trainer(max_epochs=config.finetune.epochs, log_every_n_steps=1, logger=logger,
                         enable_checkpointing=config.save_checkpoints, accelerator=accelerator,
                         gradient_clip_val=config.gradient_clip_val)

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    return model
