import hashlib

import configs
import finetune_user_model
import train_general_model
from deployment.buddy import buddy_setup

train_users = [
    hashlib.sha256("Kilian".encode("utf-8")).hexdigest(),
    hashlib.sha256("Manuel".encode("utf-8")).hexdigest(),
]


def main():
    experiment_config = configs.SmokeConfig()
    logger, experiment_config = buddy_setup(experiment_config)

    general_model = train_general_model.main(logger, experiment_config)
    for user_hash in train_users:
        finetune_user_model.main(general_model, user_hash, experiment_config)


if __name__ == '__main__':
    main()
