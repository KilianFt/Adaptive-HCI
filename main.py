import copy
import hashlib
import os
import sys

import numpy as np

import configs
import train_general_model
import finetune_user_model
import continuously_train_user_model
from deployment.buddy import buddy_setup

train_users = [
    hashlib.sha256("Kilian".encode("utf-8")).hexdigest(),
    hashlib.sha256("Manuel".encode("utf-8")).hexdigest(),
]


def main():
    if sys.gettrace():
        experiment_config = configs.SmokeConfig()
    else:
        experiment_config = configs.BaseConfig()

    entity = "delvermm" if "delverm" in os.getlogin() else "kilian"
    logger, experiment_config = buddy_setup(experiment_config, entity=entity)

    general_model, pretrain_score = train_general_model.main(logger, experiment_config)
    for user_hash in train_users:
        initial_model = copy.deepcopy(general_model)
        finetuned_user_model, finetune_score = finetune_user_model.main(initial_model, user_hash, experiment_config)
        user_model, continuous_score = continuously_train_user_model.main(finetuned_user_model, user_hash, experiment_config)

        overall_score = np.mean((pretrain_score, finetune_score, continuous_score))
        logger.log({'overall_score': overall_score})

        print(f"\nOverall score is {overall_score}\n")

def fail_early():
    configs.fail_early()


if __name__ == '__main__':
    fail_early()
    main()
