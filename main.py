import copy
import hashlib
import os
import sys
import random

import torch
import numpy as np

import configs
import train_general_model
import finetune_user_model
import continuously_train_user_model
from deployment.buddy import buddy_setup

train_users = [
    hashlib.sha256("Kilian".encode("utf-8")).hexdigest()[:15],
    # hashlib.sha256("Manuel".encode("utf-8")).hexdigest()[:15],
]


def main():
    if sys.gettrace():
        experiment_config = configs.SmokeConfig()
    else:
        experiment_config = configs.BaseConfig()

    try:
        entity = "delvermm" if "delverm" in os.getlogin() else "kilian"
    except OSError:  # Happens on mila cluster
        entity = "delvermm"

    torch.manual_seed(experiment_config.seed)
    random.seed(experiment_config.seed)

    logger, experiment_config = buddy_setup(experiment_config, entity=entity)

    general_model = train_general_model.main(logger, experiment_config)

    population_metrics = []
    for user_hash in train_users:
        initial_model = copy.deepcopy(general_model)
        finetuned_user_model = finetune_user_model.main(initial_model, user_hash, experiment_config)
        user_model, metrics = continuously_train_user_model.main(finetuned_user_model, user_hash, experiment_config)
        population_metrics.append(metrics)

    # TODO make this a bit prettier
    population_accuracies = []
    population_f1s = []
    population_one_matchs = []
    for user_metrics in population_metrics:
        user_accuracies = []
        user_f1s = []
        user_one_matchs = []
        for user_session_metric in user_metrics:
            for key, value in user_session_metric.items():
                if key.endswith("validation/acc"):
                    user_accuracies.append(value)
                elif key.endswith("validation/f1"):
                    user_f1s.append(value)
                elif key.endswith("validation/one_match"):
                    user_one_matchs.append(value)
        population_accuracies.append(np.mean(user_accuracies))
        population_f1s.append(np.mean(user_f1s))
        population_one_matchs.append(np.mean(user_one_matchs))

    logger.log({
        "population/mean_accuracy": np.mean(population_accuracies),
        "population/mean_f1": np.mean(population_f1s),
        "population/mean_one_match": np.mean(population_one_matchs),
    })


def fail_early():
    configs.fail_early()


if __name__ == '__main__':
    fail_early()
    main()
