import copy
import hashlib
import os
import sys
import collections
import dataclasses

import numpy as np
import lightning as L

import configs
import train_general_model
import train_autowriter
import finetune_user_model
import continuously_train_user_model
from deployment.buddy import buddy_setup
from interface.game import GameState # needed to load data for finetuning

train_users = [
    hashlib.sha256("Kilian".encode("utf-8")).hexdigest()[:15],
    # hashlib.sha256("Manuel".encode("utf-8")).hexdigest()[:15],
]


def get_user_results(user_metrics):
    user_results = collections.defaultdict(list)
    for user_session_metric in user_metrics:
        for key, value in user_session_metric.items():
            if key.endswith("validation/acc"):
                user_results['acc'].append(value)
            elif key.endswith("validation/f1"):
                user_results['f1'].append(value)
            elif key.endswith("validation/one_match"):
                user_results['one_match'].append(value)
    return user_results


def main():
    if sys.gettrace():
        experiment_config = configs.SmokeConfig()
    else:
        experiment_config = configs.BaseConfig()

    try:
        entity = "delvermm" if "delverm" in os.getlogin() else "kilian"
    except OSError:  # Happens on mila cluster
        entity = "delvermm"

    L.seed_everything(experiment_config.seed)

    logger, experiment_config = buddy_setup(experiment_config, entity=entity)

    general_model = train_general_model.main(logger, experiment_config)
    auto_writer = train_autowriter.main(experiment_config.auto_writer)
    population_metrics = collections.defaultdict(list)

    for user_hash in train_users:
        initial_model = copy.deepcopy(general_model)
        # TODO check for user data, if persent finetune
        finetuned_user_model = finetune_user_model.main(initial_model, user_hash, experiment_config)
        return

        user_model, user_metrics = continuously_train_user_model.main(finetuned_user_model, user_hash, experiment_config)

        user_results = get_user_results(user_metrics)
        for key, value in user_results.items():
            population_metrics[key].append(np.mean(value))

    logger.log({
        "population/mean_accuracy": np.mean(population_metrics['acc']),
        "population/mean_f1": np.mean(population_metrics["f1"]),
        "population/mean_one_match": np.mean(population_metrics["one_match"]),
    })


def fail_early():
    configs.fail_early()


if __name__ == '__main__':
    fail_early()
    main()
