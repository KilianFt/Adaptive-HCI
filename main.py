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
    hashlib.sha256("Kilian".encode("utf-8")).hexdigest()[:15],
    hashlib.sha256("Manuel".encode("utf-8")).hexdigest()[:15],
]


def main():
    if sys.gettrace():
        experiment_config = configs.SmokeConfig()
    else:
        experiment_config = configs.BaseConfig()
    experiment_config = configs.SmokeConfig()

    try:
        entity = "delvermm" if "delverm" in os.getlogin() else "kilian"
    except OSError:  # Happens on mila cluster
        entity = "delvermm"

    logger, experiment_config = buddy_setup(experiment_config, entity=entity)

    general_model = train_general_model.main(logger, experiment_config)

    population_metrics = []
    for user_hash in train_users:
        initial_model = copy.deepcopy(general_model)
        finetuned_user_model = finetune_user_model.main(initial_model, user_hash, experiment_config)
        user_model, metrics = continuously_train_user_model.main(finetuned_user_model, user_hash, experiment_config)
        population_metrics.append(metrics)

    population_accuracies = []
    for user_metrics in population_metrics:
        user_accuracies = []
        for user_session_metric in user_metrics:
            for key, value in user_session_metric.items():
                if key.endswith("validation/acc"):
                    user_accuracies.append(value)
        population_accuracies.append(np.mean(user_accuracies))

    logger.log(
        "population/mean_accuracy",
        np.mean(population_accuracies)
    )


def fail_early():
    configs.fail_early()


if __name__ == '__main__':
    fail_early()
    main()
