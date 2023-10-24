import dataclasses
import os
import sys
import configs
import experiment_buddy


def buddy_setup(exp_config: configs.BaseConfig, entity):
    import wandb
    wandb_kwargs = dict(
        monitor_gym=False,
        entity=entity,
        settings=wandb.Settings(start_method="thread"),
        save_code=True,
        config=exp_config.model_dump(),
    )
    hostname = exp_config.hostname
    sweep_config = exp_config.sweep_config
    proc_num = exp_config.proc_num

    # hostname = "aws://t4g.micro"
    if sys.gettrace() is not None and os.environ.get("BUDDY_DEBUG_DEPLOYMENT") is None:
        hostname = ""
        sweep_config = ""
    esh = "\n".join(l.strip() for l in """
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=64G
    #SBATCH --time=12:00:00
    #SBATCH --gres=gpu:32gb:1
        """.strip().split("\n")
                    ) + "\n"
    extra_modules = None
    if hostname == "mila":
        esh += "#SBATCH --partition=long\n"
        extra_modules = [
            "anaconda/3",
            "cuda/11.1",
            "pytorch/1.8.1"
        ]
    elif "cc" in hostname:
        esh += "#SBATCH --partition=cpubase_bycore_b4\n"
        esh += "#SBATCH --account=rrg-dprecup\n"
        # esh += "#SBATCH --account=rrg-bengioy-ad\n"
        extra_modules = [
            "anaconda/3",
            # "pytorch/1.7", # CC doesn't have pytorch, should be a package
            "cuda/11.1",
            "pytorch/1.8.1"
        ]
    else:
        esh = ""
    # has_conda_env_param = inspect.signature(experiment_buddy.deploy).parameters.get("conda_env") is not None
    has_conda_env_param = False
    if has_conda_env_param:
        tb = experiment_buddy.deploy(
            hostname, wandb_kwargs=wandb_kwargs, extra_slurm_headers=esh, sweep_definition=sweep_config,
            proc_num=proc_num,
            extra_modules=extra_modules, conda_env="traces_llm"
        )
    else:
        tb = experiment_buddy.deploy(
            hostname, wandb_kwargs=wandb_kwargs, extra_slurm_headers=esh, sweep_definition=sweep_config,
            proc_num=proc_num,
            extra_modules=extra_modules
        )
    updated_exp_config = exp_config.__class__(**tb.run.config)
    return tb, updated_exp_config
