import sys

sys.path.insert(0, "../jz")

from auto_sbatch import run_exp
import itertools


def run_batch(
    experiments,
    param_names,
    n_seeds=1,
    long_run=False,
    gpu=True,
    cpu_count=1,
):
    """
    makes all possible combinations of parameters within the given parameters, and for each combination calls on
    run_exp from looper.py to launch them with SLURM

    parameters
    ----------
    experiments
    param_names : list of strings
        all parameter names, in same order of apparition as in experiments
    n_seeds : int, default = 1
        number of repetitions of the same experiment to take place (another, more ordered, way to do this is to give n different log_paths, avoiding file name repetitions.)
    long_run : bool, default = False
        Wether or not to request a qos4 with long runtime on the jean zay cluster, with 100h
    gpu : bool, default = True
        Whether to request a gpu node, or a cpu node (multiprocessing only works on cpu for now)
    cpu_count: int, default = 1
        number of cpus allocated to the task. More cpus gives access to more memory, and more possible parallel processes.
    """
    parameters = ""
    for experiment in experiments:
        for i in range(len(experiment)):

            parameters += f" {param_names[i]}={experiment[i]}"

        script = "simulate.py"

        run_exp(
            script=script,
            parameters=parameters,
            gpu=gpu,
            long_run=long_run,
        )

def exp_presentation(gpu, project, env_type, model, num_gens, trial, long_run=False):
    experiments = [[project, env_type, model, num_gens, trial]]
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial"]
    run_batch(
        experiments,
        param_names,
        long_run=long_run,
        gpu=gpu,
    )

def exp_slower_abrupt(gpu,  trial, long_run=False):
    project = "Maslin/present_investigate/slower_abrupt"
    env_type = "combined"
    model = "hybrid"
    num_gens = 10000
    factor_time_abrupt = 10

    experiments = [[project, env_type, model, num_gens, trial, factor_time_abrupt]]
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--factor_time_abrupt"]
    run_batch(
        experiments,
        param_names,
        long_run=long_run,
        gpu=gpu,
    )


def exp_slower_variable(gpu, trial, long_run=False):
    project = "Maslin/present_investigate/slower_variable"
    env_type = "combined",
    model = "hybrid"
    num_gens = 10000
    factor_time_variable = 10

    experiments = [[project, env_type, model, num_gens, trial, factor_time_variable]]
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--factor_time_variable"]
    run_batch(
        experiments,
        param_names,
        long_run=long_run,
        gpu=gpu,
    )


def exp_less_variable(gpu, trial, long_run=False):
    project = "Maslin/present_investigate/less_variable"
    env_type = "combined"
    model = "hybrid"
    num_gens = 10000
    var_freq = 20

    experiments = [[project, env_type, model, num_gens, trial, var_freq]]
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--var_freq"]
    run_batch(
        experiments,
        param_names,
        long_run=long_run,
        gpu=gpu,
    )

def exp_even_less_variable(gpu, trial, long_run=False):
    project = "Maslin/present_investigate/less_variable"
    env_type = "combined"
    model = "hybrid"
    num_gens = 10000
    var_freq = 50
    var_SD = 0.1

    experiments = [[project, env_type, model, num_gens, trial, var_freq, var_SD]]
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--var_freq", "--var_SD"]
    run_batch(
        experiments,
        param_names,
        long_run=long_run,
        gpu=gpu,
    )

def exp_initialize(gpu, trial, long_run=False):
    project = "Maslin/present_investigate/initialize"
    env_type = "combined"
    model = "hybrid"
    num_gens = 10000
    num_agents = 10

    experiments = [[project, env_type, model, num_gens, trial, num_agents]]
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--num_agents"]
    run_batch(
        experiments,
        param_names,
        long_run=long_run,
        gpu=gpu,
    )

if __name__ == "__main__":
    trials = 50
    for trial in range(trials):
        # exp_presentation(gpu=False, project="Maslin/present_conf", env_type="combined", model="hybrid",
        #                  num_gens=10000, trial=trial)
        #exp_slower_abrupt(gpu=False, trial=trial)
        exp_slower_variable(gpu=False,  trial=trial)
        #exp_less_variable(gpu=False,  trial=trial)
        #exp_initialize(gpu=False, trial=trial)
        exp_even_less_variable(gpu=False, trial=trial)
