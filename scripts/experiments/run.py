import sys
import os

sys.path.insert(0, "../jz")

from auto_sbatch import run_exp
import itertools
import numpy as np

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

        if long_run:
            time = "80:00:00"
        else:
            time = "15:00:00"
        run_exp(
            script=script,
            parameters=parameters,
            gpu=gpu,
            time=time,
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
    env_type = "combined"
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


def exp_tune_var1(gpu, trial, long_run=False):
    project = "Maslin/present_investigate/tune_var1"
    env_type = "combined"
    model = "hybrid"
    num_gens = 10000
    var_freq = 100
    var_SD = 0.2

    experiments = [[project, env_type, model, num_gens, trial, var_freq, var_SD]]
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--var_freq", "--var_SD"]
    run_batch(
        experiments,
        param_names,
        long_run=long_run,
        gpu=gpu,
    )

def exp_tune_var2(gpu, trial, long_run=False):
    project = "Maslin/present_investigate/batch_3/tune_var2"
    env_type = "combined"
    model = "hybrid"
    num_gens = 10000
    var_freq = 100
    var_SD = 0.2
    factor_time_variable = 10
    factor_time_abrupt = 10


    experiments = [[project, env_type, model, num_gens, trial, var_freq, var_SD, factor_time_variable,
                    factor_time_abrupt]]
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--var_freq", "--var_SD",
                   "--factor_time_variable", "--factor_time_abrupt"]
    run_batch(
        experiments,
        param_names,
        long_run=long_run,
        gpu=gpu,
    )

def exp_tune_var3(gpu, trial, long_run=False):
    project = "Maslin/present_investigate/batch_3/tune_var3"
    env_type = "combined"
    model = "hybrid"
    num_gens = 30000
    var_freq = 30
    var_SD = 0.2
    factor_time_variable = 10
    factor_time_abrupt = 10


    experiments = [[project, env_type, model, num_gens, trial, var_freq, var_SD, factor_time_variable,
                    factor_time_abrupt]]
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--var_freq", "--var_SD",
                   "--factor_time_variable", "--factor_time_abrupt"]
    run_batch(
        experiments,
        param_names,
        long_run=long_run,
        gpu=gpu,
    )

def exp_tune_var5(gpu, trial, long_run=False):
    project = "Maslin/present_investigate/batch_3/tune_var5"
    env_type = "combined"
    model = "hybrid"
    num_gens = 30000
    var_freq = 30
    var_SD = 0.2
    factor_time_variable = 10
    factor_time_abrupt = 10


    experiments = [[project, env_type, model, num_gens, trial, var_freq, var_SD, factor_time_variable,
                    factor_time_abrupt]]
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--var_freq", "--var_SD",
                   "--factor_time_variable", "--factor_time_abrupt"]
    run_batch(
        experiments,
        param_names,
        long_run=long_run,
        gpu=gpu,
    )

def exp_tune_var6(gpu, trial, long_run=False):
    project = "Maslin/present_investigate/batch_3/tune_var6"
    env_type = "combined"
    model = "hybrid"
    num_gens = 30000
    var_freq = 85
    var_SD = 0.2
    factor_time_variable = 10
    factor_time_abrupt = 10


    experiments = [[project, env_type, model, num_gens, trial, var_freq, var_SD, factor_time_variable,
                    factor_time_abrupt]]
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--var_freq", "--var_SD",
                   "--factor_time_variable", "--factor_time_abrupt"]
    run_batch(
        experiments,
        param_names,
        long_run=long_run,
        gpu=gpu,
    )

#def exp_tune_var6_irreg(gpu, trial,  mode, long_run=False):


def exp_total_scale(gpu, trial, long_run=False):
    project = "Maslin/present_investigate/batch_3/total_scale"
    env_type = "combined"
    model = "hybrid"
    num_gens = 10000
    var_freq = 30
    var_SD = 0.2
    factor_time_variable = 10
    factor_time_abrupt = 10


    experiments = [[project, env_type, model, num_gens, trial, var_freq, var_SD, factor_time_variable,
                    factor_time_abrupt]]
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--var_freq", "--var_SD",
                   "--factor_time_variable", "--factor_time_abrupt"]


def exp_parametric(gpu, trial,  mode, long_run=False):
    var_freq_values = np.arange(10, 100, 20)
    factor_time_abrupt_values =  np.arange(5, 15, 3)
    top_dir = "Maslin/parametric/"
    experiments = []
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--var_freq", "--var_SD",
                   "--factor_time_variable", "--factor_time_abrupt"]
    env_type = "combined"
    model = "hybrid"
    num_gens = 10000
    factor_time_variable = 5
    var_SD = 0.2



    for var_freq in var_freq_values:
        for factor_time_abrupt in factor_time_abrupt_values:
            project = top_dir + "freq_" + str(var_freq) + "_time_" + str(factor_time_abrupt)
            new_exp = [project, env_type, model, num_gens, trial, var_freq, var_SD, factor_time_variable,
                    factor_time_abrupt]
            experiments.append(new_exp)
            if mode == "local":
                command = "python simulate.py "
                for idx, el in enumerate(param_names):
                    command += el + " " + str(new_exp[idx]) + " "
                print(command)
                os.system("bash -c '{}'".format(command))


    if mode == "server":
        run_batch(
            experiments,
            param_names,
            long_run=long_run,
            gpu=gpu,
        )

def debug(gpu, trial,  mode, long_run=False):
    project = "Maslin/present_investigate/batch_3/total_scale"
    env_type = "combined"
    model = "hybrid"
    num_gens = 10000
    var_freq = 30
    var_SD = 0.2
    factor_time_variable = 10
    factor_time_abrupt = 10
    irregular = [1]


    experiments = [[project, env_type, model, num_gens, trial, var_freq, var_SD, factor_time_variable,
                    factor_time_abrupt]]
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--var_freq", "--var_SD",
                   "--factor_time_variable", "--factor_time_abrupt"]

    if mode == "local":
        command = "python simulate.py "
        new_exp = [project, env_type, model, num_gens, trial, var_freq, var_SD, factor_time_variable,
                   factor_time_abrupt, irregular]
        for idx, el in enumerate(param_names):
            command += el + " " + str(new_exp[idx]) + " "
        print(command)
        os.system("bash -c '{}'".format(command))


def parametric_abrupt(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    top_dir = "Maslin/1D_mutate/parametric_abrupt/"
    experiments = []
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--factor_time_abrupt",
                   "--mutate_rate"]
    env_type = "combined"
    model = "hybrid"
    num_gens = 1000
    factor_time_abrupt_values =  np.arange(2, 15, 3)
    mutation_values = [0.0001, 0.0005, 0.001, 0.005, 0.01]


    for mutation in mutation_values:
        for factor_time_abrupt in factor_time_abrupt_values:
            project = top_dir + "mut_" + str(mutation) + "_time_" + str(factor_time_abrupt)
            new_exp = [project, env_type, model, num_gens, trial,   factor_time_abrupt, mutate_rate]
            experiments.append(new_exp)
            if mode == "local":
                command = "python simulate.py "
                for idx, el in enumerate(param_names):
                    command += el + " " + str(new_exp[idx]) + " "
                print(command)
                os.system("bash -c '{}'".format(command))


    if mode == "server":
        run_batch(
            experiments,
            param_names,
            long_run=long_run,
            gpu=gpu,
        )

if __name__ == "__main__":
    trials = int(sys.argv[1])
    mode = sys.argv[2] # server for jz experiments and local otherwise
    for trial in range(1,trials+1):
        # exp_presentation(gpu=False, project="Maslin/present_conf", env_type="combined", model="hybrid",
        #                  num_gens=10000, trial=trial)
        #exp_slower_abrupt(gpu=False, trial=trial)
        #exp_slower_variable(gpu=False,  trial=trial)
        #exp_less_variable(gpu=False,  trial=trial)
        #exp_initialize(gpu=False, trial=trial)
        #exp_even_less_variable(gpu=False, trial=trial)
        #exp_tune_var1(gpu=False, trial=trial)
        #exp_tune_var2(gpu=False, trial=trial)
        #exp_tune_var5(gpu=False, trial=trial)
        #exp_tune_var6_irreg(gpu=False, trial=trial, mode=mode)
        #debug(gpu=False, trial=trial, mode=mode)\
        #exp_parametric(gpu=True, trial=trial, mode=mode)
        parametric_abrupt(gpu=True, trial=trial, mode=mode)
