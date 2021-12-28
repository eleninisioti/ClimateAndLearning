import sys
import os

sys.path.insert(0, "../jz")

from auto_sbatch import run_exp
import itertools
import numpy as np
import datetime

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
                   "--mutate_rate", "--factor_time_steady"]
    env_type = "combined"
    model = "hybrid"
    num_gens = 1000
    factor_time_abrupt_values =  np.arange(15, 50, 5)
    factor_time_abrupt_values = factor_time_abrupt_values[::-1]
    mutation_values = [0.001]
    factor_time_steady = 0.5


    for mutate_rate in mutation_values:
        for factor_time_abrupt in factor_time_abrupt_values:
            project = top_dir + "mut_" + str(mutate_rate) + "_time_" + str(factor_time_abrupt)
            new_exp = [project, env_type, model, num_gens, trial,   factor_time_abrupt, mutate_rate, factor_time_steady]
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

def parametric_variable(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    top_dir = "Maslin/1D_mutate/parametric_variable_highermutate/"
    experiments = []
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--factor_time_abrupt",
                   "--factor_time_variable", "--mutate_rate", "--var_freq", "--var_SD", "--factor_time_steady"]
    env_type = "combined"
    model = "hybrid"
    num_gens = 1500
    factor_time_abrupt = 2
    mutate_rate = 0.01
    var_freq_values = np.arange(5, 30, 10)
    var_SD_values = [0.1, 0.4]
    factor_time_variable = 1
    factor_time_steady = 0.5

    for var_SD in var_SD_values:
        for var_freq in var_freq_values:
            project = top_dir + "SD_" + str(var_SD) + "_var_" + str(var_freq)
            new_exp = [project, env_type, model, num_gens, trial,  factor_time_abrupt, factor_time_variable,
                       mutate_rate, var_freq, var_SD, factor_time_steady]
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

def test_low(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    top_dir = "Maslin/1D_mutate/test_low/"
    experiments = []
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--factor_time_abrupt",
                   "--factor_time_steady"]
    env_type = "combined"
    model = "hybrid"
    num_gens = 1000
    factor_time_abrupt_values = [1,2,3,4]
    factor_time_steady = 1


    for factor_time_abrupt in factor_time_abrupt_values:
        project = top_dir + "_time_" + str(factor_time_abrupt)
        new_exp = [project, env_type, model, num_gens, trial,   factor_time_abrupt, factor_time_steady]
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

def parametric_low(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    top_dir = "Maslin/1D_mutate/debug/" + project + "/"

    experiments = []
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--factor_time_abrupt",
                   "--factor_time_steady", "--low_value", "--num_niches","--var_freq", "--var_SD", "--survival_type",
                   "--mutate_rate", "--genome_type"]
    env_type = "combined"
    model = "hybrid"
    num_gens = 1500
    factor_time_abrupt_values = [4]
    factor_time_steady = 1
    low_values = [1]
    niches_number_values = [1]
    var_freq = 10
    var_SD = 0.1
    survival_types = ["FP-Grove", "mixed"]
    mutate_rate = 0.001
    genome_types = ["1D_mutate_fixed", "1D_mutate"]

    for genome_type in genome_types:
        for survival_type in survival_types:

            for factor_time_abrupt in factor_time_abrupt_values:
                for low_value in low_values:
                    for niches_number in niches_number_values:
                        project = top_dir + "survival_" + survival_type +"_low_" + str(low_value) + "_niches_" + str(
                            niches_number)
                        new_exp = [project, env_type, model, num_gens, trial,   factor_time_abrupt, factor_time_steady,
                                   low_value, niches_number, var_freq, var_SD, survival_type, mutate_rate, genome_type]
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

def targeted_low(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    top_dir = "Maslin/1D_mutate/parametric_low/"
    experiments = []
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--factor_time_abrupt",
                   "--factor_time_steady", "--low_value", "--num_niches","--var_freq", "--var_SD", "--survival_type"]
    env_type = "combined"
    model = "hybrid"
    num_gens = 1500
    factor_time_abrupt_values = [3]
    factor_time_steady = 1
    low_values = [1]
    niches_number_values = [400]
    var_freq = 10
    var_SD = 0.2
    survival_types = ["FP-global", "no-pressure", "no-oressure-random", "no-presure-fifo"]
    survival_types = ["mixed"]

    for survival_type in survival_types:

        for factor_time_abrupt in factor_time_abrupt_values:
            for low_value in low_values:
                for niches_number in niches_number_values:
                    project = top_dir + "survival_" + survival_type +"_low_" + str(low_value) + "_niches_" + str(
                        niches_number)
                    new_exp = [project, env_type, model, num_gens, trial,   factor_time_abrupt, factor_time_steady,
                               low_value, niches_number, var_freq, var_SD, survival_type]
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
            gpu=gpu)

def parametric_Grove(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    top_dir = "Maslin/debug/parametric/" + project + "/"

    experiments = []
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--trial", "--survival_type",
                   "--mutate_rate", "--genome_type", "--extinctions", "--factor_time_abrupt",
                   "--factor_time_steady", "--num_niches",
                   "--only_climate"]
    factor_time_abrupt_values = [7, 10]
    factor_time_steady_values = [5]
    env_type = "change"
    model = "hybrid"
    num_gens = 1200
    survival_types = ["capacity-fitness", "limited-capacity", "FP-Grove"]
    #survival_types = ["FP-Grove"]
    mutate_rate = 0.001
    genome_types = ["1D_mutate_fixed", "1D_mutate", "1D"]
    genome_types = ["1D_mutate"]
    extinctions = [0, 1]
    num_niches_values = [1, 100, 500, 1000]
    climate_only = 0

    for num_niches in num_niches_values:
        for factor_time_abrupt in factor_time_abrupt_values:
            for factor_time_steady in factor_time_steady_values:
                for genome_type in genome_types:
                    for survival_type in survival_types:
                        for extinction in extinctions:

                            project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                      str(extinction) + "_scale_abrupt_" + str(factor_time_abrupt) + "_scale_steady_"\
                                      + str(factor_time_steady )+ "_num_niches_" + \
                                      str(num_niches)
                            new_exp = [project, env_type, model, num_gens, trial, survival_type, mutate_rate, genome_type,
                                       extinction, factor_time_abrupt, factor_time_steady,num_niches, climate_only]
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

def parametric_rest(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    top_dir = "Maslin/debug/parametric/" + project + "/"

    experiments = []
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--num_trials", "--survival_type",
                   "--mutate_rate", "--genome_type", "--extinctions", "--factor_time_abrupt",
                   "--factor_time_steady", "--num_niches",
                   "--only_climate"]
    factor_time_abrupt_values = [7]
    factor_time_steady_values = [5]
    env_type = "change"
    model = "hybrid"
    num_gens = 1300
    survival_types = ["capacity-fitness", "limited-capacity", "FP-Grove"]
    #survival_types = ["FP-Grove"]
    mutate_rate = 0.001
    genome_types = ["1D_mutate_fixed", "1D", "1D_mutate"]
    extinctions = [1]
    num_niches_values = [1, 10, 100]
    climate_only = 0

    for num_niches in num_niches_values:
        for factor_time_abrupt in factor_time_abrupt_values:
            for factor_time_steady in factor_time_steady_values:
                for genome_type in genome_types:
                    for survival_type in survival_types:
                        for extinction in extinctions:

                            project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                      str(extinction) + "_scale_abrupt_" + str(factor_time_abrupt) + "_scale_steady_"\
                                      + str(factor_time_steady )+ "_num_niches_" + \
                                      str(num_niches)
                            new_exp = [project, env_type, model, num_gens, trial, survival_type, mutate_rate, genome_type,
                                       extinction, factor_time_abrupt, factor_time_steady,num_niches, climate_only]
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

def parametric_1D(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    top_dir = "Maslin/debug/parametric/" + project + "_fixed/"

    experiments = []
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--num_trials", "--survival_type",
                   "--mutate_rate", "--genome_type", "--extinctions", "--factor_time_abrupt",
                   "--factor_time_steady", "--num_niches", "--only_climate"]
    factor_time_abrupt_values = [3]
    factor_time_steady_values = [5]
    env_type = "change"
    model = "hybrid"
    num_gens = 1300
    survival_types = ["capacity-fitness"]
    survival_types = ["capacity-fitness", "limited-capacity", "FP-Grove"]

    #survival_types = ["FP-Grove"]
    mutate_rate = 0.001
    genome_types = ["1D_mutate", "1D_mutate_fixed"]
    extinctions = [1]
    num_niches_values = [1, 10, 100]
    climate_only = 0

    for num_niches in num_niches_values:
        for factor_time_abrupt in factor_time_abrupt_values:
            for factor_time_steady in factor_time_steady_values:
                for genome_type in genome_types:
                    for survival_type in survival_types:
                        for extinction in extinctions:

                            project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                      str(extinction) + "_scale_abrupt_" + str(factor_time_abrupt) + "_scale_steady_"\
                                      + str(factor_time_steady )+ "_num_niches_" + \
                                      str(num_niches)
                            new_exp = [project, env_type, model, num_gens, trial, survival_type, mutate_rate, genome_type,
                                       extinction, factor_time_abrupt, factor_time_steady,num_niches, climate_only]
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

def parametric(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    top_dir = "Maslin/debug/parametric/" + project + "/"

    experiments = []
    param_names = ["--project", "--env_type", "--num_gens", "--num_trials", "--survival_type",
                   "--mutate_mutate_rate", "--genome_type", "--extinctions", "--factor_time_abrupt",
                   "--factor_time_steady", "--num_niches", "--only_climate"]
    factor_time_abrupt_values = [1, 7]
    factor_time_steady_values = [5]
    env_type = "change"
    num_gens = 1300
    survival_types = ["capacity-fitness", "FP-Grove", "limited-capacity"]
    mutate_mutate_rate = 0.001
    genome_types = ["1D", "1D_mutate", "1D_mutate_fixed"]
    extinctions = [1]
    num_niches_values = [1, 10, 100]
    climate_only = 0

    for num_niches in num_niches_values:
        for factor_time_abrupt in factor_time_abrupt_values:
            for factor_time_steady in factor_time_steady_values:
                for genome_type in genome_types:
                    for survival_type in survival_types:
                        for extinction in extinctions:

                            project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                      str(extinction) + "_scale_abrupt_" + str(factor_time_abrupt) + "_scale_steady_"\
                                      + str(factor_time_steady )+ "_num_niches_" + \
                                      str(num_niches)
                            new_exp = [project, env_type, num_gens, trial, survival_type, mutate_mutate_rate,
                                       genome_type,
                                       extinction, factor_time_abrupt, factor_time_steady,num_niches, climate_only]
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


def debug_CF(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    top_dir = "Maslin/debug/parametric/" + project + "_debug/"

    experiments = []
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--num_trials", "--survival_type",
                   "--mutate_rate", "--genome_type", "--extinctions", "--factor_time_abrupt",
                   "--factor_time_steady", "--num_niches", "--only_climate"]
    factor_time_abrupt_values = [1]
    factor_time_steady_values = [5]
    env_type = "change"
    model = "hybrid"
    num_gens = 700
    survival_types = ["capacity-fitness"]
    #survival_types = ["capacity-fitness"]
    mutate_rate = 0.001
    genome_types = ["1D"]
    extinctions = [1]
    num_niches_values = [1]
    climate_only = 0

    for num_niches in num_niches_values:
        for factor_time_abrupt in factor_time_abrupt_values:
            for factor_time_steady in factor_time_steady_values:
                for genome_type in genome_types:
                    for survival_type in survival_types:
                        for extinction in extinctions:

                            project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                      str(extinction) + "_scale_abrupt_" + str(factor_time_abrupt) + "_scale_steady_"\
                                      + str(factor_time_steady )+ "_num_niches_" + \
                                      str(num_niches)
                            new_exp = [project, env_type, model, num_gens, trial, survival_type, mutate_rate, genome_type,
                                       extinction, factor_time_abrupt, factor_time_steady,num_niches, climate_only]
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

def parametric_1D_abrupt(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    top_dir = "Maslin/debug/parametric/" + project + "_fixed/"

    experiments = []
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--num_trials", "--survival_type",
                   "--mutate_rate", "--genome_type", "--extinctions", "--factor_time_abrupt",
                   "--factor_time_steady", "--num_niches", "--only_climate"]
    factor_time_abrupt_values = [1]
    factor_time_steady_values = [5]
    env_type = "change"
    model = "hybrid"
    num_gens = 1300
    survival_types = ["capacity-fitness"]
    survival_types = ["capacity-fitness", "limited-capacity", "FP-Grove"]

    #survival_types = ["FP-Grove"]
    mutate_rate = 0.001
    genome_types = ["1D"]
    extinctions = [0]
    num_niches_values = [1, 10, 100]
    climate_only = 0

    for num_niches in num_niches_values:
        for factor_time_abrupt in factor_time_abrupt_values:
            for factor_time_steady in factor_time_steady_values:
                for genome_type in genome_types:
                    for survival_type in survival_types:
                        for extinction in extinctions:

                            project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                      str(extinction) + "_scale_abrupt_" + str(factor_time_abrupt) + "_scale_steady_"\
                                      + str(factor_time_steady )+ "_num_niches_" + \
                                      str(num_niches)
                            new_exp = [project, env_type, model, num_gens, trial, survival_type, mutate_rate, genome_type,
                                       extinction, factor_time_abrupt, factor_time_steady,num_niches, climate_only]
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


def debug_1D_LC(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    top_dir = "Maslin/debug/parametric/" + project + "_debug1DLC/"

    experiments = []
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--num_trials", "--survival_type",
                   "--mutate_rate", "--genome_type", "--extinctions", "--factor_time_abrupt",
                   "--factor_time_steady", "--num_niches", "--only_climate"]
    factor_time_abrupt_values = [1,3, 5]
    factor_time_steady_values = [5]
    env_type = "change"
    model = "hybrid"
    num_gens = 1300
    survival_types = ["limited-capacity"]

    #survival_types = ["FP-Grove"]
    mutate_rate = 0.001
    genome_types = ["1D"]
    extinctions = [1]
    num_niches_values = [10]
    climate_only = 0

    for num_niches in num_niches_values:
        for factor_time_abrupt in factor_time_abrupt_values:
            for factor_time_steady in factor_time_steady_values:
                for genome_type in genome_types:
                    for survival_type in survival_types:
                        for extinction in extinctions:

                            project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                      str(extinction) + "_scale_abrupt_" + str(factor_time_abrupt) + "_scale_steady_"\
                                      + str(factor_time_steady )+ "_num_niches_" + \
                                      str(num_niches)
                            new_exp = [project, env_type, model, num_gens, trial, survival_type, mutate_rate, genome_type,
                                       extinction, factor_time_abrupt, factor_time_steady,num_niches, climate_only]
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

def debug_1D(gpu, trial,  mode, long_run=False):
    #var_freqvalues = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    top_dir = "Maslin/debug/parametric/" + project + "_debug1D/"

    experiments = []
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--num_trials", "--survival_type",
                   "--mutate_rate", "--genome_type", "--extinctions", "--factor_time_abrupt",
                   "--factor_time_steady", "--num_niches", "--only_climate", "--scale_weights"]
    factor_time_abrupt_values = [7]
    factor_time_steady_values = [5]
    env_type = "change"
    model = "hybrid"
    num_gens = 1300
    survival_types = ["limited-capacity"]

    survival_types = ["FP-Grove", "capacity-fitness"]
    mutate_rate = 0.001
    genome_types = ["1D"]
    extinctions = [1]
    num_niches_values = [1]
    climate_only = 0
    scale_weights = 1

    for num_niches in num_niches_values:
        for factor_time_abrupt in factor_time_abrupt_values:
            for factor_time_steady in factor_time_steady_values:
                for genome_type in genome_types:
                    for survival_type in survival_types:
                        for extinction in extinctions:

                            project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                      str(extinction) + "_scale_abrupt_" + str(factor_time_abrupt) + "_scale_steady_"\
                                      + str(factor_time_steady )+ "_num_niches_" + \
                                      str(num_niches)
                            new_exp = [project, env_type, model, num_gens, trial, survival_type, mutate_rate, genome_type,
                                       extinction, factor_time_abrupt, factor_time_steady,num_niches, climate_only,
                                       scale_weights]
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


def parametric_sin(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    top_dir = "Maslin/debug/parametric/" + project + "_fixed/"

    experiments = []
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--num_trials", "--survival_type",
                   "--mutate_rate", "--genome_type", "--extinctions", "--factor_time_abrupt",
                   "--factor_time_steady", "--num_niches", "--only_climate"]
    factor_time_abrupt_values = [7]
    factor_time_steady_values = [5]
    env_type = "sin"
    model = "hybrid"
    num_gens = 1300
    survival_types = ["capacity-fitness", "limited-capacity", "FP-Grove"]
    mutate_rate = 0.001
    genome_types = ["1D", "1D_mutate", "1D_mutate_fixed"]
    extinctions = [1]
    num_niches_values = [1, 10, 100]
    climate_only = 0

    for num_niches in num_niches_values:
        for factor_time_abrupt in factor_time_abrupt_values:
            for factor_time_steady in factor_time_steady_values:
                for genome_type in genome_types:
                    for survival_type in survival_types:
                        for extinction in extinctions:

                            project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                      str(extinction) + "_scale_abrupt_" + str(factor_time_abrupt) + "_scale_steady_"\
                                      + str(factor_time_steady )+ "_num_niches_" + \
                                      str(num_niches)
                            new_exp = [project, env_type, model, num_gens, trial, survival_type, mutate_rate, genome_type,
                                       extinction, factor_time_abrupt, factor_time_steady,num_niches, climate_only]
                            experiments.append(new_exp)
                            if mode == "local":
                                command = "python simulate.py "
                                for idx, el in enumerate(param_names):
                                    command += el + " " + str(new_exp[idx]) + " "
                                print(command)
                                os.system("bash -c '{}'".format(command))
                                #quit()

    if mode == "server":
        run_batch(
            experiments,
            param_names,
            long_run=long_run,
            gpu=gpu,
        )

def parametric_Maslin(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    top_dir = "Maslin/debug/parametric/" + project + "_fixed/"

    experiments = []
    param_names = ["--project", "--env_type", "--model", "--num_gens", "--num_trials", "--survival_type",
                   "--mutate_rate", "--genome_type", "--extinctions", "--factor_time_abrupt",
                   "--factor_time_steady", "--num_niches", "--only_climate", "--factor_time_variable",
                   "--var_SD"]
    factor_time_abrupt_values = [3]
    factor_time_steady_values = [1]
    factor_time_variable_values = [1]
    var_freq_values = np.arange(10, 100, 20)
    var_SD = 0.2
    env_type = "combined"
    model = "hybrid"
    num_gens = 1500
    survival_types = ["capacity-fitness", "limited-capacity", "FP-Grove"]
    mutate_rate = 0.001
    genome_types = ["1D", "1D_mutate", "1D_mutate_fixed"]
    extinctions = [1]
    num_niches_values = [1, 10, 100]
    climate_only = 0

    for var_freq in var_freq_values:

        for factor_time_variable in factor_time_variable_values:

            for num_niches in num_niches_values:
                for factor_time_abrupt in factor_time_abrupt_values:
                    for factor_time_steady in factor_time_steady_values:
                        for genome_type in genome_types:
                            for survival_type in survival_types:
                                for extinction in extinctions:

                                    project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                              str(extinction) + "_scale_abrupt_" + str(factor_time_abrupt) + "_scale_steady_"\
                                              + str(factor_time_steady )+ "_num_niches_" + \
                                              str(num_niches) + "_scale_variable_"+ str(factor_time_variable) + \
                                              "_var_freq_"+ str(var_freq )
                                    new_exp = [project, env_type, model, num_gens, trial, survival_type, mutate_rate, genome_type,
                                               extinction, factor_time_abrupt, factor_time_steady,num_niches,
                                               climate_only, factor_time_variable, var_freq]
                                    experiments.append(new_exp)
                                    if mode == "local":
                                        command = "python simulate.py "
                                        for idx, el in enumerate(param_names):
                                            command += el + " " + str(new_exp[idx]) + " "
                                        print(command)
                                        os.system("bash -c '{}'".format(command))
                                        #quit()

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
    for trial in range(1, trials+1):
        parametric(gpu=True, trial=trial, mode=mode)
        #debug_CF(gpu=True, trial=trial, mode=mode)
        #parametric_sin(gpu=True, trial=trial, mode=mode)
        #parametric_sin(gpu=True, trial=trial, mode=mode)
        #parametric_Maslin(gpu=True, trial=trial, mode=mode)
        #parametric_rest(gpu=True, trial=trial, mode=mode)
        #debug_1D_LC(gpu=True, trial=trial, mode=mode)
        #debug_1D(gpu=True, trial=trial, mode=mode)

