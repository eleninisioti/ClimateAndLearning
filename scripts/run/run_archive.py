import sys
import os

from jz_utils import run_exp, run_batch
import numpy as np
import datetime


def parametric_Grove(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    top_dir = "Maslin/debug/parametric/" + project + "_Grove/"

    experiments = []
    param_names = ["--project", "--env_type", "--num_gens", "--num_trials", "--selection_type",
                   "--mutate_mutate_rate", "--genome_type", "--extinctions", "--factor_time_abrupt",
                   "--factor_time_steady", "--num_niches", "--only_climate", "--climate_mean_init"]
    factor_time_abrupt_values = [1, 7]
    factor_time_steady_values = [5]
    env_type = "change"
    num_gens = 1300
    selection_types = ["capacity-fitness", "limited-capacityv2", "FP-Grove"]
    mutate_mutate_rate = 0.001
    genome_types = ["1D", "1D_mutate", "1D_mutate_fixed"]
    extinctions = [1]
    num_niches_values = [1, 10, 50, 100]
    climate_only = 0
    climate_mean_init = 0.2

    for num_niches in num_niches_values:
        for factor_time_abrupt in factor_time_abrupt_values:
            for factor_time_steady in factor_time_steady_values:
                for genome_type in genome_types:
                    for selection_type in selection_types:
                        for extinction in extinctions:

                            project = top_dir + "selection_" + selection_type + "genome_" + genome_type + \
                                      "extinctions_"\
                                      + \
                                      str(extinction) + "_scale_abrupt_" + str(factor_time_abrupt) + "_scale_steady_"\
                                      + str(factor_time_steady )+ "_num_niches_" + \
                                      str(num_niches)
                            new_exp = [project, env_type, num_gens, trial, selection_type, mutate_mutate_rate,
                                       genome_type, extinction, factor_time_abrupt,
                                       factor_time_steady,num_niches, climate_only, climate_mean_init]
                            experiments.append(new_exp)
                            if mode == "local":
                                command = "python simulate.py "
                                for idx, el in enumerate(param_names):
                                    command += el + " " + str(new_exp[idx]) + " "
                                command += "&"
                                print(command)
                                os.system("bash -c '{}'".format(command))

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
    top_dir = "Maslin/debug/parametric/" + project + "/"

    experiments = []

    param_names = ["--project", "--env_type","--num_gens", "--num_trials", "--selection_type",
                   "--mutate_mutate_rate", "--genome_type", "--extinctions", "--factor_time_abrupt",
                   "--factor_time_steady", "--num_niches", "--only_climate", "--factor_time_variable",
                   "--var_SD", "--climate_mean_init"]
    factor_time_abrupt_values = [7]
    factor_time_steady_values = [5]
    factor_time_variable_values = [3]
    var_freq_values = np.arange(10, 100, 30)
    var_SD = 0.2
    env_type = "combined"
    num_gens = 1500
    survival_types = ["FP-Grove", "capacity-fitness", "limited-capacityv2"]
    mutate_rate = 0.001
    genome_types = ["1D", "1D_mutate", "1D_mutate_fixed"]
    extinctions = [1]
    num_niches_values = [1, 10, 50, 100]
    num_niches_values = [1,100]

    climate_only = 0
    climate_mean_init = 0.2


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
                                              "_var_freq_"+ str(var_freq ) + \
                                              "_var_SD_"+ str(var_SD)
                                    new_exp = [project, env_type, num_gens, trial, survival_type, mutate_rate, genome_type,
                                               extinction, factor_time_abrupt, factor_time_steady,num_niches,
                                               climate_only, factor_time_variable, var_freq, var_SD, climate_mean_init]
                                    experiments.append(new_exp)
                                    if mode == "local":
                                        command = "python simulate.py "
                                        for idx, el in enumerate(param_names):
                                            command += el + " " + str(new_exp[idx]) + " "
                                        command += "&"
                                        print(command)
                                        os.system("bash -c '{}'".format(command))


    if mode == "server":
        run_batch(
            experiments,
            param_names,
            long_run=long_run,
            gpu=gpu,
        )


def survival_s2g2_A4(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    if mode == "local":
        top_dir = "papers/gecco/parametric_sin/" + project + "/"
    else:
        top_dir = "/gpfsscratch/rech/imi/utw61ti/ClimateAndLearning_log/projects/survival_s2g2_A4/" + project + "/"
    if not os.path.exists(top_dir):
        os.makedirs(top_dir)
    experiments = []

    param_names = ["--project", "--env_type","--num_gens", "--num_trials", "--selection_type",
                   "--mutate_mutate_rate", "--genome_type", "--extinctions",  "--num_niches",
                   "--only_climate",  "--climate_mean_init", "--amplitude", "--period"]
    amplitudes = [4]
    env_type = "sin"
    num_gens = 1500
    survival_types = ["capacity-fitness"]
    mutate_mutate_rate = 0.001
    genome_types = ["1D_mutate"]
    extinctions = [1]
    num_niches_values = [1,10,100]
    climate_only = 0
    climate_mean_init = 0.2
    periods = [int(num_gens),int(num_gens/2), int(num_gens/8), int(num_gens/16), int(num_gens/32)]
    num_gens = num_gens # make sure we have at least 3 cycles

    for num_niches in num_niches_values:
        for amplitude in amplitudes:
            for period in periods:
                for genome_type in genome_types:
                    for survival_type in survival_types:
                        for extinction in extinctions:
                            project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                      str(extinction) + "_num_niches_" + \
                                      str(num_niches) + "_amplitude_" + str(amplitude) + "_period_" + str(period)
                            new_exp = [project, env_type, num_gens, trial, survival_type, mutate_mutate_rate,
                                       genome_type, extinction, num_niches, climate_only, climate_mean_init,
                                       amplitude, period]
                            experiments.append(new_exp)
                            if mode == "local":
                                command = "python simulate.py "
                                for idx, el in enumerate(param_names):
                                    command += el + " " + str(new_exp[idx]) + " "
                                command += "&"
                                print(command)
                                os.system("bash -c '{}'".format(command))

    if mode == "server":
        run_batch(
            experiments,
            param_names,
            long_run=long_run,
            gpu=gpu,
        )


def survival_s2g2_N100(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    if mode == "local":
        top_dir = "papers/gecco/parametric_sin/" + project + "/"
    else:
        top_dir = "/gpfsscratch/rech/imi/utw61ti/ClimateAndLearning_log/projects/survival_s2g2_N100/" + project + "/"
    if not os.path.exists(top_dir):
        os.makedirs(top_dir)
    experiments = []

    param_names = ["--project", "--env_type","--num_gens", "--num_trials", "--selection_type",
                   "--mutate_mutate_rate", "--genome_type", "--extinctions",  "--num_niches",
                   "--only_climate",  "--climate_mean_init", "--amplitude", "--period"]
    amplitudes = [1,8]
    env_type = "sin"
    num_gens = 1500
    survival_types = ["capacity-fitness"]
    mutate_mutate_rate = 0.001
    genome_types = ["1D_mutate"]
    extinctions = [1]
    num_niches_values = [100]
    climate_only = 0
    climate_mean_init = 0.2
    periods = [int(num_gens),int(num_gens/2), int(num_gens/8), int(num_gens/16), int(num_gens/32)]
    num_gens = num_gens # make sure we have at least 3 cycles

    for num_niches in num_niches_values:
        for amplitude in amplitudes:
            for period in periods:
                for genome_type in genome_types:
                    for survival_type in survival_types:
                        for extinction in extinctions:
                            project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                      str(extinction) + "_num_niches_" + \
                                      str(num_niches) + "_amplitude_" + str(amplitude) + "_period_" + str(period)
                            new_exp = [project, env_type, num_gens, trial, survival_type, mutate_mutate_rate,
                                       genome_type, extinction, num_niches, climate_only, climate_mean_init,
                                       amplitude, period]
                            experiments.append(new_exp)
                            if mode == "local":
                                command = "python simulate.py "
                                for idx, el in enumerate(param_names):
                                    command += el + " " + str(new_exp[idx]) + " "
                                command += "&"
                                print(command)
                                os.system("bash -c '{}'".format(command))

    if mode == "server":
        run_batch(
            experiments,
            param_names,
            long_run=long_run,
            gpu=gpu,
        )


def parametric_noisy(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    top_dir = "papers/gecco/parametric_noisy/" + project + "/"

    experiments = []

    param_names = ["--project", "--env_type","--num_gens", "--num_trials", "--selection_type",
                   "--mutate_mutate_rate", "--genome_type", "--extinctions",  "--num_niches",
                   "--only_climate",  "--climate_mean_init", "--noise_std"]
    noise_stds = [0.05, 0.2, 0.4, 0.8, 1.6]
    env_type = "noisy"
    num_gens = 1500
    survival_types = ["FP-Grove", "capacity-fitness", "limited-capacity"]
    mutate_mutate_rate = 0.001
    genome_types = ["1D", "1D_mutate", "1D_mutate_fixed"]
    extinctions = [1]
    num_niches_values = [1, 5, 10, 40, 100]
    climate_only = 1
    climate_mean_init_values = [0.2, 0.5, 1, 2, 4, 8]

    for num_niches in num_niches_values:
        for noise_std in noise_stds:
            for climate_mean_init in climate_mean_init_values:
                for genome_type in genome_types:
                    for survival_type in survival_types:
                        for extinction in extinctions:
                            project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                      str(extinction) + "_num_niches_" + \
                                      str(num_niches) + "_climate_" + str(climate_mean_init) +\
                                      "_noisestd_" + str(noise_std)
                            new_exp = [project, env_type, num_gens, trial, survival_type, mutate_mutate_rate,
                                       genome_type, extinction, num_niches, climate_only, climate_mean_init,
                                       noise_std]
                            experiments.append(new_exp)
                            if mode == "local":
                                command = "python simulate.py "
                                for idx, el in enumerate(param_names):
                                    command += el + " " + str(new_exp[idx]) + " "
                                command += "&"
                                print(command)
                                os.system("bash -c '{}'".format(command))

    if mode == "server":
        run_batch(
            experiments,
            param_names,
            long_run=long_run,
            gpu=gpu,
        )

def sigma_stable_parta(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    if mode == "local":
        top_dir = "papers/gecco/parametric_stable/" + project + "/"
    else:
        top_dir = "/gpfsscratch/rech/imi/utw61ti/ClimateAndLearning_log/projects/sigma_stable_parta/" + project + "/"
    if not os.path.exists(top_dir):
        os.makedirs(top_dir)
    experiments = []

    param_names = ["--project", "--env_type","--num_gens", "--num_trials", "--selection_type",
                   "--mutate_mutate_rate", "--genome_type", "--extinctions",  "--num_niches",
                   "--only_climate",  "--climate_mean_init"]
    env_type = "stable"
    num_gens = 1000
    survival_types = ["limited-capacity"]
    mutate_mutate_rate = 0.001
    genome_types = ["1D_mutate"]
    extinctions = [1]
    num_niches_values = [1,10,100]
    climate_only = 0
    climate_mean_init_values = [8,16]
    num_gens = num_gens # make sure we have at least 3 cycles

    for num_niches in num_niches_values:
        for climate_mean_init in climate_mean_init_values:
                for genome_type in genome_types:
                    for survival_type in survival_types:
                        for extinction in extinctions:
                            project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                      str(extinction) + "_num_niches_" + \
                                      str(num_niches) + "_climate_" + str(climate_mean_init)
                            new_exp = [project, env_type, num_gens, trial, survival_type, mutate_mutate_rate,
                                       genome_type, extinction, num_niches, climate_only, climate_mean_init]
                            experiments.append(new_exp)
                            if mode == "local":
                                command = "python simulate.py "
                                for idx, el in enumerate(param_names):
                                    command += el + " " + str(new_exp[idx]) + " "
                                command += "&"
                                print(command)
                                os.system("bash -c '{}'".format(command))

    if mode == "server":
        run_batch(
            experiments,
            param_names,
            long_run=long_run,
            gpu=gpu,
        )

def sigma_stable_partb(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    if mode == "local":
        top_dir = "papers/gecco/parametric_stable/" + project + "/"
    else:
        top_dir = "/gpfsscratch/rech/imi/utw61ti/ClimateAndLearning_log/projects/sigma_stable_partb/" + project + "/"
    if not os.path.exists(top_dir):
        os.makedirs(top_dir)
    experiments = []

    param_names = ["--project", "--env_type","--num_gens", "--num_trials", "--selection_type",
                   "--mutate_mutate_rate", "--genome_type", "--extinctions",  "--num_niches",
                   "--only_climate",  "--climate_mean_init"]
    env_type = "stable"
    num_gens = 1000
    survival_types = ["limited-capacity"]
    mutate_mutate_rate = 0.001
    genome_types = ["1D_mutate"]
    extinctions = [1]
    num_niches_values = [5,40]
    climate_only = 0
    climate_mean_init_values = [0.2, 2]
    num_gens = num_gens # make sure we have at least 3 cycles

    for num_niches in num_niches_values:
        for climate_mean_init in climate_mean_init_values:
                for genome_type in genome_types:
                    for survival_type in survival_types:
                        for extinction in extinctions:
                            project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                      str(extinction) + "_num_niches_" + \
                                      str(num_niches) + "_climate_" + str(climate_mean_init)
                            new_exp = [project, env_type, num_gens, trial, survival_type, mutate_mutate_rate,
                                       genome_type, extinction, num_niches, climate_only, climate_mean_init]
                            experiments.append(new_exp)
                            if mode == "local":
                                command = "python simulate.py "
                                for idx, el in enumerate(param_names):
                                    command += el + " " + str(new_exp[idx]) + " "
                                command += "&"
                                print(command)
                                os.system("bash -c '{}'".format(command))

    if mode == "server":
        run_batch(
            experiments,
            param_names,
            long_run=long_run,
            gpu=gpu,
        )


def missing_s1_N100(gpu, trial,  mode, long_run=False):
    #var_freq_values = np.arange(10, 100, 20)
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    if mode == "local":
        top_dir = "papers/gecco/parametric_stable/" + project + "/"
    else:
        top_dir = "/gpfsscratch/rech/imi/utw61ti/ClimateAndLearning_log/projects/missing_s1_100/" + project + "/"
    if not os.path.exists(top_dir):
        os.makedirs(top_dir)
    experiments = []

    param_names = ["--project", "--env_type","--num_gens", "--num_trials", "--selection_type",
                   "--mutate_mutate_rate", "--genome_type", "--extinctions",  "--num_niches",
                   "--only_climate",  "--climate_mean_init"]
    env_type = "stable"
    num_gens = 1000
    survival_types = ["limited-capacity"]
    mutate_mutate_rate = 0.001
    genome_types = ["1D_mutate"]
    extinctions = [1]
    num_niches_values = [100]
    climate_only = 0
    climate_mean_init_values = [4]
    num_gens = num_gens # make sure we have at least 3 cycles

    for num_niches in num_niches_values:
        for climate_mean_init in climate_mean_init_values:
                for genome_type in genome_types:
                    for survival_type in survival_types:
                        for extinction in extinctions:
                            project = top_dir + "survival_" + survival_type + "genome_" + genome_type + "extinctions_" + \
                                      str(extinction) + "_num_niches_" + \
                                      str(num_niches) + "_climate_" + str(climate_mean_init)
                            new_exp = [project, env_type, num_gens, trial, survival_type, mutate_mutate_rate,
                                       genome_type, extinction, num_niches, climate_only, climate_mean_init]
                            experiments.append(new_exp)
                            if mode == "local":
                                command = "python simulate.py "
                                for idx, el in enumerate(param_names):
                                    command += el + " " + str(new_exp[idx]) + " "
                                command += "&"
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
    for trial in range(1, trials+1):
        missing_s1_N100(gpu=True, trial=trial, mode=mode, long_run=True)
        #sigma_stable_parta(gpu=True, trial=trial, mode=mode, long_run=True)
        #sigma_stable_partb(gpu=True, trial=trial, mode=mode, long_run=True)
        #survival_s2g2_N100(gpu=True, trial=trial, mode=mode, long_run=True)
        #survival_s2g2_A4(gpu=True, trial=trial, mode=mode, long_run=True)
        #parametric_sin(gpu=True, trial=trial, mode=mode, long_run=True)
        #fig_sigma_constant(gpu=True, trial=trial, mode=mode, long_run=False)
        #parametric_noisy(gpu=True, trial=trial, mode=mode, long_run=True)


