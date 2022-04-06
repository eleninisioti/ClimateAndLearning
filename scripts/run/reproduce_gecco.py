""" This script can be used to rerun all simulations in the paper.
"""
import sys
import os
from jz_utils import  run_batch
import datetime
import numpy as np

def stable_sigma(trial, long_run):
    "Reproduce experiments with stable environment"
    top_dir = setup_dir() + "_sigma_0.2/"
    experiments = []

    param_names = ["--project",
                   "--env_type",
                   "--num_gens",
                   "--trial",
                   "--selection_type",
                   "--genome_type",
                   "--num_niches",
                   "--climate_mean_init"]
    env_type = "stable"
    num_gens = 300
    selection_types = ["NF", "no-evolv"]
    genome_types = ["evolv"]
    num_niches_values = [100]
    climate_mean_init_values = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 8]

    for N in num_niches_values:
        for climate_mean_init in climate_mean_init_values:
            for G in genome_types:
                for S in selection_types:
                        project = top_dir + "S_" + S + "_G_" + G + "_N_" + \
                                  str(N) + "_climate_" + str(climate_mean_init)
                        new_exp = [project, env_type, num_gens, trial, S, G, N, climate_mean_init]
                        experiments.append(new_exp)
                        if mode == "local":
                            command = "python simulate.py "
                            for idx, el in enumerate(param_names):
                                command += el + " " + str(new_exp[idx]) + " "
                            # command += "&" # uncomment to run all experiments simultaneously
                            print(command)
                            quit()
                            os.system("bash -c '{}'".format(command))

    if mode == "server":
        run_batch(experiments, param_names, long_run=long_run, gpu=True)


def stable_selection(trial, long_run):
    "Reproduce experiments with stable environment"
    top_dir = setup_dir() + "_selection/"
    experiments = []

    param_names = ["--project",
                   "--env_type",
                   "--num_gens",
                   "--trial",
                   "--selection_type",
                   "--genome_type",
                   "--num_niches",
                   "--climate_mean_init"]
    env_type = "stable"
    num_gens = 300
    selection_types = ["NF", "F", "N"]
    genome_types = ["evolv"]
    num_niches_values = [100]
    climate_mean_init_values = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 8]


    for N in num_niches_values:
        for climate_mean_init in climate_mean_init_values:
            for G in genome_types:
                for S in selection_types:
                        project = top_dir + "S_" + S + "_G_" + G + "_N_" + \
                                  str(N) + "_climate_" + str(climate_mean_init)
                        new_exp = [project, env_type, num_gens, trial, S, G, N, climate_mean_init]
                        experiments.append(new_exp)
                        if mode == "local":
                            command = "python simulate.py "
                            for idx, el in enumerate(param_names):
                                command += el + " " + str(new_exp[idx]) + " "
                            # command += "&" # uncomment to run all experiments simultaneously
                            print(command)
                            os.system("bash -c '{}'".format(command))
                            quit()

    if mode == "server":
        run_batch(experiments, param_names, long_run=long_run, gpu=True)

def stable_extinct(trial, long_run):
    "Reproduce experiments with stable environment"
    top_dir = setup_dir() + "_extinct/"
    experiments = []

    param_names = ["--project",
                   "--env_type",
                   "--num_gens",
                   "--trial",
                   "--selection_type",
                   "--genome_type",
                   "--num_niches",
                   "--climate_mean_init"]
    env_type = "stable"
    num_gens = 300
    selection_types = ["F"]
    genome_types = ["evolv"]
    num_niches_values = [100]
    climate_mean_init_values = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 8]

    for N in num_niches_values:
        for climate_mean_init in climate_mean_init_values:
            for G in genome_types:
                for S in selection_types:
                        project = top_dir + "S_" + S + "_G_" + G + "_N_" + \
                                  str(N) + "_climate_" + str(climate_mean_init)
                        new_exp = [project, env_type, num_gens, trial, S, G, N, climate_mean_init]
                        experiments.append(new_exp)
                        if mode == "local":
                            command = "python simulate.py "
                            for idx, el in enumerate(param_names):
                                command += el + " " + str(new_exp[idx]) + " "
                            # command += "&" # uncomment to run all experiments simultaneously
                            print(command)
                            quit()

                            os.system("bash -c '{}'".format(command))

    if mode == "server":
        run_batch(experiments, param_names, long_run=long_run, gpu=True)


def stable_diversity(trial, long_run):
    "Reproduce experiments with stable environment"
    top_dir = setup_dir() + "_diversity/"
    experiments = []

    param_names = ["--project",
                   "--env_type",
                   "--num_gens",
                   "--trial",
                   "--selection_type",
                   "--genome_type",
                   "--num_niches",
                   "--climate_mean_init"]
    env_type = "stable"
    num_gens = 300
    selection_types = ["N"]
    genome_types = ["evolv"]
    num_niches_values = [100]
    climate_mean_init_values = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 8]

    for N in num_niches_values:
        for climate_mean_init in climate_mean_init_values:
            for G in genome_types:
                for S in selection_types:
                        project = top_dir + "S_" + S + "_G_" + G + "_N_" + \
                                  str(N) + "_climate_" + str(climate_mean_init)
                        new_exp = [project, env_type, num_gens, trial, S, G, N, climate_mean_init]
                        experiments.append(new_exp)
                        if mode == "local":
                            command = "python simulate.py "
                            for idx, el in enumerate(param_names):
                                command += el + " " + str(new_exp[idx]) + " "
                            # command += "&" # uncomment to run all experiments simultaneously
                            print(command)

                            os.system("bash -c '{}'".format(command))

    if mode == "server":
        run_batch(experiments, param_names, long_run=long_run, gpu=True)



def sin_survival_N100(trial, long_run):
    "Reproduce experiments with noisy environment"
    top_dir = setup_dir() + "_sin_/"
    experiments = []

    param_names = ["--project",
                   "--env_type",
                   "--num_gens",
                   "--trial",
                   "--selection_type",
                   "--genome_type",
                   "--num_niches",
                   "--climate_mean_init",
                   "--amplitude",
                   "--period"]

    env_type = "sin"
    num_gens = 1500
    selection_types = ["NF"]

    genome_types = ["evolv"]
    num_niches_values = [100]
    amplitude_values = [0.2,1,8]
    climate_mean_init = 0.2
    period_values = [int(num_gens), int(num_gens / 2), int(num_gens / 8), int(num_gens / 16), int(num_gens / 32)]
    for period in period_values:
        for amplitude in amplitude_values:
            for num_niches in num_niches_values:
                for genome_type in genome_types:
                    for selection in selection_types:
                            project = top_dir + "S_" + selection + "_G_" + genome_type + "_N_" + str(num_niches) +\
                                      "_climate_" + str(climate_mean_init) + "_T_" + str(period) + "_A_" + str(
                                amplitude)
                            new_exp = [project, env_type, num_gens, trial, selection, genome_type,  num_niches,
                                       climate_mean_init, amplitude,period]
                            experiments.append(new_exp)
                            if mode == "local":
                                command = "python simulate.py "
                                for idx, el in enumerate(param_names):
                                    command += el + " " + str(new_exp[idx]) + " "
                                # command += "&" # uncomment to run all experiments simultaneously
                                print(command)
                                quit()
                                os.system("bash -c '{}'".format(command))

    if mode == "server":

        run_batch(experiments, param_names, long_run=long_run, gpu=True)


def sin_survival_A4(trial, long_run):
    "Reproduce experiments with noisy environment"
    top_dir = setup_dir() + "_sin_/"
    experiments = []

    param_names = ["--project",
                   "--env_type",
                   "--num_gens",
                   "--trial",
                   "--selection_type",
                   "--genome_type",
                   "--num_niches",
                   "--climate_mean_init",
                   "--amplitude",
                   "--period"]

    env_type = "sin"
    num_gens = 1500
    selection_types = ["NF"]
    genome_types = ["evolv"]
    num_niches_values = [1,10,100]
    amplitude_values = [4]
    climate_mean_init = 0.2
    period_values = [int(num_gens), int(num_gens / 2), int(num_gens / 8), int(num_gens / 16), int(num_gens / 32)]
    for period in period_values:
        for amplitude in amplitude_values:
            for num_niches in num_niches_values:
                for genome_type in genome_types:
                    for selection in selection_types:
                            project = top_dir + "S_" + selection + "_G_" + genome_type + "_N_" + str(num_niches) +\
                                      "_climate_" + str(climate_mean_init) + "_T_" + str(period) + "_A_" + str(
                                amplitude)
                            new_exp = [project, env_type, num_gens, trial, selection, genome_type,  num_niches,
                                       climate_mean_init, amplitude,period]
                            experiments.append(new_exp)
                            if mode == "local":
                                command = "python simulate.py "
                                for idx, el in enumerate(param_names):
                                    command += el + " " + str(new_exp[idx]) + " "
                                # command += "&" # uncomment to run all experiments simultaneously
                                print(command)
                                quit()
                                os.system("bash -c '{}'".format(command))

    if mode == "server":

        run_batch(experiments, param_names, long_run=long_run, gpu=True)

def sin_evolution_quick(trial, long_run):
    "Reproduce experiments with noisy environment"
    top_dir = setup_dir() + "_sin_evolution_quick/"
    experiments = []

    param_names = ["--project",
                   "--env_type",
                   "--num_gens",
                   "--trial",
                   "--selection_type",
                   "--genome_type",
                   "--num_niches",
                   "--climate_mean_init",
                   "--amplitude",
                   "--period"]

    env_type = "sin"
    num_gens = 500
    selection_types = ["N", "F"]
    genome_types = ["evolv"]
    num_niches_values = [100]
    amplitude_values = [0.2]
    climate_mean_init = 0.2
    period_values = [46]
    for period in period_values:
        for amplitude in amplitude_values:
            for num_niches in num_niches_values:
                for genome_type in genome_types:
                    for selection in selection_types:
                            project = top_dir + "S_" + selection + "_G_" + genome_type + "_N_" + str(num_niches) +\
                                      "_climate_" + str(climate_mean_init) + "_T_" + str(period) + "_A_" + str(
                                amplitude)
                            new_exp = [project, env_type, num_gens, trial, selection, genome_type,  num_niches,
                                       climate_mean_init, amplitude,period]
                            experiments.append(new_exp)
                            if mode == "local":
                                command = "python simulate.py "
                                for idx, el in enumerate(param_names):
                                    command += el + " " + str(new_exp[idx]) + " "
                                command += "&" # uncomment to run all experiments simultaneously
                                print(command)
                                #os.system("bash -c '{}'".format(command))

    if mode == "server":

        run_batch(experiments, param_names, long_run=long_run, gpu=True)


def sin_evolution_slow(trial, long_run):
    "Reproduce experiments with noisy environment"
    top_dir = setup_dir() + "_sin_evolution_slow/"
    experiments = []

    param_names = ["--project",
                   "--env_type",
                   "--num_gens",
                   "--trial",
                   "--selection_type",
                   "--genome_type",
                   "--num_niches",
                   "--climate_mean_init",
                   "--amplitude",
                   "--period"]

    env_type = "sin"
    num_gens = 1500
    selection_types = ["NF"]

    genome_types = ["evolv"]
    num_niches_values = [100]
    amplitude_values = [8]
    climate_mean_init = 0.2
    period_values = [750]
    for period in period_values:
        for amplitude in amplitude_values:
            for num_niches in num_niches_values:
                for genome_type in genome_types:
                    for selection in selection_types:
                            project = top_dir + "S_" + selection + "_G_" + genome_type + "_N_" + str(num_niches) +\
                                      "_climate_" + str(climate_mean_init) + "_T_" + str(period) + "_A_" + str(
                                amplitude)
                            new_exp = [project, env_type, num_gens, trial, selection, genome_type,  num_niches,
                                       climate_mean_init, amplitude,period]
                            experiments.append(new_exp)
                            if mode == "local":
                                command = "python simulate.py "
                                for idx, el in enumerate(param_names):
                                    command += el + " " + str(new_exp[idx]) + " "
                                command += "&" # uncomment to run all experiments simultaneously
                                print(command)
                                #os.system("bash -c '{}'".format(command))

    if mode == "server":

        run_batch(experiments, param_names, long_run=long_run, gpu=True)

def noisy(trial, long_run=False):
    """Reproduce experiments with noisy environment
    """
    top_dir = setup_dir() + "_noisy_/"
    experiments = []

    param_names = ["--project",
                   "--env_type",
                   "--num_gens",
                   "--trial",
                   "--selection_type",
                   "--genome_type",
                   "--num_niches",
                   "--climate_mean_init",
                   "--noise_std"]
    env_type = "noisy"
    num_gens = 500
    selection_types = ["NF", "N", "F"]
    genome_types = ["evolv"]
    num_niches_values = [40]
    noise_std_values = [0.2]
    climate_mean_init_values = [2]

    for noise_std in noise_std_values:
        for N in num_niches_values:
            for climate_mean_init in climate_mean_init_values:
                for G in genome_types:
                    for S in selection_types:
                            project = top_dir + "selection_" + S + "_G_" + G + "_N_" + str(N) + "_climate_" +\
                                      str(climate_mean_init) + "_noise_" + str(noise_std)
                            new_exp = [project, env_type, num_gens, trial, S,
                                       G,  N, climate_mean_init, noise_std]
                            experiments.append(new_exp)
                            if mode == "local":
                                command = "python simulate.py "
                                for idx, el in enumerate(param_names):
                                    command += el + " " + str(new_exp[idx]) + " "
                                command += "&" # uncomment to run all experiments simultaneously
                                print(command)

                                #os.system("bash -c '{}'".format(command))

    if mode == "server":
        run_batch(experiments, param_names, long_run=long_run, gpu=True)


def noisy_survival(trial, long_run=False):
    """Reproduce experiments with noisy environment
    """
    top_dir = setup_dir() + "_noisy_/"
    experiments = []

    param_names = ["--project",
                   "--env_type",
                   "--num_gens",
                   "--trial",
                   "--selection_type",
                   "--genome_type",
                   "--num_niches",
                   "--climate_mean_init",
                   "--noise_std",
                   "--mean_fitness",
                   "--reproduce_once"]
    env_type = "noisy"
    num_gens = 500
    selection_types = ["NF", "N", "F"]
    genome_types = ["evolv"]
    num_niches_values = [40]
    noise_std_values = np.arange(0.05,0.82, 0.05)
    climate_mean_init_values = [2]

    for noise_std in noise_std_values:
        for N in num_niches_values:
            for climate_mean_init in climate_mean_init_values:
                for G in genome_types:
                    for S in selection_types:
                            project = top_dir + "selection_" + S + "_G_" + G + "_N_" + str(N) + "_climate_" +\
                                      str(climate_mean_init) + "_noise_" + str(noise_std)
                            new_exp = [project, env_type, num_gens, trial, S,
                                       G,  N, climate_mean_init, noise_std]
                            experiments.append(new_exp)
                            if mode == "local":
                                command = "python simulate.py "
                                for idx, el in enumerate(param_names):
                                    command += el + " " + str(new_exp[idx]) + " "
                                # command += "&" # uncomment to run all experiments simultaneously
                                print(command)
                                quit()

                                #os.system("bash -c '{}'".format(command))

    if mode == "server":
        run_batch(experiments, param_names, long_run=long_run, gpu=True)


def setup_dir():
    """ Set up the top directory for this batch of experiments.

    Parameters
    ----------
    mode: str
        Chooe between local and server (for jeanzay)
    """
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)

    if mode == "local":
        top_dir = "../projects/debug/" + project
        if not os.path.exists(top_dir):
            os.makedirs(top_dir)

        if not os.path.exists(top_dir):
            os.makedirs(top_dir)
    elif mode == "server":
        top_dir = project

    return top_dir

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("You need to provide the number of trials and mode.")
    else:
        trials = int(sys.argv[1]) # number of independent trials
        mode = sys.argv[2] # this should be server for running jz experiments

        for trial in range(trials):
            stable_sigma(trial, long_run=False)
            stable_selection(trial, long_run=False)
            stable_diversity(trial, long_run=False)
            stable_extinct(trial, long_run=False)
            sin_survival_A4(trial, long_run=False)
            sin_survival_N100(trial, long_run=False)
            sin_evolution_slow(trial, long_run=False)
            sin_evolution_quick(trial, long_run=False)
            noisy(trial, long_run=False)
            noisy_survival(trial, long_run=False)


