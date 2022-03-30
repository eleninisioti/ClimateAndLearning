""" This script can be used to rerun all simulations in the paper.
"""

import sys
import os
from jz_utils import  run_batch
import datetime


def stable_sigma(trial, long_run):
    "Reproduce experiments with stable environment"
    top_dir = setup_dir() + "_stable_sigma/"
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
    num_gens = 500
    selection_types = ["NF"]
    genome_types = ["evolv"]
    num_niches_values = [1,5,10,50,100]
    climate_mean_init_values = [0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 4, 6, 8]

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


def stable_selection(trial, long_run):
    "Reproduce experiments with stable environment"
    top_dir = setup_dir() + "_stable_sigma/"
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
    num_gens = 500
    selection_types = ["NF", "F", "N"]
    genome_types = ["evolv"]
    num_niches_values = [100]
    climate_mean_init_values = [0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 4, 6, 8]

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
    elif mode == "server":
        top_dir = "/gpfsscratch/rech/imi/utw61ti/ClimateAndLearning_log/projects/" + project

    if not os.path.exists(top_dir):
        os.makedirs(top_dir)

    return top_dir

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("You need to provide the number of trials and mode.")
    else:
        trials = int(sys.argv[1]) # number of independent trials
        mode = sys.argv[2] # this should be server for running jz experiments

        for trial in range(1, trials+1):
            stable_sigma(trial, long_run=False)
            stable_selection(trial, long_run=False)

