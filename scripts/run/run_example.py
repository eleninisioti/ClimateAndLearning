""" This is an example script that you can you to run simulations.
"""

import sys
import os
sys.path.append(os.getcwd() + "/../scripts/run")
from jz_utils import run_exp, run_batch
import datetime




def parametric(gpu, trial, mode, long_run=False):
    """ Example script of how to run batch experiment"""
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    top_dir = "my_examples/" + project + "_debug/"

    experiments = []
    param_names = ["--project", "--env_type", "--num_gens", "--num_trials", "--selection_type",
                   "--mutate_mutate_rate", "--genome_type", "--extinctions", "--factor_time_abrupt",
                   "--factor_time_steady", "--num_niches", "--only_climate"]
    factor_time_abrupt_values = [7]
    factor_time_steady_values = [5]
    env_type = "change"
    num_gens = 700
    selection_types = ["capacity-fitness"]
    mutate_mutate_rate = 0.001
    genome_types = ["1D", "1D_mutate", "1D_mutate_fixed"]
    extinctions = [1]
    num_niches_values = [1]
    climate_only = 0

    for num_niches in num_niches_values:
        for factor_time_abrupt in factor_time_abrupt_values:
            for factor_time_steady in factor_time_steady_values:
                for genome_type in genome_types:
                    for selection_type in selection_types:
                        for extinction in extinctions:

                            project = top_dir + "selection_" + selection_type + "genome_" + genome_type + \
                                      "extinctions_" + str(extinction) + "_scale_abrupt_" + str(factor_time_abrupt) + \
                                      "_scale_steady_" + str(factor_time_steady) + "_num_niches_" + str(num_niches)
                            new_exp = [project, env_type, num_gens, trial, selection_type, mutate_mutate_rate,
                                       genome_type,
                                       extinction, factor_time_abrupt, factor_time_steady, num_niches, climate_only]
                            experiments.append(new_exp)
                            if mode == "local":
                                command = "python simulate.py "
                                for idx, el in enumerate(param_names):
                                    command += el + " " + str(new_exp[idx]) + " "
                                print(command)
                                os.system("bash -c '{}'".format(command))

    if mode == "server":
        run_batch(experiments,
                  param_names,
                  long_run=long_run,
                  gpu=gpu)


if __name__ == "__main__":
    trials = int(sys.argv[1]) # number independent trials for confidence intervals
    mode = sys.argv[2]  # server for jz experiments, local otherwise
    for trial in range(1, trials + 1):
        parametric(gpu=True, trial=trial, mode=mode)
