import sys
import os
from jz_utils import run_batch
import datetime


def example(trial, long_run):
    """
    RUns an example simulation for a sinusoid environment under NF-selection and evolvable genome, 10 niches.

    """
    top_dir = setup_dir()
    experiments = []

    param_names = ["--project",
                   "--env_type",
                   "--num_gens",
                   "--num_trials",
                   "--selection_type",
                   "--genome_type",
                   "--num_niches",
                   "--climate_mean_init"]
    # ----- configuration -----
    env_type = "stable"
    num_gens = 1000
    S = "NF"
    G = "evolv"
    N = 10
    climate_mean_init = 2
    # ------------------------

    project = top_dir + "S_" + S + "_G_" + G + "_N_" + str(N) + "_climate_" + str(climate_mean_init)
    new_exp = [project, env_type, num_gens, trial, S, G, N, climate_mean_init]
    experiments.append(new_exp)
    if mode == "local":
        command = "python simulate.py "
        for idx, el in enumerate(param_names):
            command += el + " " + str(new_exp[idx]) + " "
        # command += "&" # uncomment to run all experiments simultaneously
        print(command)
        os.system("bash -c '{}'".format(command))

    elif mode == "server":
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
        top_dir = "projects/papers/gecco/parametric_stable/" + project + "/"
    elif mode == "server":
        top_dir = "/gpfsscratch/rech/imi/utw61ti/ClimateAndLearning_log/projects/" + project + "/"

    if not os.path.exists(top_dir):
        os.makedirs(top_dir)

    return top_dir


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("You need to provide the number of trials.")
    else:
        trials = int(sys.argv[1])  # number of independent trials
        mode = "local"  # this should be server for running jz experiments

        for trial in range(1, trials + 1):
            example(trial, long_run=False)
