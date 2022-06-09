""" This script can be used to rerun all simulations in the paper.
"""
import sys
import os
import datetime
import numpy as np


# ----- helper functions ------
def exec_command(config):
    """ Composes a command based on configuration to run as command-line.

    Parameters
    ----------
    config: dict
        input flags for configuration a simulation

    """
    command = "python simulate.py "
    for flag, value in config.items():
        command += flag + " " + str(value) + " "

    os.system("bash -c '{}'".format(command))


def setup_dir():
    """ Set up the top directory for this batch of experiments.
    """
    now = datetime.datetime.now()
    project = str(now.day) + "_" + str(now.month) + "_" + str(now.year)

    top_dir = "../projects/paper/" + project
    if not os.path.exists(top_dir):
        os.makedirs(top_dir)

    if not os.path.exists(top_dir):
        os.makedirs(top_dir)

    return top_dir


# --------------------------------------

def fig2():
    """ Reproduce simulations for Figure 2
    """

    top_dir = setup_dir() + "_fig2/"

    flags = ["--project",
             "--env_type",
             "--num_gens",
             "--trial",
             "--selection_type",
             "--genome_type",
             "--num_niches",
             "--climate_mean_init"]

    env_type = "stable"
    num_gens = 500
    selection_type = "NF"
    genome_type = "evolv"
    num_niches_values = [1, 10, 50, 100]
    climate_mean_init_values = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 8]

    for N in num_niches_values:
        for climate_mean_init in climate_mean_init_values:
            # unique directory for project
            project = top_dir + "S_" + selection_type + "_G_" + genome_type + "_N_" + \
                      str(N) + "_climate_" + str(climate_mean_init)

            # compose command to run as command-line
            values = [project, env_type, num_gens, trial, selection_type, genome_type, N, climate_mean_init]
            config = dict(zip(flags, values))
            exec_command(config)


def fig3():
    """ Reproduce simulations for Figure 3
    """
    top_dir = setup_dir() + "_fig3/"

    flags = ["--project",
             "--env_type",
             "--num_gens",
             "--trial",
             "--selection_type",
             "--genome_type",
             "--num_niches",
             "--climate_mean_init"]

    env_type = "stable"
    num_gens = 1500
    genome_type = "evolv"
    num_niches = 100
    selection_types = ["N", "NF", "F"]
    climate_mean_init_values = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 8]

    for climate_mean_init in climate_mean_init_values:
        for S in selection_types:
            project = top_dir + "S_" + S + "_G_" + genome_type + "_N_" + \
                      str(num_niches) + "_climate_" + str(climate_mean_init)
            values = [project, env_type, num_gens, trial, S, genome_type, num_niches, climate_mean_init]
            config = dict(zip(flags, values))
            exec_command(config)


def fig4():
    """ Reproduce simulations for Figure 4
    """
    top_dir = setup_dir() + "_fig4/"

    flags = ["--project",
             "--env_type",
             "--num_gens",
             "--trial",
             "--selection_type",
             "--genome_type",
             "--num_niches",
             "--climate_mean_init"]

    env_type = "stable"
    num_gens = 1500
    selection_types = ["NF", "N"]
    genome_types = ["evolv", "no-evolv"]
    num_niches = 100
    climate_mean_init_values = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 8]

    for climate_mean_init in climate_mean_init_values:
        for G in genome_types:
            for S in selection_types:
                project = top_dir + "S_" + S + "_G_" + G + "_N_" + \
                          str(num_niches) + "_climate_" + str(climate_mean_init)
                values = [project, env_type, num_gens, trial, S, G, num_niches, climate_mean_init]
                config = dict(zip(flags, values))
                exec_command(config)


def fig5():
    """ Reproduce simulations for Figure 5
    """
    top_dir = setup_dir() + "_fig5/"

    flags = ["--project",
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
    selection_type = "NF"
    genome_type = "evolv"
    climate_mean_init = 0.2
    num_niches_values = [1, 10, 100]
    amplitude = [4]
    period_values = [int(num_gens), int(num_gens / 2), int(num_gens / 8), int(num_gens / 16), int(num_gens / 32)]

    for period in period_values:
        for num_niches in num_niches_values:
            project = top_dir + "S_" + selection_type + "_G_" + genome_type + "_N_" + str(num_niches) + \
                      "_climate_" + str(climate_mean_init) + "_T_" + str(period) + "_A_" + str(
                amplitude)
            values = [project, env_type, num_gens, trial, selection_type, genome_type, num_niches,
                      climate_mean_init, amplitude, period]
            config = dict(zip(flags, values))
            exec_command(config)


def fig6():
    """ Reproduce simulations for Figure 6
    """
    top_dir = setup_dir() + "_fig6/"

    flags = ["--project",
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
    selection_type = "NF"
    genome_type = "evolv"
    num_niches = 100
    climate_mean_init = 0.2
    amplitude_values = [0.2, 1, 8]
    period_values = [int(num_gens), int(num_gens / 2), int(num_gens / 8), int(num_gens / 16), int(num_gens / 32)]

    for period in period_values:
        for amplitude in amplitude_values:
            project = top_dir + "S_" + selection_type + "_G_" + genome_type + "_N_" + str(num_niches) + \
                      "_climate_" + str(climate_mean_init) + "_T_" + str(period) + "_A_" + str(
                amplitude)
            values = [project, env_type, num_gens, trial, selection_type, genome_type, num_niches,
                      climate_mean_init, amplitude, period]
            config = dict(zip(flags, values))
            exec_command(config)


def fig7():
    """ Reproduce simulations for Figure 7
    """
    top_dir = setup_dir() + "_fig7/"

    flags = ["--project",
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
    selection_type = "NF"
    genome_type = "evolv"
    num_niches = 100
    amplitude = 8
    climate_mean_init = 0.2
    period = 750

    project = top_dir + "S_" + selection_type + "_G_" + genome_type + "_N_" + str(num_niches) + \
              "_climate_" + str(climate_mean_init) + "_T_" + str(period) + "_A_" + str(
        amplitude)
    values = [project, env_type, num_gens, trial, selection_type, genome_type, num_niches,
              climate_mean_init, amplitude, period]
    config = dict(zip(flags, values))
    exec_command(config)


def fig8():
    """ Reproduce simulations for Figure 8
    """
    top_dir = setup_dir() + "_fig8/"

    flags = ["--project",
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
    genome_type = "evolv"
    num_niches = 100
    amplitude = 0.2
    climate_mean_init = 0.2
    period = 46

    for selection in selection_types:
        project = top_dir + "S_" + selection + "_G_" + genome_type + "_N_" + str(num_niches) + \
                  "_climate_" + str(climate_mean_init) + "_T_" + str(period) + "_A_" + str(
            amplitude)
        values = [project, env_type, num_gens, trial, selection, genome_type, num_niches,
                  climate_mean_init, amplitude, period]
        config = dict(zip(flags, values))
        exec_command(config)


def fig9():
    """ Reproduce simulations for Figure 9
    """
    top_dir = setup_dir() + "_fig9/"

    flags = ["--project",
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
    selection_type = "NF"
    genome_type = "evolv"
    num_niches = 100
    climate_mean_init = 2
    noise_std_values = np.arange(0.05, 0.82, 0.1)

    for noise_std in noise_std_values:
        project = top_dir + "selection_" + selection_type + "_G_" + genome_type + "_N_" + str(num_niches) + \
                  "_climate_" \
                  + \
                  str(climate_mean_init) + "_noise_" + str(noise_std)
        values = [project, env_type, num_gens, trial, selection_type,
                  genome_type, num_niches, climate_mean_init, noise_std]
        config = dict(zip(flags, values))
        exec_command(config)


def fig10():
    """ Reproduce simulations for Figure 10.
    """
    top_dir = setup_dir() + "_fig10/"

    flags = ["--project",
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
    genome_type = "evolv"
    num_niches = 40
    noise_std = 0.2
    climate_mean_init = 2
    selection_types = ["NF", "N", "F"]

    for S in selection_types:
        project = top_dir + "selection_" + S + "_G_" + genome_type + "_N_" + str(num_niches) + "_climate_" + \
                  str(climate_mean_init) + "_noise_" + str(noise_std)
        values = [project, env_type, num_gens, trial, S, genome_type, num_niches, climate_mean_init, noise_std]
        config = dict(zip(flags, values))
        exec_command(config)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("You need to provide the number of trials.")
    else:
        trials = int(sys.argv[1])  # number of independent trials

        for trial in range(trials):
            fig2()
            fig3()
            fig4()
            fig5()
            fig6()
            fig7()
            fig8()
            fig9()
            fig10()
