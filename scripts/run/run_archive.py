
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

    #command += " &"

    print(command)

    #os.system("bash -c '{}'".format(command))
    quit()


def create_jzscript(config):
    command = "python simulate.py "
    for flag, value in config.items():
        command += flag + " " + str(value) + " "
    now = datetime.datetime.now()
    scripts_dir =  "../jz_scripts/" + str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    if not os.path.exists(scripts_dir):
        os.makedirs(scripts_dir)
    if config["--env_type"] == "stable":
        climate_string = "_climate_mean_init_" + str(config["--climate_mean_init"])
    elif config["--env_type"] == "sin":
        climate_string = "_amplitude_" + str(config["--amplitude"]) + "_periodi_" + str(config["--period"])

    elif config["--env_type"] == "noisy":
        climate_string = "_noise_" + str(config["--noise_std"])

    script_path = scripts_dir + "/climate_" + config["--env_type"] + "_select_" + config["--selection_type"] +\
                  "_genome_" + config["--genome_type"] + climate_string + "_trial_" + str(config["--trial"])   +".sh"
    with open(script_path, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -J fully\n")
        # fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH -t 20:00:00\n")
        fh.writelines("#SBATCH -N 1\n")
        fh.writelines("#SBATCH --ntasks-per-node=10\n")
        scratch_dir = "/scratch/enisioti/climate_log/jz_logs"

        fh.writelines("#SBATCH --output=" + scratch_dir + "/%j.out\n")
        fh.writelines("#SBATCH --error=" + scratch_dir + "/%j.err\n")
        #fh.writelines("module load pytorch-gpu/py3/1.7.1\n")

        fh.writelines(command)


def setup_dir(project="", mode = "local"):
    """ Set up the top directory for this batch of experiments.
    """
    now = datetime.datetime.now()
    date = str(now.day) + "_" + str(now.month) + "_" + str(now.year)

    if mode == "local":
        top_dir = "../projects/" + project + "/" + date
    elif mode == "server":
        top_dir = "/scratch/enisioti/climate_log/projects/" + project + "/" + date
    if not os.path.exists(top_dir):
        os.makedirs(top_dir)

    if not os.path.exists(top_dir):
        os.makedirs(top_dir)

    return top_dir


def fig5_xland():
    """ Reproduce simulations for Figure 5
    """
    top_dir = setup_dir() + "_fig5_xland/"

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
    selection_type = "xland"
    genome_type = "evolv"
    climate_mean_init = 0.2
    num_niches_values = [1, 10, 100]
    amplitude = 4
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


def fig6_xland():
    """ Reproduce simulations for Figure 6
    """
    top_dir = setup_dir() + "_fig6_xland/"

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
    selection_type = "xland"
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

def fig9_xland():
    """ Reproduce simulations for Figure 9
    """
    top_dir = setup_dir() + "_fig9_xland/"

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
    selection_type = "xland"
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


def xland():
    """ Reproduce simulations for Figure 9
    """
    """ Reproduce simulations for Figure 8
    """
    top_dir = setup_dir() + "_xland/"

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
    selection_type = "xland"
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


def niche_construction_stable(mode):
    top_dir = setup_dir(project="niche_construction", mode=mode) + "/stable/"

    flags = ["--project",
             "--env_type",
             "--num_gens",
             "--trial",
             "--selection_type",
             "--genome_type",
             "--num_niches",
             "--climate_mean_init"]

    env_type = "stable"
    num_gens = 1000
    genome_types = [" niche-construction"]
    num_niches = 100
    selection_types = [ "NF"]
    climate_mean_init_values = [0.2, 0.4, 0.6, 0.8, 1, 2, 4, 8]
    climate_mean_init_values = [0.6]


    for genome_type in genome_types:

        for climate_mean_init in climate_mean_init_values:
            for S in selection_types:

                if not (genome_type=="evolv" and S=="F"):
                    project = top_dir + "S_" + S + "_G_" + genome_type + "_N_" + \
                              str(num_niches) + "_climate_" + str(climate_mean_init)
                    values = [project, env_type, num_gens, trial, S, genome_type, num_niches, climate_mean_init]
                    config = dict(zip(flags, values))
                    if mode == "local":
                        exec_command(config)
                    elif mode == "server":
                        create_jzscript(config)

def niche_construction_periodic(mode):
    top_dir = setup_dir(project="niche_construction", mode=mode) + "/periodic/"

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
    num_gens = 1000
    num_niches = 100
    genome_types = ["niche-construction-v2", "niche-construction"]

    selection_types = ["NF", "F"]
    climate_mean_init = 0.2
    amplitude_values = [0.2, 1, 4, 8]
    amplitude_values = [4]
    period_values = [int(num_gens), int(num_gens / 2), int(num_gens / 8), int(num_gens / 16), int(num_gens / 32)]
    period_values = [int(num_gens/2)]



    for period in period_values:
        for amplitude in amplitude_values:

            for genome_type in genome_types:

                for selection in selection_types:
                    if not (genome_type == "evolv" and selection == "F"):
                        project = top_dir + "S_" + selection + "_G_" + genome_type + "_N_" + str(num_niches) + \
                                  "_climate_" + str(climate_mean_init) + "_T_" + str(period) + "_A_" + str(
                            amplitude)
                        values = [project, env_type, num_gens, trial, selection, genome_type, num_niches,
                                  climate_mean_init, amplitude, period]
                        config = dict(zip(flags, values))
                        if mode == "local":
                            exec_command(config)
                        elif mode == "server":
                            create_jzscript(config)

def niche_construction_noisy(mode):
    top_dir = setup_dir(project="niche_construction", mode=mode) + "/noisy/"

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
    num_niches = 100
    selection_types = ["NF", "F"]
    genome_types = ["niche-construction-v2", "niche-construction"]
    climate_mean_init = 2
    noise_std_values = np.arange(0.05, 0.82, 0.1)
    noise_std_values = [0.2]

    for noise_std in noise_std_values:

        for genome_type in genome_types:
            for S in selection_types:
                if not (genome_type == "evolv" and S == "F"):
                    project = top_dir + "selection_" + S + "_G_" + genome_type + "_N_" + str(num_niches) + "_climate_" + \
                              str(climate_mean_init) + "_noise_" + str(noise_std)
                    values = [project, env_type, num_gens, trial, S, genome_type, num_niches, climate_mean_init, noise_std]
                    config = dict(zip(flags, values))
                    if mode == "local":
                        exec_command(config)
                    elif mode == "server":
                        create_jzscript(config)


def niche_construction_noisy_parametric(mode):
    top_dir = setup_dir(project="niche_construction", mode=mode) + "/noisy/"

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
    genome_type = "niche-construction"
    num_niches = 100
    noise_std_values = np.arange(0.05, 1.82, 0.2)
    climate_mean_init = 2
    selection_types = ["NF", "N"]
    #selection_types = ["N"]
    for noise_std in noise_std_values:
        for S in selection_types:
            project = top_dir + "selection_" + S + "_G_" + genome_type + "_N_" + str(num_niches) + "_climate_" + \
                      str(climate_mean_init) + "_noise_" + str(noise_std)
            values = [project, env_type, num_gens, trial, S, genome_type, num_niches, climate_mean_init, noise_std]
            config = dict(zip(flags, values))
            if mode == "local":
                exec_command(config)
            elif mode == "server":
                create_jzscript(config)


def manim_fig8(mode):
    top_dir = setup_dir(mode=mode) + "/manim_fig8/"

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
    selection_types = ["NF", "F"]
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
        if mode == "local":
            exec_command(config)
        elif mode == "server":
            create_jzscript(config)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("You need to provide the number of trials and mode.")
    else:
        trials = int(sys.argv[1])  # number of independent trials
        mode = sys.argv[2]

        for trial in range(trials):
            niche_construction_stable(mode)
            #niche_construction_periodic(mode)
            #niche_construction_noisy(mode)
            #niche_construction_noisy_parametric(mode)
            #manim_fig8(mode)