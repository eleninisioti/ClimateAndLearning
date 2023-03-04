import numpy as np
import os
import pandas as pd
import yaml
import pickle

# map metrics to labels used for plotting
axes_labels = {"climate_mean_init": "$\\bar{e}$, Mean climate",
               "num_niches": "$N$, Number of niches",
               "period": "$T$, Period of sinusoid",
               "amplitude": "$A$, Amplitude of sinusoid",
               "SD": "$\\bar{\sigma}^*$, Plasticity",
               "R": "$\\bar{r}^*$, Evolvability",
               "Dispersal": "$D^*$, Dispersal",
               "diversity": "$V^*$, Diversity",
               "noise_std": "$\sigma_N$, Standard deviation of noise "}

short_labels = {"num_niches": "$N$",
                "amplitude": "$A_e$",
                "noise_std": "$\sigma_N$"}



def load_results(results_dir, variable, labels):
    """ Loads all results saved under a project for plotting

    Parameters
    ----------
    results_dir: str
        the absolute path of the current project
    variable: str
        name of the metric we want to plot
    labels: list of str
        name of the parameters for which we compare (appearing in legend)
    """
    # find all projects
    projects = [os.path.join(results_dir, o) for o in os.listdir(results_dir)]
    projects = [el for el in projects if "plots" not in el]

    results = pd.DataFrame()
    for p in projects:
        config_file = p + "/config.yml"

        with open(config_file, "rb") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        trial_dirs = list(next(os.walk(p + "/trials"))[1])
        for trial, trial_dir in enumerate(trial_dirs):
            # load outcome of trial
            try:
                if variable == "Dispersal":
                    log = pickle.load(open(p + "/trials/" + trial_dir + '/log_updated.pickle', 'rb'))
                else:
                    log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))

            except IOError:
                break

            if variable == "survival":
                num_agents = list(log["num_agents"])
                if len(num_agents):
                    num_agents = num_agents[-1]
                    if num_agents > 1:
                        trial_variable = 1
                    else:
                        trial_variable = 0
                else:
                    trial_variable= len(log["Climate"]) / config.num_gens
            elif variable == "extinctions":
                trial_variable = np.mean(log[variable][:100])
            else:
                trial_variable = np.mean(log[variable][100:]) # compute the average after convergence

            # add new results to dataframe
            new_dict = {variable: [trial_variable],
                        "Trial": [trial],
                        "Climate": [config.climate_mean_init]}
            for label in labels:
                label_value = find_label(config, label)
                try:
                    label_value = float(label_value )
                except:
                    label_value = str(label_value )
                new_dict[label] = [label_value ]
            new_row = pd.DataFrame.from_dict(new_dict)

            if results.empty:
                results = new_row
            else:
                results = results.append(new_row)

    return results


def find_index(trial_dir):
    """ Returns the index of a trial based on its directory.

    Parameters
    ----------
    trial_dir: str
        directory of trial (in the form "*/trial_{trial_idx}")
    """
    # find the index of current trial
    trial = trial_dir.find("trial_")
    trial = int(trial_dir[(trial + 6):])
    return trial

def find_label(config, parameter="selection"):
    """ Returns the label for plotting based on the selected parameter.

    Parameters
    ----------
    config: dict
        configuration of project

    parameter: str
        name of parameter
    """
    label = ""
    if parameter == "selection":
        if config.selection_type == "NF":
            label += "NF-selection"
        elif config.selection_type == "N":
            label += "N-selection"
        elif config.selection_type == "F":
            label += "F-selection"
    elif parameter == "genome":
        if config.genome_type == "evolv":
            label += "$R$"
        elif config.genome_type == "no-evolv":
            label += "$R_{no-evolve}$"
        elif config.genome_type == "niche-construction":
            label += "$R_{NC}$"
        elif config.genome_type == "niche-construction-v2":
            label += "$R_{c2}$"


    else:
        label += str(getattr(config,parameter))
    return label

def compute_dispersal(log, log_niches, num_latitudes):
    """ Compute dispersal

    Parameters
    ---------
    log_niches: Dataframe
        information about niches

    num_latitudess: int
        number of latitudes(niches)
    """
    new_log = pd.DataFrame()
    window = 10 # temporal window to detect survival
    trials = list(set(log["Trial"]))
    for trial in trials:
        log_trial = log.loc[(log['Trial'] == trial)]
        climate = log_trial["Climate"].to_list()
        inhabited_niches = log_niches[trial]["inhabited_niches"]
        num_gens = min([len(inhabited_niches), len(climate)])

        all_DI = []
        for lat in range(-int(num_latitudes/2), int(num_latitudes/2 + 0.5)):
            # for each latitude
            survivals = []
            for gen in range(num_gens):
                if lat in inhabited_niches[gen]:
                    survival = 1
                else:
                    survival = 0

                survivals.append(survival)

            if not len(survivals):
                # in case there was a mass extinction before the first window
                survivals = [0]*window

            # for which generations was there at least individual for at least the previous #window generation?
            DI = list(np.convolve(survivals, np.ones(window, dtype=int), 'valid'))
            DI = [1 if el==window else 0 for el in DI]
            all_DI.append(DI)

        # dispersal is the summation of persistence across latitudes
        dispersal = []
        for el in range(len(all_DI[0])):
            # for each generation
            sum_disp = 0
            for lat_DI in all_DI:
                # sum over all latitudes
                sum_disp += lat_DI[el]
            dispersal.append(sum_disp)

        # the first #window generations have a placeholder value
        while len(dispersal) < num_gens:
            dispersal = [1] + dispersal

        # TODO: why do I compute the mean here?
        x = np.mean(np.array(dispersal), axis=0)
        log_trial["Dispersal"] = x

        if new_log.empty:
            new_log = log_trial
        else:
            new_log = new_log.append(log_trial)

    return new_log
