""" This script can be used to produce plots used in our Gecco paper.
"""
import sys
import os
import yaml
import matplotlib.pyplot as plt
import pickle5 as pickle
import numpy as np
import pandas as pd
import seaborn as sns

params = {'legend.fontsize': 5,
          "figure.autolayout": True,
          'font.size': 8}
plt.rcParams.update(params)
ci=95
cm = 1 / 2.54
figsize=(8.48 * cm, 6 * cm)

def sigma_constant(results_dir):
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of niches"
    """
    # find all projects
    projects = [os.path.join(results_dir, o) for o in os.listdir(results_dir)]
    projects = [el for el in projects if "plots" not in el ]

    count = 0
    for p in projects:
        config_file = p + "/config.yml"

        with open(config_file, "rb") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        trial_dirs = list(next(os.walk(p+ "/trials"))[1])
        for trial, trial_dir in enumerate(trial_dirs):
            # load outcome of trial
            log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))
            trial_sigma = np.mean(log["SD"][-100:])
            new_row = pd.DataFrame.from_dict({ 'SD': [trial_sigma],
                                               "Trial": [trial],
                                               "Num_niches": [config.num_niches],
                                               "Climate": [config.climate_mean_init]})
            if not count:
                results = new_row
            else:
                results = results.append(new_row)
            count =1
    niches = list(set(results["Num_niches"].to_list()))
    niches.sort()
    cm = 1 / 2.54
    plt.figure(figsize=(8.48*cm, 6*cm))
    for niche in niches:
        results_niche = results.loc[results['Num_niches'] == niche]
        sns.lineplot(data=results_niche, x="Climate", y="SD", ci=ci, label="$N=$" + str(
            niche))
    plt.yscale('log')
    #plt.ylim([10**(-19), 100])
    plt.xlabel("$e_{0,0}$, Reference Environmental State")
    plt.ylabel("$\\bar{\sigma}$, Plasticity")
    plt.legend(loc="best")
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/sigma_stable_main.png")
    plt.clf()

def find_label(config):
    label=""
    if config.selection_type == "capacity-fitness":
        label += "QD-selection"
    elif config.selection_type == "limited-capacity":
        label += "D-selection"
    elif config.selection_type == "FP-Grove":
        label += "Q-selection"
        
    """if config.genome_type == "1D":
        label+= "$G=G_{no-evolve}$"

    elif config.genome_type == "1D_mutate_fixed":
        label +=  "$G=G_{evolve_constant}$"

    elif config.genome_type == "1D_mutate":
        label += "$G=G_{evolve}$"""""

    return label





def sigma_appendices(results_dir, y_variables, label="Num_niches"):
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of niches"

    Args:
        case (str): pick among: "s2_g0","s2_g1", "s0_g2", "s1_g2"
        y_variable (str): pick among "SD" and "R"
    """
    fig, axs = plt.subplots(len(y_variables), figsize=(figsize[0], figsize[1] / 2*len(y_variables)))
    for y_idx, y_variable in enumerate(y_variables):
        # find all projects
        projects = [os.path.join(results_dir, o) for o in os.listdir(results_dir)]
        projects = [el for el in projects if "plots" not in el ]

        count = 0
        for p in projects:
            config_file = p + "/config.yml"

            with open(config_file, "rb") as f:
                config = yaml.load(f, Loader=yaml.UnsafeLoader)

            trial_dirs = list(next(os.walk(p+ "/trials"))[1])
            for trial, trial_dir in enumerate(trial_dirs):
                # load outcome of trial
                log = pickle.load(open(p + "/trials/" + trial_dir + '/log_updated.pickle', 'rb'))
                trial_sigma = np.mean(log[y_variable][-100:])
                if label == "Num_niches":
                    new_row = pd.DataFrame.from_dict({ y_variable: [trial_sigma],
                                                       "Trial": [trial],
                                                       "Num_niches": [config.num_niches],
                                                       "Climate": [config.climate_mean_init]})
                else:
                    new_row = pd.DataFrame.from_dict({y_variable: [trial_sigma],
                                                      "Trial": [trial],
                                                      "model": [find_label(config)],
                                                      "Climate": [config.climate_mean_init]})
                if not count:
                    results = new_row
                else:
                    results = results.append(new_row)
                count =1

        niches = list(set(results[label].to_list()))
        niches.sort()

        #plt.figure(figsize=(8.48*cm, 6*cm/scale_y))
        for niche in niches:
            results_niche = results.loc[results[label] == niche]

            if label == "Num_niches":

                sns.lineplot(ax=axs[y_idx], data=results_niche, x="Climate", y=y_variable, ci=ci, label="$N=$" + str(
                    niche),legend=0)
            else:
                sns.lineplot(ax=axs[y_idx], data=results_niche, x="Climate", y=y_variable, ci=ci, label= str(
                    niche),legend=0)

        if y_variable == "SD":
            axs[y_idx].set(ylabel="$\\bar{\sigma}$, Plasticity")
        elif y_variable == "R":
            axs[y_idx].set(ylabel="$\\bar{r}$, Evolvability")
        elif y_variable == "Dispersal":
            axs[y_idx].set(ylabel="$D$, Dispersal")
        axs[y_idx].set_yscale('log')

        handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left')

    #plt.legend(loc="best")
    plt.yscale('log')
    plt.xlabel("$e_{0,0}$, Reference Environmental State")
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/combined.png")
    plt.clf()

def extinctions_stable(results_dir):
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of niches"
    """
    labels = {"FP-Grove": "Q-selection",
              "capacity-fitness": "QD-selection"}
    # find all projects
    projects = [os.path.join(results_dir, o) for o in os.listdir(results_dir)]
    projects = [el for el in projects if "plots" not in el]

    count = 0
    for p in projects:
        config_file = p + "/config.yml"

        with open(config_file, "rb") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        trial_dirs = list(next(os.walk(p + "/trials"))[1])
        for trial, trial_dir in enumerate(trial_dirs):
            # load outcome of trial
            log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))
            total_pop = config.capacity*config.climate_mean_init
            print(total_pop, labels[config.selection_type], config.climate_mean_init)
            total_pop=1
            trial_extinctions = np.mean(log["extinctions"])/total_pop
            new_row = pd.DataFrame.from_dict({"extinctions": [trial_extinctions],
                                              "Trial": [trial],
                                              "Selection": [labels[config.selection_type]],
                                              "Climate": [config.climate_mean_init]})
            if not count:
                results = new_row
            else:
                results = results.append(new_row)
            count = 1
    selections = list(set(results["Selection"].to_list()))
    cm = 1 / 2.54
    plt.figure(figsize=(8.48 * cm, 6 * cm))
    for selection in selections:
        results_niche = results.loc[results['Selection'] == selection]
        sns.lineplot(data=results_niche, x="Climate", y="extinctions", ci=ci, label="$s=$" + selection)

    plt.xlabel("$e_{0,0}$, Reference Environmental State")
    plt.ylabel("$E$, Number of extinctions")
    plt.legend(loc="best")
    plt.savefig(results_dir + "/plots/extinct_constant.png")
    plt.clf()

def extinctions_stable_appendices(results_dir, label):
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of
    niches"
    """
    labels_selection = {"FP-Grove": "Q-selection",
                        "capacity-fitness": "QD-selection",
                        "limited-capacity": "D-selection"}

    labels_genome = {"1D": "$R_{no-evolve}$", "1D_mutate": "$R$", "1D_mutate_fixed": "$R_{constant}$"}

    # find all projects
    projects = [os.path.join(results_dir, o) for o in os.listdir(results_dir)]
    projects = [el for el in projects if "plots" not in el]

    count = 0
    for p in projects:
        config_file = p + "/config.yml"

        with open(config_file, "rb") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        trial_dirs = list(next(os.walk(p + "/trials"))[1])
        for trial, trial_dir in enumerate(trial_dirs):
            # load outcome of trial
            log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))
            trial_extinctions = np.mean(log["extinctions"])
            if label == "Selection":
                new_row = pd.DataFrame.from_dict({"extinctions": [trial_extinctions],
                                                  "Trial": [trial],
                                                  "Selection": [labels_selection[config.selection_type]],
                                                  "Climate": [config.climate_mean_init]})
            elif label == "Genome":
                new_row = pd.DataFrame.from_dict({"extinctions": [trial_extinctions],
                                                  "Trial": [trial],
                                                  "Genome": [labels_genome[config.genome_type]],
                                                  "Climate": [config.climate_mean_init]})
            elif label == "Niches":
                new_row = pd.DataFrame.from_dict({"extinctions": [trial_extinctions],
                                                  "Trial": [trial],
                                                  "Niches": [config.num_niches],
                                                  "Climate": [config.climate_mean_init]})
            if not count:
                results = new_row
            else:
                results = results.append(new_row)
            count = 1
    methods = list(set(results[label].to_list()))
    cm = 1 / 2.54
    plt.figure(figsize=(8.48 * cm, 6 * cm))
    for method in methods:
        results_niche = results.loc[results[label] == method]
        sns.lineplot(data=results_niche, x="Climate", y="extinctions", ci=ci, label= label + "=" + str(method))

    plt.xlabel("$e_0$, Reference Environmental State")
    plt.ylabel("$E$, Extinction events")
    plt.legend(loc="best")
    plt.savefig(results_dir + "/plots/extinct_stable.png")
    plt.clf()


def diversity_stable(results_dir):
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of niches"
    """
    labels = {"FP-Grove": "Q-selection",
              "capacity-fitness": "QD-selection",
                        "limited-capacity": "D-selection"}
    # find all projects
    projects = [os.path.join(results_dir, o) for o in os.listdir(results_dir)]
    projects = [el for el in projects if "plots" not in el]

    count = 0
    for p in projects:
        config_file = p + "/config.yml"

        with open(config_file, "rb") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        trial_dirs = list(next(os.walk(p + "/trials"))[1])
        for trial, trial_dir in enumerate(trial_dirs):
            # load outcome of trial
            log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))
            trial_diversity = np.mean(log["diversity"][100:])
            new_row = pd.DataFrame.from_dict({"Diversity": [trial_diversity],
                                              "Trial": [trial],
                                              "Selection": [labels[config.selection_type]],
                                              "Climate": [config.climate_mean_init]})
            if not count:
                results = new_row
            else:
                results = results.append(new_row)
            count = 1
    selections = list(set(results["Selection"].to_list()))
    plt.figure(figsize=figsize)
    for selection in selections:
        results_niche = results.loc[results['Selection'] == selection]
        sns.lineplot(data=results_niche, x="Climate", y="Diversity", ci=ci, label=selection)

    plt.xlabel("$e_{0,0}$, Reference Environmental State")
    plt.ylabel("$V$, Divesity")
    plt.legend(loc="best")
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/diversity_stable.png")
    plt.clf()



def diversity_stable_appendices(results_dir, label):
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of
    niches"
    """
    labels_selection = {"FP-Grove": "Q-selection",
                        "capacity-fitness": "QD-selection",
                        "limited-capacity": "D-selection"}

    labels_genome = {"1D": "$R_{no-evolve}$", "1D_mutate": "$R$", "1D_mutate_fixed": "$R_{constant}$"}

    # find all projects
    projects = [os.path.join(results_dir, o) for o in os.listdir(results_dir)]
    projects = [el for el in projects if "plots" not in el]

    count = 0
    for p in projects:
        config_file = p + "/config.yml"

        with open(config_file, "rb") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        trial_dirs = list(next(os.walk(p + "/trials"))[1])
        for trial, trial_dir in enumerate(trial_dirs):
            # load outcome of trial
            log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))
            trial_extinctions = np.mean(log["diversity"])
            if label == "Selection":
                new_row = pd.DataFrame.from_dict({"diversity": [trial_extinctions],
                                                  "Trial": [trial],
                                                  "Selection": [labels_selection[config.selection_type]],
                                                  "Climate": [config.climate_mean_init]})
            elif label == "Genome":
                new_row = pd.DataFrame.from_dict({"diversity": [trial_extinctions],
                                                  "Trial": [trial],
                                                  "Genome": [labels_genome[config.genome_type]],
                                                  "Climate": [config.climate_mean_init]})
            elif label == "Niches":
                new_row = pd.DataFrame.from_dict({"diversity": [trial_extinctions],
                                                  "Trial": [trial],
                                                  "Niches": [config.num_niches],
                                                  "Climate": [config.climate_mean_init]})
            if not count:
                results = new_row
            else:
                results = results.append(new_row)
            count = 1
    methods = list(set(results[label].to_list()))
    cm = 1 / 2.54
    plt.figure(figsize=(8.48 * cm, 6 * cm))
    for method in methods:
        results_niche = results.loc[results[label] == method]
        sns.lineplot(data=results_niche, x="Climate", y="diversity", ci=ci, label= label + "=" + str(method))

    plt.xlabel("$e_{0,0}$, Reference Environmental State")
    plt.ylabel("$D$, D")
    plt.legend(loc="best")
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/diversity_stable.png")
    plt.clf()

def dispersal_stable(results_dir):
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of niches"
    """
    labels = {"FP-Grove": "Q-selection",
              "capacity-fitness": "QD-selection",
                        "limited-capacity": "D-selection"}
    # find all projects
    projects = [os.path.join(results_dir, o) for o in os.listdir(results_dir)]
    projects = [el for el in projects if "plots" not in el]

    count = 0
    for p in projects:
        config_file = p + "/config.yml"

        with open(config_file, "rb") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        trial_dirs = list(next(os.walk(p + "/trials"))[1])
        for trial, trial_dir in enumerate(trial_dirs):
            # load outcome of trial
            log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))
            trial_diversity = np.mean(log["dispersal"][100:])
            new_row = pd.DataFrame.from_dict({"Dispersal": [trial_diversity],
                                              "Trial": [trial],
                                              "Selection": [labels[config.selection_type]],
                                              "Climate": [config.climate_mean_init]})
            if not count:
                results = new_row
            else:
                results = results.append(new_row)
            count = 1
    selections = list(set(results["Selection"].to_list()))
    cm = 1 / 2.54
    plt.figure(figsize=(8.48 * cm, 6 * cm))
    for selection in selections:
        results_niche = results.loc[results['Selection'] == selection]
        sns.lineplot(data=results_niche, x="Climate", y="Dispersal", ci=ci, label="Selection=" + selection)

    plt.xlabel("$e_{0,0}$, Reference Environmental State")
    plt.ylabel("D, Dispersal")
    plt.legend(loc="best")
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/dispersal_stable.png")
    plt.clf()

def mass_periodic(results_dir, label="$A_e$"):
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of niches"
    """
    labels = {"FP-Grove": "Q-selection",
              "capacity-fitness": "QD-selection",
                        "limited-capacity": "D-selection"}
    # find all projects
    projects = [os.path.join(results_dir, o) for o in os.listdir(results_dir)]
    projects = [el for el in projects if "plots" not in el]

    count = 0
    for p in projects:
        print(p)
        config_file = p + "/config.yml"

        with open(config_file, "rb") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        trial_dirs = list(next(os.walk(p + "/trials"))[1])
        for trial, trial_dir in enumerate(trial_dirs):
            # load outcome of trial
            log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))
            trial_diversity = np.mean(log["diversity"][100:])
            print(len(log["Climate"]), config.num_gens)
            trial_duration = len(log["Climate"])/config.num_gens
            print(config.num_gens)
            if label == "$A_e$":
                label_value = config.amplitude
            elif label== "N":
                label_value = config.num_niches
            new_row = pd.DataFrame.from_dict({"Duration": [trial_duration],
                                              "Trial": [trial],
                                              "Period": [config.period],
                                              label: [label_value]})
            if not count:
                results = new_row
            else:
                results = results.append(new_row)
            count = 1
    amplitudes= list(set(results[label].to_list()))
    cm = 1 / 2.54
    plt.figure(figsize=(8.48 * cm, 6 * cm))
    for amplitude in amplitudes:
        results_ampl = results.loc[results[label] == amplitude]
        sns.lineplot(data=results_ampl, x="Period", y="Duration", ci=ci, label=label + "=" + str(amplitude))

    plt.xlabel("$T$, Period of sinusoid")
    plt.ylabel("$\\bar{v}$, Average Survival")
    plt.legend(loc="best")
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/survival_stable.png")
    plt.clf()




if __name__ == "__main__":
    results_dir = "../projects/papers/gecco/stable/sigma_main"
    sigma_constant(results_dir=results_dir)

    results_dir = "../projects/papers/gecco/stable/sigma_appendices/s2_g0"
    #sigma_appendices(results_dir=results_dir, y_variable="SD")

    results_dir = "../projects/papers/gecco/stable/sigma_appendices/s2_g0"
    #sigma_appendices(results_dir=results_dir, y_variable="R")

    results_dir = "../projects/papers/gecco/stable/sigma_appendices/s2_g1"
    #sigma_appendices(results_dir=results_dir, y_variable="SD")

    results_dir = "../projects/papers/gecco/stable/sigma_appendices/s2_g1"
    #sigma_appendices(results_dir=results_dir, y_variable="R")

    results_dir = "../projects/papers/gecco/stable/sigma_appendices/s0_g2"
    #sigma_appendices(results_dir=results_dir, y_variable="SD")

    results_dir = "../projects/papers/gecco/stable/sigma_appendices/s0_g2"
    #sigma_appendices(results_dir=results_dir, y_variable="R")

    results_dir = "../projects/papers/gecco/stable/sigma_appendices/s1_g2"
    #sigma_appendices(results_dir=results_dir, y_variable="SD")

    results_dir = "../projects/papers/gecco/stable/sigma_appendices/s1_g2"
    #sigma_appendices(results_dir=results_dir, y_variable="R")

    results_dir = "../projects/papers/gecco/stable/sigma_appendices/N_100"
    sigma_appendices(results_dir=results_dir, y_variables=["SD","R", "Dispersal"], label="model")

    results_dir = "../projects/papers/gecco/stable/sigma_appendices/N_100"
    #sigma_appendices(results_dir=results_dir, y_variable="SD", label="model")

    results_dir = "../projects/papers/gecco/stable/sigma_appendices/N_100"
    #sigma_appendices(results_dir=results_dir, y_variable="Dispersal", label="model")

    results_dir = "../projects/papers/gecco/stable/extinct_main"
    #extinctions_stable(results_dir)

    #results_dir = "../projects/papers/gecco/stable/extinct_appendices_s"
    #extinctions_stable_appendices(results_dir,label="Selection")

    #results_dir = "../projects/papers/gecco/stable/extinct_appendices_g"
    #extinctions_stable_appendices(results_dir, label="Genome")

    #results_dir = "../projects/papers/gecco/stable/extinct_appendices_N"
    #extinctions_stable_appendices(results_dir, label="Niches")

    results_dir = "../projects/papers/gecco/stable/diversity_main"
    diversity_stable(results_dir)

    #results_dir = "../projects/papers/gecco/stable/dispersal_main"
    #dispersal_stable(results_dir)

    #results_dir = "../projects/papers/gecco/stable/diversity_appendices_g"
    #diversity_stable_appendices(results_dir, label="Genome")

    #results_dir = "../projects/papers/gecco/stable/diversity_appendices_N"
    #diversity_stable_appendices(results_dir, label="Niches")

    results_dir = "../projects/papers/gecco/periodic/survival/s2_g2_100"
    #mass_periodic(results_dir)

    results_dir = "../projects/papers/gecco/periodic/survival/s2_g2_A4"
    #mass_periodic(results_dir, label="N")

    #results_dir = "../projects/papers/gecco/periodic/survival/s0_g2_100"
    #mass_periodic(results_dir)

    #results_dir = "../projects/papers/gecco/periodic/survival/s0_g2_A4"
    #mass_periodic(results_dir, label="num_niches")

    #results_dir = "../projects/papers/gecco/periodic/survival/s1_g2_100"
    #mass_periodic(results_dir)

    #results_dir = "../projects/papers/gecco/periodic/survival/s1_g2_A4"
    #mass_periodic(results_dir, label="num_niches")