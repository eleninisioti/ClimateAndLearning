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

params = {'legend.fontsize': 6,
          "figure.autolayout": True,
          'font.size': 8}
plt.rcParams.update(params)
ci=95


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
    plt.xlabel("$e_0$, Reference Environmental State")
    plt.ylabel("$\\bar{\sigma}$, Average Plasticity")
    plt.legend(loc="best")
    plt.savefig(results_dir + "/plots/sigma_constant.png")
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
            trial_extinctions = np.mean(log["extinctions"])
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

    plt.xlabel("$e_0$, Reference Environmental State")
    plt.ylabel("$E$, Extinction events")
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
    cm = 1 / 2.54
    plt.figure(figsize=(8.48 * cm, 6 * cm))
    for selection in selections:
        results_niche = results.loc[results['Selection'] == selection]
        sns.lineplot(data=results_niche, x="Climate", y="Diversity", ci=ci, label="Selection=" + selection)

    plt.xlabel("$e_0$, Reference Environmental State")
    plt.ylabel("$\\bar{v}$, Average Divesity")
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

    plt.xlabel("$e_0$, Reference Environmental State")
    plt.ylabel("$\\bar{v}$, Average_diversity")
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

    plt.xlabel("$e_0$, Reference Environmental State")
    plt.ylabel("$\\bar{d}$, Average Dispersal")
    plt.legend(loc="best")
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/dispersal_stable.png")
    plt.clf()

def mass_periodic(results_dir):
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
            trial_duration = len(log["Climate"])
            new_row = pd.DataFrame.from_dict({"Duration": [trial_duration],
                                              "Trial": [trial],
                                              "Period": [config.period],
                                              "Amplitude": [config.amplitude]})
            if not count:
                results = new_row
            else:
                results = results.append(new_row)
            count = 1
    amplitudes= list(set(results["Amplitude"].to_list()))
    cm = 1 / 2.54
    plt.figure(figsize=(8.48 * cm, 6 * cm))
    for amplitude in amplitudes:
        results_ampl = results.loc[results['Amplitude'] == amplitude]
        sns.lineplot(data=results_ampl, x="Period", y="Duration", ci=ci, label="Amplitude=" + str(amplitude))

    plt.xlabel("$T$, Period of sinusoid")
    plt.ylabel("$\\bar{v}$, Average Survival")
    plt.legend(loc="best")
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/survival_stable.png")
    plt.clf()

def plot_evolution_common(include, project, log, num_niches):
    """ Plot the evolution of climate and population dynamics.

    Parameters
    ----------
    log: dict
        results produced during simulation

    include: list of str
        which parameters to include in the plot. options are ["climate", "mean", "sigma", "mutate",
        "n_agents", "n_extinctions", "fitness"]
    """
    params = {'legend.fontsize': 20,
              'legend.handlelength': 2,
              'font.size': 20,
              "figure.autolayout": True}
    plt.rcParams.update(params)
    params = {'legend.fontsize': 6,
              "figure.autolayout": True,
              'font.size': 8}
    plt.rcParams.update(params)
    ci = 95
    cm = 1 / 2.54
    scale = 1
    fig_size = (8.48 * cm / scale, 6 * cm / scale)
    fig, axs = plt.subplots(len(include), figsize=(fig_size[0], fig_size[1] / 2 * len(include)))
    if include == ["climate"]:
        axs = [axs]

    count = 0
    interval = int(len(log["Climate"])/2000)
    y_upper_thres = 10 ** (10)
    y_lower_thres = 1 / (10 ** (10))

    ## TODO: remove
    # self.log["Generation"] = [idx for idx in range(len(self.log["Climate"]))]

    if "climate" in include:
        log_climate = log.loc[(log['Trial'] == 0)]

        # find mean across niches:
        climate_avg = []
        for el in list(log_climate["Climate"]):
            niches_states = [el + 0.01 * idx for idx in range(-int(num_niches / 2),
                                                              int(num_niches / 2 + 0.5))]
            climate_avg.append(np.mean(niches_states))
        log_climate["Climate_avg"] = climate_avg
        # sns.lineplot(, data=self.log, x="Generation", y="Climate_avg")
        x = log_climate["Generation"][::interval]
        y = log_climate["Climate_avg"][::interval]
        # y = y.clip(upper=y_upper_thres)
        # y = y.clip(lower=y_lower_thres)

        if self.climate_noconf:
            sns.lineplot(ax=axs[count], x=x, y=y, ci=None)
        else:
            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)
        # axs[count].plot(self.log["Generation"], self.log["Climate_avg"])
        # axs[count].fill_between(x, (y - ci), (y + ci), color='b', alpha=.1)
        axs[count].set(ylabel="$\\bar{e}$")
        axs[count].set(xlabel=None)
        count += 1

    if "mean" in include:
        x = log_climate["Generation"][::self.interval]
        y = log_climate["Mean"][::self.interval]
        y = y.clip(upper=y_upper_thres)
        y = y.clip(lower=y_lower_thres)

        sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

        # sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="Mean")
        axs[count].set(ylabel="$\\bar{\mu}$")
        axs[count].set(xlabel=None)
        count += 1

    if "sigma" in include:
        x = log_climate["Generation"][::self.interval]
        y = log_climate["SD"][::self.interval]
        y = y.clip(upper=y_upper_thres)
        y = y.clip(lower=y_lower_thres)

        sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

        # sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="SD")
        axs[count].set(ylabel="$\\bar{\sigma}$")
        axs[count].set(xlabel=None)
        axs[count].set_yscale('log')
        count += 1

    if "mutate" in include:
        x = log_climate["Generation"][::interval]
        y = log_climate["R"][::interval]
        y = y.clip(upper=y_upper_thres)
        y = y.clip(lower=y_lower_thres)

        sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

        # sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="R")
        axs[count].set(ylabel="$\\bar{r}$")
        axs[count].set(xlabel=None)
        axs[count].set_yscale('log')
        count += 1

    if "fitness" in include:
        x = log_climate["Generation"][::interval]
        y = log_climate["Fitness"][::interval]
        y = y.clip(upper=y_upper_thres)
        y = y.clip(lower=y_lower_thres)

        sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

        # sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="Fitness")
        axs[count].set(xlabel="Time (in generations)")
        axs[count].set(ylabel="$\\bar{f}$")
        count += 1

    if "extinct" in include:
        x = log_climate["Generation"][::interval]
        y = log_climate["extinctions"][::interval]
        y = y.clip(upper=y_upper_thres)
        y = y.clip(lower=y_lower_thres)

        sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

        # sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="extinctions")
        axs[count].set(xlabel="Time (in generations)")
        axs[count].set(ylabel="Extinctions")
        count += 1

    if "num_agents" in include:
        x = log_climate["Generation"][::self.interval]
        y = log_climate["num_agents"][::self.interval]
        y = y.clip(upper=y_upper_thres)
        y = y.clip(lower=y_lower_thres)

        sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

        # sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="num_agents")
        axs[count].set(xlabel="Time (in generations)")
        axs[count].set(ylabel="$N$, number of agents")
        count += 1
    if "diversity" in include:
        x = log_climate["Generation"][::self.interval]
        y = log_climate["diversity"][::self.interval]
        y = y.clip(upper=y_upper_thres)
        y = y.clip(lower=y_lower_thres)

        sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

        # sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="diversity")
        axs[count].set(xlabel="Time (in generations)")
        axs[count].set(ylabel="$V$, diversity")
        count += 1
    if "fixation_index" in include:
        x = log_climate["Generation"][::self.interval]
        y = log_climate["fixation_index"][::self.interval]
        y = y.clip(upper=y_upper_thres)
        y = y.clip(lower=y_lower_thres)

        sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

        # sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="fixation_index")
        axs[count].set(xlabel="Time (in generations)")
        axs[count].set(ylabel="$F_{st}$, fixation_index")
        count += 1
    if "dispersal" in include:
        self.log = compute_dispersal(self.log, self.log_niches, self.num_niches)
        x = self.log["Generation"][::self.interval]
        y = self.log["Dispersal"][::self.interval]
        y = y.clip(upper=y_upper_thres)
        y = y.clip(lower=y_lower_thres)

        sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

        # sns.lineplot(ax=axs[count], data=log, x="Generation", y="Dispersal")
        axs[count].set(xlabel="Time (in generations)")
        axs[count].set(ylabel="$D$")
        count += 1

    axs[count - 1].set(xlabel="Time (in generations)")



if __name__ == "__main__":
    #results_dir = "../projects/papers/gecco/heatmaps_stable/s2_g0_1"
    #sigma_constant(results_dir=results_dir)

    #results_dir = "../projects/papers/gecco/stable/extinct_main"
    #extinctions_stable(results_dir)

    #results_dir = "../projects/papers/gecco/stable/extinct_appendices_s"
    #extinctions_stable_appendices(results_dir,label="Selection")

    #results_dir = "../projects/papers/gecco/stable/extinct_appendices_g"
    #extinctions_stable_appendices(results_dir, label="Genome")

    #results_dir = "../projects/papers/gecco/stable/extinct_appendices_N"
    #extinctions_stable_appendices(results_dir, label="Niches")

    results_dir = "../projects/papers/gecco/stable/diversity_main"
    #diversity_stable(results_dir)

    #results_dir = "../projects/papers/gecco/stable/dispersal_main"
    #dispersal_stable(results_dir)

    results_dir = "../projects/papers/gecco/stable/diversity_appendices_g"
    #diversity_stable_appendices(results_dir, label="Genome")

    results_dir = "../projects/papers/gecco/stable/diversity_appendices_N"
    #diversity_stable_appendices(results_dir, label="Niches")

    results_dir = "../projects/papers/gecco/periodic/survival/s2_g2_100"
    mass_periodic(results_dir)