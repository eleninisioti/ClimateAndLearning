""" This script can be used to reproduce all figures in our paper 'Plasticity and evolvability under environmental
 variability: the joint role of fitness-based selection and niche-limited competition'.
"""
import os
import yaml
import matplotlib.pyplot as plt
import pickle5 as pickle
import numpy as np
import pandas as pd
import seaborn as sns
from types import SimpleNamespace
from utils import find_label, label_colors

# ----- configuration for figures -----
params = {'legend.fontsize': 5,
          "figure.autolayout": True,
          'font.size': 8}
plt.rcParams.update(params)
ci = 95
cm = 1 / 2.54
figsize = (8.48 * cm, 6 * cm)
# -------------------------------------


def sigma():
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of niches.

    """
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
            try:
                log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))
            except IOError:
                break
            trial_sigma = np.mean(log["SD"][-100:])
            new_row = pd.DataFrame.from_dict({'SD': [trial_sigma],
                                              "Trial": [trial],
                                              "Num_niches": [config.num_niches],
                                              "Climate": [config.climate_mean_init]})
            if not count:
                results = new_row
            else:
                results = results.append(new_row)
            count = 1
    niches = list(set(results["Num_niches"].to_list()))
    niches.sort()
    cm = 1 / 2.54
    plt.figure(figsize=(8.48 * cm, 6 * cm))
    for niche in niches:
        results_niche = results.loc[results['Num_niches'] == niche]
        sns.lineplot(data=results_niche, x="Climate", y="SD", ci=ci, label="$N=$" + str(
            niche))
    plt.yscale('log')
    plt.xlabel("$e_{0}^0$, Reference Environmental State")
    plt.ylabel("$\\bar{\sigma}^*$, Plasticity")
    plt.legend(loc="upper right")
    plt.ylim([10^(-16), 10])
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/stable_sigma.pdf", dpi=300)
    plt.clf()


def sigma_selection( y_variables, label="Num_niches"):
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of niches"

    Parameters
    ----------
    label: str
    y_variables: list of string
    """
    fig, axs = plt.subplots(len(y_variables), figsize=(figsize[0], figsize[1] / 2 * len(y_variables)))
    for y_idx, y_variable in enumerate(y_variables):
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
                try:
                    log = pickle.load(open(p + "/trials/" + trial_dir + '/log_updated.pickle', 'rb'))
                except IOError:
                    break
                trial_sigma = np.mean(log[y_variable][-100:])
                if label == "Num_niches":
                    new_row = pd.DataFrame.from_dict({y_variable: [trial_sigma],
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
                count = 1

        niches = list(set(results[label].to_list()))
        niches.sort()

        for niche in niches:
            results_niche = results.loc[results[label] == niche]

            if label == "Num_niches":

                sns.lineplot(ax=axs[y_idx], data=results_niche, x="Climate", y=y_variable, ci=ci, label="$N=$" + str(
                    niche), legend=0)
            else:
                sns.lineplot(ax=axs[y_idx], data=results_niche, x="Climate", y=y_variable, ci=ci, label=str(
                    niche), legend=0)

        if y_variable == "SD":
            axs[y_idx].set(ylabel="$\\bar{\sigma}^*$, Plasticity")
        elif y_variable == "R":
            axs[y_idx].set(ylabel="$\\bar{r}^*$, Evolvability")
        elif y_variable == "Dispersal":
            axs[y_idx].set(ylabel="$D^*$, Dispersal")
        axs[y_idx].set_yscale('log')
        axs[y_idx].set(xlabel="$e_{0}^0$, Reference Environmental State")

        handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left')
    for ax in axs.flat:
        ax.label_outer()
    # plt.legend(loc="best")
    plt.yscale('log')
    axs[1].set(xlabel="$e_{0}^0$, Reference Environmental State")
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/stable_selection.pdf", dpi=300)
    plt.clf()


def extinct():
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of niches"
    """
    # find all projects
    projects = [os.path.join(results_dir, o) for o in os.listdir(results_dir)]
    projects = [el for el in projects if "plots" not in el]
    labels_genome = {"1D": "$O_{no-evolve}$", "1D_mutate": "$O_{evolve}$", "1D_mutate_fixed": "$R_{constant}$"}

    count = 0
    for p in projects:
        config_file = p + "/config.yml"

        with open(config_file, "rb") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        trial_dirs = list(next(os.walk(p + "/trials"))[1])
        for trial, trial_dir in enumerate(trial_dirs):
            # load outcome of trial
            try:
                log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))
            except IOError:
                break
            total_pop = config.capacity * config.climate_mean_init
            total_pop = 1
            trial_extinctions = np.mean(log["extinctions"]) / total_pop
            new_row = pd.DataFrame.from_dict({"extinctions": [trial_extinctions],
                                              "Trial": [trial],
                                              "Selection": [find_label(config)],
                                              "Climate": [config.climate_mean_init],
                                              "Genome": [find_label(config,parameter="genome")]}
                                             )
            if not count:
                results = new_row
            else:
                results = results.append(new_row)
            count = 1
    selections = list(set(results["Selection"].to_list()))
    genomes = list(set(results["Genome"].to_list()))
    cm = 1 / 2.54
    plt.figure(figsize=(8.48 * cm, 6 * cm))
    for selection in selections:
        for genome in genomes:
            results_niche = results.loc[results['Selection'] == selection]
            results_niche = results_niche.loc[results_niche['Genome'] == genome]
            sns.lineplot(data=results_niche, x="Climate", y="extinctions", ci=ci, label=selection + ", " + genome)

    plt.xlabel("$e_{0}^0$, Reference Environmental State")
    plt.ylabel("$X^*$, Number of extinctions")
    plt.yscale('log')
    plt.legend(loc="best")
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/stable_extinct.pdf", dpi=300)
    plt.clf()


def diversity():
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of niches"
    """

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
            try:
                log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))
            except IOError:
                break
            trial_diversity = np.mean(log["diversity"][100:])
            new_row = pd.DataFrame.from_dict({"Diversity": [trial_diversity],
                                              "Trial": [trial],
                                              "Selection": [find_label(config)],
                                              "Climate": [config.climate_mean_init]})
            if not count:
                results = new_row
            else:
                results = results.append(new_row)
            count = 1
    selections = list(set(results["Selection"].to_list()))
    selections = ["F-selection",  "N-selection","NF-selection"]
    plt.figure(figsize=figsize)
    for selection in selections:
        results_niche = results.loc[results['Selection'] == selection]
        sns.lineplot(data=results_niche, x="Climate", y="Diversity", ci=ci, label=selection)

    plt.xlabel("$e_{0}^0$, Reference Environmental State")
    plt.ylabel("$V^*$, Divesity")
    plt.legend(loc="best")
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/stable_diversity.pdf", dpi=300)
    plt.clf()


def survival(label="$A_e$"):
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of niches"
    """
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
            try:
                log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))
            except IOError:
                break
            trial_diversity = np.mean(log["diversity"][100:])
            trial_duration = len(log["Climate"]) / config.num_gens
            if label == "$A_e$":
                label_value = config.amplitude
            elif label == "N":
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
    amplitudes = list(set(results[label].to_list()))
    cm = 1 / 2.54
    plt.figure(figsize=(8.48 * cm, 6 * cm))
    for amplitude in amplitudes:
        results_ampl = results.loc[results[label] == amplitude]

        sns.lineplot(data=results_ampl, x="Period", y="Duration", ci=ci, label=label + "=" + str(amplitude))

    plt.xlabel("$T_e$, Period of sinusoid")
    plt.ylabel("$A$, Survival")
    plt.legend(loc="best")
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/survival.pdf", dpi=300)
    plt.clf()


def survival_noisy(label="$A_e$"):
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of niches"
    """
    labels = {"FP-Grove": "Q-selection",
              "capacity-fitness": "QD-selection",
              "limited-capacity": "D-selection"}
    # find all projects
    projects = [os.path.join(results_dir, o) for o in os.listdir(results_dir)]
    projects = [el for el in projects if "plots" not in el]
    ordered_projects = []
    order = ["F_G", "N_G", "NF_G"]
    for o in order:

        for p in projects:
            if o in p:
                ordered_projects.append(p)

    projects = ordered_projects

    count = 0
    for p in projects:
        print(p)
        config_file = p + "/config.yml"

        with open(config_file, "rb") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        trial_dirs = list(next(os.walk(p + "/trials"))[1])
        for trial, trial_dir in enumerate(trial_dirs):
            # load outcome of trial
            try:
                log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))
            except IOError:
                break
            trial_diversity = np.mean(log["diversity"][100:])
            if len(log["Climate"])  == 500:
                print("change length")
                length = 1500
            else:

                length = len(log["Climate"])
            config.num_gens = 1500
            trial_duration = length / config.num_gens
            if label == "$A_e$":
                label_value = config.amplitude
            elif label == "N":
                label_value = config.num_niches
            elif label == "method":
                label_value = "method"
            new_row = pd.DataFrame.from_dict({"Duration": [trial_duration],
                                              "Trial": [trial],
                                              "Noise": [config.noise_std],
                                              "Method": [find_label(config)],
                                              label: [label_value]})
            if not count:
                results = new_row
            else:
                results = results.append(new_row)
            count = 1
    amplitudes = list(set(results[label].to_list()))
    methods = list(set(results["Method"].to_list()))
    methods = ["F-selection", "N-selection", "NF-selection"]
    cm = 1 / 2.54
    plt.figure(figsize=(8.48 * cm, 6 * cm))
    for amplitude in amplitudes:
        for method in methods:
            results_ampl = results.loc[results[label] == amplitude]
            results_ampl = results_ampl.loc[results_ampl["Method"] == method]

            sns.lineplot(data=results_ampl, x="Noise", y="Duration", ci=ci, label=method)

    plt.xlabel("$\sigma_{N}$, Standard deviation of noise")
    plt.ylabel("$A$, Survival")
    plt.legend(loc="best")
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/survival_noisy.pdf", dpi=300)
    plt.clf()


def evolution_compare(include):
    """ Plot the evolution of climate and population dynamics.

    Parameters
    ----------
    log: dict
        results produced during simulation

    include: list of str
        which parameters to include in the plot. options are ["climate", "mean", "sigma", "mutate",
        "n_agents", "n_extinctions", "fitness"]
    """
    ci = 95
    cm = 1 / 2.54
    scale = 1
    fig_size = (8.48 * cm / scale, 6 * cm / scale)
    fig, axs = plt.subplots(len(include), figsize=(fig_size[0], fig_size[1] / 3 * len(include)))
    count = 0
    y_upper_thres = 10 ** (10)
    y_lower_thres = 1 / (10 ** (10))
    interval = 1

    # load and config files for both projects
    results = {}
    projects = [os.path.join(results_dir, o) for o in os.listdir(results_dir)]
    ordered_projects = []
    order = ["F_G", "N_G", "NF_G"]
    for o in order:

        for p in projects:
            if o in p:
                ordered_projects.append(p)

    projects = ordered_projects

    for p in projects:
        if "plots" not in p:
            trial_dirs = list(next(os.walk(p + "/trials"))[1])
            for trial, trial_dir in enumerate(trial_dirs):
                # load outcome of trial
                try:
                    log = pickle.load(open(p + "/trials/" + trial_dir + '/log_updated.pickle', 'rb'))
                    log_niches = pickle.load(open(p + "/trials/" + trial_dir + '/log_niches.pickle', 'rb'))

                except IOError:
                    print("No log file for project")
                    return 1

                if trial == 0:
                    log_df = log
                    log_niches_total = [log_niches]
                else:
                    log_df = log_df.append(log)
                    log_niches_total.append(log_niches)

            skip_lines = 1
            with open(p + "/config.yml") as f:
                for i in range(skip_lines):
                    _ = f.readline()
                config = yaml.load(f)
                config = SimpleNamespace(**config)

            log_df = log_df.loc[(log_df['Generation'] < 500)]

            label = find_label(config)
            results[label] = [log_df, log_niches_total, config]

    if "climate" in include:

        for key, value in results.items():
            label = key
            log = value[0]
            log_niches = value[1]
            config = value[2]
            log_trial = log.loc[(log['Trial'] == 0)]

            # find mean across niches:
            climate_avg = []
            for el in list(log_trial["Climate"]):
                niches_states = [el + 0.01 * idx for idx in range(-int(config.num_niches / 2),
                                                                  int(config.num_niches / 2 + 0.5))]
                climate_avg.append(np.mean(niches_states))
            log_trial["Climate_avg"] = climate_avg
            x = log_trial["Generation"][::interval]
            y = log_trial["Climate_avg"][::interval]

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci, label=label, legend=0)
            if len(y) > 100:
                break
        axs[count].set(ylabel="$e_0$")
        axs[count].set(xlabel=None)
        count += 1

    if "mean" in include:
        for key, value in results.items():
            label = key
            log = value[0]
            log_niches = value[1]
            config = value[2]
            x = log["Generation"][::interval]
            y = log["Mean"][::interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci, label=label, legend=0)

        axs[count].set(ylabel="$\\bar{\mu}$ ")
        axs[count].set(xlabel=None)
        count += 1

    if "sigma" in include:
        for key, value in results.items():
            label = key
            log = value[0]
            log_niches = value[1]
            config = value[2]
            x = log["Generation"][::interval]
            y = log["SD"][::interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci, label=label, legend=0)

        # sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="SD")
        axs[count].set(ylabel="$\\bar{\sigma}$")
        axs[count].set(xlabel=None)
        axs[count].set_yscale('log')
        axs[count].set_ylim([0.0001, 1])
        count += 1

    if "mutate" in include:
        for key, value in results.items():
            label = key
            log = value[0]
            x = log["Generation"][::interval]
            y = log["R"][::interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci, label=label, legend=0)

        # sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="R")
        axs[count].set(ylabel="$\\bar{r}$")
        axs[count].set(xlabel=None)
        axs[count].set_yscale('log')
        axs[count].set_yticks(ticks=[1, 10 ** (-5), 10 ** (-10)])

        count += 1

    if "fitness" in include:
        for key, value in results.items():
            label = key
            log = value[0]
            log_niches = value[1]
            config = value[2]
            x = log["Generation"][::interval]
            y = log["Fitness"][::interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci, label=label, legend=0)

        # sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="Fitness")
        axs[count].set(xlabel="Time (in generations)")
        axs[count].set(ylabel="$\\bar{f}$")
        count += 1

    if "extinct" in include:
        for key, value in results.items():
            label = key
            log = value[0]
            log_niches = value[1]
            config = value[2]
            x = log["Generation"][::interval]
            y = log["extinctions"][::interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci, label=label, legend=0)

        # sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="extinctions")
        axs[count].set(xlabel="Time (in generations)")
        axs[count].set(ylabel="$E$")
        count += 1

    if "num_agents" in include:
        for key, value in results.items():
            label = key
            log = value[0]
            log_niches = value[1]
            config = value[2]
            x = log["Generation"][::interval]
            y = log["num_agents"][::interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci, label="label", legend=0)

        # sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="num_agents")
        axs[count].set(xlabel="Time (in generations)")
        axs[count].set(ylabel="$N$, number of agents")
        count += 1
    if "diversity" in include:
        for key, value in results.items():
            label = key
            log = value[0]
            log_niches = value[1]
            config = value[2]
            x = log["Generation"][::interval]
            y = log["diversity"][::interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci, label=label, legend=0)

        # sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="diversity")
        axs[count].set(xlabel="Time (in generations)")
        axs[count].set(ylabel="$V$")
        count += 1
    if "fixation_index" in include:
        for key, value in results.items():
            label = key
            log = value[0]
            log_niches = value[1]
            config = value[2]
            x = log["Generation"][::interval]
            y = log["fixation_index"][::interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci, label=label, legend=0)

        # sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="fixation_index")
        axs[count].set(xlabel="Time (in generations)")
        axs[count].set(ylabel="$F_{st}$, fixation_index")
        count += 1
    if "dispersal" in include:
        for key, value in results.items():
            label = key
            log = value[0]
            log_niches = value[1]
            config = value[2]

            x = log["Generation"][::interval]
            y = log["Dispersal"][::interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci, label=label, legend=0)

        # sns.lineplot(ax=axs[count], data=log, x="Generation", y="Dispersal")
        axs[count].set(xlabel="Time (in generations)")
        axs[count].set(ylabel="$D$")
        count += 1
        handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left')
    for ax in axs.flat:
        ax.label_outer()
    axs[count - 1].set(xlabel="$G$, Generation")

    # -------------------------------------------------------------
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/evolution_other.pdf", dpi=300)
    plt.clf()
    return


if __name__ == "__main__":
    # ------ stable climate function -----
    results_dir = "../projects/paper/stable/sigma"
    #sigma()

    results_dir = "../projects/paper/stable/selection"
    #sigma_selection( y_variables=["SD", "Dispersal"], label="model")

    results_dir = "../projects/paper/stable/extinct"
    #extinct()

    results_dir = "../projects/paper/stable/diversity"
    #diversity()
    # ---------------------------------------
    # ------ sinusoid climate function -----
    results_dir = "../projects/paper/sin/survival_N"
    #survival()

    results_dir = "../projects/paper/sin/survival_A"
    #survival(label="N")

    results_dir = "../projects/paper/sin/evolution_slow"
    include = ["climate", "mean",
               "sigma", "mutate",
               "dispersal", "diversity"]
    #evolution_compare(include)

    results_dir = "../projects/paper/sin/evolution_quick"
    include = ["climate", "mean",
               "sigma", "mutate",
               "dispersal", "diversity"]
    #evolution_compare(include)

    results_dir = "../projects/paper/noisy/survival"
    survival_noisy()

    results_dir = "../projects/paper/noisy/evolution"
    include = ["climate", "mean",
               "sigma", "mutate",
               "dispersal", "diversity"]
    #evolution_compare(include)

