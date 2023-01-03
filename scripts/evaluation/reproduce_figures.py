""" This script can be used to reproduce all figures in our paper 'Plasticity and evolvability under environmental
 variability: the joint role of fitness-based selection and niche-limited competition'.
"""
import sys
import os
import yaml
import matplotlib.pyplot as plt
import pickle5 as pickle
import numpy as np
import pandas as pd
import seaborn as sns
from types import SimpleNamespace
from utils import find_label, load_results, axes_labels, short_labels
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.evaluation.plot_projects import Plotter

# ----- configuration for figures -----
params = {'legend.fontsize': 5,
          "figure.autolayout": True,
          'font.size': 8,
          'pdf.fonttype':42,
          'ps.fonttype':42}
plt.rcParams.update(params)

ci = 95
cm = 1 / 2.54
figsize = (8.48 * cm, 6 * cm) # these values to fit in latex column
# -------------------------------------


def fig2():
    """ Plot Figure 2 """
    variable = "SD"
    labels = ["num_niches"]
    results = load_results(results_dir, variable, labels)

    niches = list(set([int(el) for el in results["num_niches"].to_list()]))
    niches.sort()
    plt.figure(figsize=figsize)

    for niche in niches:
        results_niche = results.loc[results['num_niches'] == niche]
        sns.lineplot(data=results_niche, x="Climate", y="SD", ci=ci, label="$N=$" + str(
            niche))

    plt.yscale('log')
    plt.xlabel("$e_{0}^0$, Reference Environmental State")
    plt.ylabel("$\\bar{\sigma}^*$, Plasticity")
    plt.legend(loc="upper right")
    plt.ylim([10**(-20), 1000])

    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/fig2.pdf", dpi=300)
    plt.savefig(save_dir + "/fig2.png", dpi=300)

    plt.clf()


def fig3():
    """ Plot Figure 3 """
    y_variables = ["SD", "Dispersal", "diversity"] # each will be a horizontal subplot
    fig, axs = plt.subplots(len(y_variables), figsize=(figsize[0], figsize[1] / 2 * len(y_variables)))
    label = ["selection"]

    for y_idx, y_variable in enumerate(y_variables):

        results = load_results(results_dir, y_variable, label)

        methods = list(set(results["selection"].to_list()))
        methods.sort()
        for method in methods:
            results_method = results.loc[results["selection"] == method]
            sns.lineplot(ax=axs[y_idx], data=results_method, x="Climate", y=y_variable, ci=ci, label=str(method),
                         legend=0)

        axs[y_idx].set(ylabel=axes_labels[y_variable])
        if y_variable == "SD":
            axs[y_idx].set_yscale('log')

        axs[y_idx].set(xlabel="$e_{0}^0$, Reference Environmental State")
        handles, labels = axs[-1].get_legend_handles_labels()

    # subplots share the same horizontal axis
    fig.legend(handles, labels, loc='lower left')
    for ax in axs.flat:
        ax.label_outer()
    axs[1].set(xlabel="$e_{0}^0$, Reference Environmental State")

    # final savvig
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/fig3.pdf", dpi=300)
    plt.savefig(save_dir + "/fig3.png", dpi=300)

    plt.clf()


def fig4():
    """ Plot Figure 4 """
    labels = ["selection", "genome"]
    y_variable = "extinctions"
    results = load_results(results_dir, y_variable, labels)


    selections = list(set(results["selection"].to_list()))
    genomes = list(set(results["genome"].to_list()))

    plt.figure(figsize=figsize)
    for selection in selections:
        for genome in genomes:
            results_niche = results.loc[results['selection'] == selection]
            results_niche = results_niche.loc[results_niche['genome'] == genome]
            sns.lineplot(data=results_niche, x="Climate", y="extinctions", ci=ci, label=selection + ", " + genome)

    plt.xlabel("$e_{0}^0$, Reference Environmental State")
    plt.ylabel("$X^{early}$, Number of extinctions")
    plt.yscale('log')
    plt.legend(loc="best")

    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/fig4.pdf", dpi=300)
    plt.savefig(save_dir + "/fig4.png", dpi=300)

    plt.clf()


def fig5_6(label="amplitude"):
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of niches"
    """
    labels = [label, "period"]
    y_variable = "survival"
    results = load_results(results_dir, y_variable, labels)

    methods = [float(el) for el in list(set(results[label].to_list()))]
    methods.sort()
    plt.figure(figsize=figsize)
    for method in methods:
        results_ampl = results.loc[results[label] == method]

        sns.lineplot(data=results_ampl, x="period", y="survival", ci=ci, label=short_labels[label] + "=" + str(
            method))

    plt.xlabel("$T_e$, Period of sinusoid")
    plt.ylabel("$A$, Survival")
    plt.legend(loc="best")
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/fig5.pdf", dpi=300)
    plt.savefig(save_dir + "/fig5.png", dpi=300)

    plt.clf()


def fig_7_8_10(include, cut=False):
    """ Plot the evolution of climate and population dynamics.
    """
    results = {}
    projects = [os.path.join(results_dir, o) for o in os.listdir(results_dir)]
    ordered_projects = []
    order = ["_F_G", "_N_G", "_NF_G"]
    for o in order:
        for p in projects:
            if o in p:
                ordered_projects.append(p)
    projects = ordered_projects

    for p in projects:
        if "plots" not in p:
            trial_dirs = list(next(os.walk(p + "/trials"))[1])
            log_niches_total = {}
            log_df = pd.DataFrame()
            for trial_idx, trial_dir in enumerate(trial_dirs):
                try:
                    log = pickle.load(open(p+"/trials/" + trial_dir + '/log_updated.pickle', 'rb'))
                    log_niches = pickle.load(open(p+"/trials/" + trial_dir + '/log_niches.pickle', 'rb'))
                    trial = trial_dir.find("trial_")
                    trial = int(trial_dir[(trial + 6):])
                    if cut:
                        log= log.loc[(log["Generation"] < 370)]
                    if log_df.empty:
                        log_df = log
                    else:
                        log_df = log_df.append(log)
                    log_niches_total[trial] = log_niches

                except IOError:
                    print("No log file for project. ", trial_dir)

            skip_lines = 1
            with open(p + "/config.yml") as f:
                for i in range(skip_lines):
                    _ = f.readline()
                config = yaml.load(f)
                config = SimpleNamespace(**config)


            label = find_label(config)
            results[label] = [log_df, log_niches_total, config]


    plotter = Plotter(project=results_dir,
                      num_niches=config.num_niches,
                      log={},
                      log_niches={},
                      include=include)
    plotter.compare_evolution(results, save_name="compare")

    return

def fig9(label):
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of niches"
    """
    labels = [label, "noise_std"]
    y_variable = "survival"
    results = load_results(results_dir, y_variable, labels)

    methods =  list(set(results[label].to_list()))
    methods.sort()
    plt.figure(figsize=figsize)
    for method in methods:
        results_ampl = results.loc[results[label] == method]

        sns.lineplot(data=results_ampl, x="noise_std", y="survival", ci=ci, label=str(method))

    plt.xlabel(axes_labels["noise_std"])
    plt.ylabel("$A$, Survival")
    plt.legend(loc="upper right")
    save_dir = results_dir + "/plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + "/fig9.pdf", dpi=300)
    plt.savefig(save_dir + "/fig9.png", dpi=300)

    plt.clf()

if __name__ == "__main__":
    # ------ stable climate function -----
    results_dir = "../projects/paper/stable/fig2"
    results_dr = "/scratch/enisioti/climate_log/projects/niche_construction/30_12_2022/stable/F"
    fig2()

    results_dr = "/scratch/enisioti/climate_log/projects/niche_construction/30_12_2022/stable/NF"
    fig2()

    results_dr = "/scratch/enisioti/climate_log/projects/niche_construction/30_12_2022/stable/N"
    fig2()

    results_dir = "../projects/paper/stable/fig3"
    #fig3()

    results_dir = "../projects/paper/stable/fig4"
    #fig4()
    # ---------------------------------------
    # ------ sinusoid climate function -----
    results_dir = "/scratch/enisioti/climate_log/projects/niche_construction/30_12_2022/periodic/F"
    fig5_6(label="amplitude")

    results_dir = "/scratch/enisioti/climate_log/projects/niche_construction/30_12_2022/periodic/NF"
    fig5_6(label="amplitude")

    results_dir = "/scratch/enisioti/climate_log/projects/niche_construction/30_12_2022/periodic/N"
    fig5_6(label="amplitude")


    results_dir = "../projects/paper/sin/fig5"
    #fig5_6(label="amplitude")

    results_dir = "../projects/paper/sin/fig6"
    #fig5_6(label="num_niches")

    results_dir = "../projects/paper/sin/fig7"
    include = ["climate", "mean",
               "sigma", "mutate",
               "dispersal", "diversity"]
    #fig_7_8_10(include)

    results_dir = "../projects/paper/sin/fig8"
    include = ["climate", "mean",
               "sigma", "mutate",
               "dispersal", "diversity"]
    #fig_7_8_10(include)
    # ---------------------------------------
    # ------ noisy climate function -----
    results_dir = "../projects/paper/noisy/fig9"
    #fig9(label="selection")

    results_dir = "../projects/paper/noisy/fig10"
    include = ["climate", "mean",
               "sigma", "mutate",
               "dispersal", "diversity"]
    #fig_7_8_10(include, cut=True)
    # ---------------------------------------

