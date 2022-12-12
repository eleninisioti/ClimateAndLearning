""" This script can be used to produce plots for multiple projects under a common directory.

For each project it plots:
* the evolution of climate and population dynamics
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import matplotlib.pyplot as plt
import pickle5 as pickle
import yaml
import math
import seaborn as sns
import numpy as np
from utils import compute_dispersal, find_index
import numpy as np
import pandas as pd
from findpeaks import findpeaks
import pandas as pd
import seaborn as sns
from types import SimpleNamespace
from scripts.evaluation.utils import find_label, load_results, axes_labels, short_labels
from scipy import stats

class Plotter:

    def __init__(self, project, num_niches, log, log_niches, include):
        """ Class constructor.

        Parameters
        ----------
        project: str
            project directory (absolute path)

        num_niches: int
            number of niches
        """
        self.project = project
        self.num_niches = num_niches
        self.log = log
        self.log_niches = log_niches
        self.include = include

        # ----- global configuration of plots -----
        params = {'legend.fontsize': 6,
                  "figure.autolayout": True,
                  'font.size': 8,
                  'pdf.fonttype':42,
                  'ps.fonttype':42}
        plt.rcParams.update(params)

        cm = 1 / 2.54  # for converting inches to cm
        self.fig_size = (8.48 * cm, 6 * cm)  # these dimensions chosen to fit in latex column

        self.ci = 95 # for confidence intervals
        self.fig, self.axs = plt.subplots(len(include), figsize=(self.fig_size[0], self.fig_size[1] / 3 * len(include)))

        # ------------------------------------------
        self.labels = {"climate_mean_init": "$\\bar{e}$, Mean climate",
                       "num_niches": "$N$, Number of niches",
                       "period": "$T$, Period of sinusoid",
                       "amplitude": "$A$, Amplitude of sinusoid"}
        self.label_colors = {"F-selection": "blue", "N-selection": "orange", "NF-selection": "green"}

    def compare_intrinsic(self, results, save_dir):
        for key, log_total in results.items():
            log = log_total[1][0]
            histories = log["histories"]
            curves = log["intrinsic_curves"]
            total_stats = []
            total_p_values = []
            for gen, gen_curves in enumerate(curves):
                gen_histories = histories[gen]
                gen_stats = 0
                gen_p_values = 0

                for agent, agent_curves in enumerate(gen_curves):
                    agent_histories = gen_histories[agent]
                    if len(agent_histories):
                        statistic, p_value = stats.ks_2samp(agent_histories, agent_curves)
                        # plt.hist(agent_histories)
                        # plt.hist(agent_curves)
                        # plt.show()
                        # plt.clf()
                        gen_stats += statistic
                        gen_p_values += p_value
                total_stats.append(gen_stats / (len(gen_curves) - 1))
                total_p_values.append(gen_p_values / (len(gen_curves) - 1))

            plt.plot(range(len(curves)), total_p_values, label=key)
        plt.savefig(save_dir + "/plots/p_values.pdf")
        plt.savefig(save_dir + "/plots/p_values.png")
        plt.xlabel("Generation, $g$")
        plt.ylabel("p-value, $p$")
        plt.clf()

    def compare_evolution(self, results, save_name):
        """ Compare the evolution of climate and population dynamics for different methods.
        """
        count = 0
        step =10
        # ----- plot climate curve -----
        if "climate" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                config = value[2]
                first_trial = np.min(log['Trial'])
                log_trial = log.loc[(log['Trial'] == first_trial)]

                # find mean climate across niches
                climate_avg = []
                for el in list(log_trial["Climate"]):
                    niches_states = [el + 0.01 * idx for idx in range(-int(self.num_niches / 2),
                                                                      int(self.num_niches / 2 + 0.5))]
                    climate_avg.append(np.mean(niches_states))

                log_trial["Climate_avg"] = climate_avg
                x = log_trial["Generation"][::step]
                y = log_trial["Climate_avg"][::step]
                sns.lineplot(ax=self.axs[count], data=log_trial, x="Generation", y="Climate_avg", ci=None, label=label)


            self.axs[count].set(ylabel="$e_0$")
            self.axs[count].set(xlabel=None)
            self.axs[count].get_legend().remove()

            count += 1
        # ----------------------------------------
        # ----- plot average preferred niche -----
        if "mean" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                x = log["Generation"][::step]
                y = log["Mean"][::step]

                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="Mean", ci=self.ci, label=label)

            self.axs[count].set(ylabel="$\\bar{\mu}$ ")
            self.axs[count].set(xlabel=None)
            self.axs[count].set_yscale('log')

            self.axs[count].get_legend().remove()

            count += 1
        # ----------------------------------------
        # ----- plot average preferred niche -----
        if "construct" in self.include and ("construct" in self.log.keys()):
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                x = log["Generation"][::step]
                y = log["construct"][::step]

                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="construct", ci=self.ci, label=label)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$c$,")
            #self.axs[count].set_yscale('log')
            #self.axs[count].set(ylim=(math.pow(10,-10), math.pow(10,5)))



            self.axs[count].get_legend().remove()

            count +=1
        # -----------------------------------
        # ----- plot average preferred niche -----
        if "construct_sigma" in self.include and ("construct_sigma" in self.log.keys()):
            x = self.log["Generation"][::step]
            y = self.log["construct_sigma"][::step]

            sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="construct_sigma", ci=self.ci)

            self.axs[count].set(ylabel="$c_{\sigma}$, variance  \n of construct ")
            self.axs[count].set(xlabel=None)

            count += 1
            # -----------------------------------
        # ----- plot average plasticity -----
        if "sigma" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                x = log["Generation"][::step]
                y = log["SD"][::step]

                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="SD", ci=self.ci, label=label)

            self.axs[count].set(ylabel="$\\bar{\sigma}$")
            self.axs[count].set(xlabel=None)
            self.axs[count].set_yscale('log')
            self.axs[count].get_legend().remove()


            count += 1
        # ------------------------------------
        # ----- plot average evolvability -----
        if "mutate" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                x = log["Generation"][::step]
                y = log["R"][::step]
                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="R", ci=self.ci, label=label)

            self.axs[count].set(ylabel="$\\bar{r}$")
            self.axs[count].set(xlabel=None)
            self.axs[count].set_yscale('log')
            self.axs[count].get_legend().remove()

            count += 1
        # --------------------------------
        # ----- plot average fitness -----
        if "fitness" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                x = log["Generation"][::step]
                y = log["Fitness"][::step]

                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="Fitness", ci=self.ci, label=label)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$\\bar{f}$")
            self.axs[count].get_legend().remove()

            count += 1
        # ------------------------------------
        # ----- plot average extinctions -----

        if "extinct" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                x = log["Generation"][::step]
                y = log["extinctions"][::step]

                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="extinctions", ci=self.ci, label=label)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$E$")
            self.axs[count].get_legend().remove()

            count += 1
        # ----------------------------------
        # ----- plot number of agents  -----
        if "num_agents" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                x = log["Generation"][::step]
                y = log["num_agents"][::step]

                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="num_agents", ci=self.ci, label=label)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$K$, \n number of agents")
            self.axs[count].set_yscale('log')
            self.axs[count].set_ylim((0, 10000000))

            self.axs[count].get_legend().remove()

            count += 1
        # ------------------------------------------
        # ----- plot fixation index  -----
        if "competition" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                x = log["Generation"][::step]
                y = log["competition"][::step]

                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="competition", ci=self.ci, label=label)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="competition")
            self.axs[count].get_legend().remove()
            count +=1
        # --------------------------
        # ----- plot genomic diversity -----
        if "diversity" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                x = log["Generation"][::step]
                y = log["diversity"][::step]

                sns.lineplot(ax=self.axs[count],data=log, x="Generation", y="diversity", ci=self.ci, label=label)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$V$")
            self.axs[count].get_legend().remove()

            count += 1
        # ----- plot dispersal  -----
        if "dispersal" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                x = log["Generation"][::step]
                y = log["Dispersal"][::step]

                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="Dispersal",ci=self.ci, label=label)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$D$")
            self.axs[count].get_legend().remove()

            count += 1


        # all sub-plots share the same horizontal axis
        for ax in self.axs.flat:
            ax.label_outer()
        self.axs[count - 1].set(xlabel="$G$, Generation")

        self.axs[count-2].legend()

        # -------------------------------------------------------------
        save_dir = self.project + "/plots"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + "/evolution_" + save_name + ".pdf", dpi=300)
        plt.savefig(save_dir + "/evolution_" + save_name + ".png", dpi=300)

        plt.clf()
        return self.log


if __name__ == "__main__":
    #results_dir = "../projects/niche_construction/3_10_2022/fig10"
    results_dir = "../projects/" + sys.argv[1]
    parameter = sys.argv[2]

    results = {}
    projects = [os.path.join(results_dir, o) for o in os.listdir(results_dir)]
    ordered_projects = []

    if parameter == "selection":
        order = ["_F_G", "_N_G", "_NF_G"]
    else:
        order = ["evolv", "niche_construct"]
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
                    log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))
                    log_niches = pickle.load(open(p + "/trials/" + trial_dir + '/log_niches.pickle', 'rb'))
                    trial = trial_dir.find("trial_")
                    #trial = int(trial_dir[(trial + 6):])

                    if log_df.empty:
                        log_df = log
                    else:
                        log_df = log_df.append(log)
                    log_niches_total[trial] = log_niches

                except (IOError,EOFError) as e  :
                    print("No log file for project. ", trial_dir)

            skip_lines = 1
            with open(p + "/config.yml") as f:
                for i in range(skip_lines):
                    _ = f.readline()
                config = yaml.load(f)
                #config = SimpleNamespace(**config)

            label = find_label(SimpleNamespace(**config), parameter)
            results[label] = [log_df, log_niches_total, config]

    if config["only_climate"]:
        include = ["climate"]
    else:
        include = ["climate","mutate",  "num_agents"]

        if config["genome_type"] != "intrinsic":
            include.append("sigma")
            include.append("mean")


        if config["genome_type"] == "niche-construction" and parameter !="genome":
            include.append("construct")
            include.append("construct_sigma")
            #include.append( "constructed")


    plotter = Plotter(project=results_dir,
                      num_niches=config["num_niches"],
                      log={},
                      log_niches={},
                      include=include)
    plotter.compare_evolution(results, save_name="compare_select")
    #plotter.compare_intrinsic(results, results_dir)
