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
import seaborn as sns
import numpy as np
from utils import compute_dispersal, find_index
import numpy as np
import pandas as pd
from findpeaks import findpeaks
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

    def compare_evolution(self, results, save_name):
        """ Compare the evolution of climate and population dynamics for different methods.
        """
        count = 0
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
                x = log_trial["Generation"]
                y = log_trial["Climate_avg"]
                sns.lineplot(ax=self.axs[count], x=x, y=y, ci=None, label=label)


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
                x = log["Generation"]
                y = log["Mean"]

                sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci, label=label)

            self.axs[count].set(ylabel="$\\bar{\mu}$ ")
            self.axs[count].set(xlabel=None)
            self.axs[count].get_legend().remove()

            count += 1
        # -----------------------------------
        # ----- plot average plasticity -----
        if "sigma" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                x = log["Generation"]
                y = log["SD"]

                sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci, label=label)

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
                x = log["Generation"]
                y = log["R"]
                sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci, label=label)

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
                x = log["Generation"]
                y = log["Fitness"]

                sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci, label=label)

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
                x = log["Generation"]
                y = log["extinctions"]

                sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci, label=label)

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
                x = log["Generation"]
                y = log["num_agents"]

                sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci, label=label)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$P$, \n number of agents")
            self.axs[count].get_legend().remove()

            count += 1
        # --------------------------
        # ----- plot genomic diversity -----
        if "diversity" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                x = log["Generation"]
                y = log["diversity"]

                sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci, label=label)

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
                x = log["Generation"]
                y = log["Dispersal"]

                sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci, label=label)

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


    def plot_evolution(self, save_name):
        """ Plot the evolution of climate and population dynamics.

        Parameters
        ----------
        include: list of string
            which evaluation metrics to include. options are ["climate", "mean", "sigma", "mutate",
            "n_agents", "n_extinctions", "fitness"]

        save_name: string
            this string will be appended at the end of the plot name

        """
        count = 0
        first_trial = np.min(self.log['Trial'])
        unique_trials = list(set(self.log['Trial']))

        # ----- plot climate curve -----
        if "climate" in self.include:
            log_trial = self.log.loc[(self.log['Trial'] == first_trial)] # only plot the first trial of climate

            # find mean climate across niches
            climate_avg = []
            for el in list(log_trial["Climate"]):
                niches_states = [el + 0.01 * idx for idx in range(-int(self.num_niches / 2),
                                                                  int(self.num_niches / 2 + 0.5))]
                climate_avg.append(np.mean(niches_states))

            log_trial["Climate_avg"] = climate_avg
            x = log_trial["Generation"]
            y = log_trial["Climate_avg"]
            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=None)

            self.axs[count].set(ylabel="$e_0$")
            self.axs[count].set(xlabel=None)

            count += 1
        # ----------------------------------------
        # ----- plot average preferred niche -----
        if "mean" in self.include:
            x = self.log["Generation"]
            y = self.log["Mean"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(ylabel="$\\bar{\mu}$ ")
            self.axs[count].set(xlabel=None)

            count += 1
        # ----------------------------------------
        # ----- plot average preferred niche -----
        if "construct" in self.include:
            x = self.log["Generation"]
            y = self.log["construct"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(ylabel="$c$, construct ")
            self.axs[count].set(xlabel=None)

            count += 1
        # -----------------------------------
        # ----- plot average preferred niche -----
        if "construct_sigma" in self.include:
            x = self.log["Generation"]
            y = self.log["construct_sigma"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(ylabel="$c_{\sigma}$, variance  \n of construct ")
            self.axs[count].set(xlabel=None)

            count += 1
        # -----------------------------------
        # ----- plot capacity -----
        if "constructed" in self.include:
            x = self.log["Generation"]
            y = self.log["constructed"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(ylabel="$C_{total}$,\n World construction ")
            self.axs[count].set(xlabel=None)
            #self.axs[count].set_yscale('log')


            count += 1
        # -----------------------------------

        # ----- plot average plasticity -----
        if "sigma" in self.include:
            x = self.log["Generation"]
            y = self.log["SD"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(ylabel="$\\bar{\sigma}$")
            self.axs[count].set(xlabel=None)
            self.axs[count].set_yscale('log')

            count += 1
        # ------------------------------------
        # ----- plot average evolvability -----
        if "mutate" in self.include:
            x = self.log["Generation"]
            y = self.log["R"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(ylabel="$\\bar{r}$")
            self.axs[count].set(xlabel=None)
            self.axs[count].set_yscale('log')
            count += 1
        # --------------------------------
        # ----- plot average fitness -----

        if "fitness" in self.include:
            x = self.log["Generation"]
            y = self.log["Fitness"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$\\bar{f}$")
            count += 1
        # ------------------------------------
        # ----- plot average extinctions -----

        if "extinct" in self.include:
            x = self.log["Generation"]
            y = self.log["extinctions"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$E$")
            count += 1
        # ----------------------------------
        # ----- plot number of agents  -----
        if "num_agents" in self.include:
            x = self.log["Generation"]
            y = self.log["num_agents"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$N$, number of agents")
            count += 1
        # --------------------------
        # ----- plot genomic diversity -----
        if "diversity" in self.include:
            x = self.log["Generation"]
            y = self.log["diversity"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$V$")
            count += 1

        # ------------------------------------------
        # ----- plot genomic diversity of preferred state -----
        if "diversity_mean" in self.include:
            x = self.log["Generation"]
            y = self.log["diversity_mean"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$V_{\mu}$")
            count += 1
        # ------------------------------------------
        # ----- plot genomic diversity of plasticity -----
        if "diversity_sigma" in self.include:
            x = self.log["Generation"]
            y = self.log["diversity_sigma"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$V_{\sigma}$")
            count += 1
        # ------------------------------------------
        # ----- plot genomic diversity of evolvability  -----
        if "diversity_mutate" in self.include:
            x = self.log["Generation"]
            y = self.log["diversity_mutate"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$V_{r}$")
            count += 1
        # ------------------------------------------
        # ----- plot fixation index  -----
        if "fixation_index" in self.include:
            x = self.log["Generation"]
            y = self.log["fixation_index"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$F_{st}$, fixation_index")
            count += 1

        # ------------------------------------------
        # ----- plot fixation index  -----
        if "competition" in self.include:
            x = self.log["Generation"]
            y = self.log["competition"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$C$, competition")
            count += 1
        # ------------------------------------------
        # ----- plot fixation index  -----
        if "not_reproduced" in self.include:
            x = self.log["Generation"]
            y = self.log["not_reproduced"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$\\bar{R}$, Not reproduced")
            count += 1
        # ------------------------------------------
        # ----- plot dispersal  -----
        if "dispersal" in self.include:
            self.log = compute_dispersal(self.log, self.log_niches, self.num_niches)
            x = self.log["Generation"]
            y = self.log["Dispersal"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$D$")
            count += 1

        # all sub-plots share the same horizontal axis
        for ax in self.axs.flat:
            ax.label_outer()
        self.axs[count - 1].set(xlabel="$G$, Generation")

        # -------------------------------------------------------------
        save_dir = self.project + "/plots"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + "/evolution_" + save_name + ".pdf", dpi=300)
        plt.savefig(save_dir + "/evolution_" + save_name + ".png", dpi=300)

        plt.clf()
        return self.log


def plot_intrinsic_old(climate, log, save_dir):

    curves = log["intrinsic_curves"]
    means = []
    sigmas = []
    modes = []
    if not os.path.exists(save_dir + "/plots/histograms"):
        os.makedirs(save_dir + "/plots/histograms")

    for gen, gen_curves in enumerate(curves):
        gen_mean = 0
        gen_sigma= 0
        gen_modes = 0
        print("gneeration", gen, "number of agents", len(gen_curves))
        for agent, agent_curves in enumerate(gen_curves):
            n, bins, x = plt.hist(agent_curves)
            mean_histogram =np.mean(n)
            gen_modes += len([el for  el in n if el > mean_histogram/2])
            gen_mean += np.mean(agent_curves)
            gen_sigma += np.var(agent_curves)
            if agent%50==0:
                plt.savefig(save_dir + "/plots/histograms/intrinsic_gen_" + str(gen) + "_agent_"+ str(agent) +".png")

            plt.clf()

        means.append(gen_mean/len(gen_curves))
        sigmas.append(gen_sigma / len(gen_curves))
        modes.append(gen_modes/len(gen_curves))


    climate = climate[:len(means)]

    fig, axs = plt.subplots(4, sharex=True)
    axs[0].plot(range(len(climate)), climate)
    axs[0].set(ylabel="Climate, $e$")
    axs[1].plot(range(len(climate)), means)
    axs[1].set(ylabel="Mean of $i$")
    axs[2].plot(range(len(climate)), sigmas)
    axs[2].set(ylabel="Var of $i$")
    axs[3].plot(range(len(climate)), modes)
    axs[3].set(ylabel="Modes of $i$")
    axs[3].set(xlabel="Generation, $g$")

    plt.savefig(save_dir + "/plots/intrinsic.pdf")
    plt.savefig(save_dir + "/plots/intrinsic.png")


def plot_intrinsic(climate, log, save_dir):


    # ----- plot evolution of tolerance curves ------


    curves = log["intrinsic_curves"]

    total_data = []
    agents_data = {}

    min_climate_axis = int(np.floor(min(climate))) -1
    max_climate_axis = int(np.ceil(min(climate))) +1

    climate_axis = np.arange(min_climate_axis, max_climate_axis, (max_climate_axis-min_climate_axis)/20)
    climate_axis = np.asarray([np.round(el, 2) for el in climate_axis])

    for gen, gen_curves in enumerate(curves):
        print("gen is", gen)

        total_gen_curves = [0]*20
        for agent, agent_curves in enumerate(gen_curves):
            if agent%10 == 0:
                print("agentn is", gen)
                agent_transform = [0]*20

                for el in agent_curves:
                    i = (np.abs(climate_axis - el)).argmin()
                    total_gen_curves[i] += 1
                    agent_transform[i] += 1

                #total_gen_curves = [el + agent_curves[idx] for idx, el in enumerate(total_gen_curves)]

                if agent in agents_data.keys():
                    agents_data[agent].append(agent_transform)
                else:
                    agents_data[agent] = [agent_transform]

        total_data.append(total_gen_curves)

    sns.heatmap(np.transpose(np.array(total_data)), yticklabels=climate_axis)
    plt.xlabel("Generation")
    plt.ylabel("TC mean")
    plt.savefig(save_dir + "/plots/total_heatmap.pdf")
    plt.savefig(save_dir + "/plots/total_heatmap.png")
    plt.clf()

    for agent, agent_data in agents_data.items():
        agent_array = np.transpose(np.array(agent_data))
        sns.heatmap(agent_array, yticklabels=climate_axis)
        plt.xlabel("Generation")
        plt.ylabel("TC mean")
        plt.savefig(save_dir + "/plots/heatmap_" + str(agent) + ".pdf")
        plt.savefig(save_dir + "/plots/heatmap_" + str(agent) + ".png")
        plt.clf()

    # -------------------------------
    # ----- plot evolution of disagreement ------
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
                statistic, p_value= stats.ks_2samp(agent_histories, agent_curves)
                #plt.hist(agent_histories)
                #plt.hist(agent_curves)
                #plt.show()
                #plt.clf()
                gen_stats += statistic
                gen_p_values += p_value
        total_stats.append(gen_stats/(len(gen_curves)-1))
        total_p_values.append(gen_p_values/(len(gen_curves)-1))

    plt.plot(range(len(curves)), total_stats)
    plt.xlabel("Generation, $g$")
    plt.ylabel("p-value, $p$")
    plt.savefig(save_dir + "/plots/stats.pdf")
    plt.savefig(save_dir + "/plots/stats.png")
    plt.clf()

    plt.plot(range(len(curves)), total_p_values)
    plt.savefig(save_dir + "/plots/p_values.pdf")
    plt.savefig(save_dir + "/plots/p_values.png")
    plt.clf()






def run(project, total):
    """ Produce plots for a single project.

    Parameters
    ----------
    project: str
        name of project directory (absolute path)

    total: int
        if 1 only plot average across trials, if 0 only plot independently for each trial

    """
    # ----- collect data from each trial -----
    log_df = pd.DataFrame()
    log_niches_total = {}

    trial_dirs = [os.path.join(project + "/trials", o) for o in os.listdir(project + "/trials")]

    # ---------------------------------
    # load  project configuration
    skip_lines = 1
    with open(project + "/config.yml") as f:
        for i in range(skip_lines):
            _ = f.readline()
        config = yaml.load(f)


    for trial_idx, trial_dir in enumerate(trial_dirs):
        try:
            log = pickle.load(open(trial_dir + '/log.pickle', 'rb'))
            log_niches = pickle.load(open(trial_dir + '/log_niches.pickle', 'rb'))
            #log = log.assign(Trial=trial_idx)

            trial = find_index(trial_dir)
            if log_df.empty:
                log_df = log
            else:
                log_df = log_df.append(log)
            log_niches_total[trial_idx] = log_niches

            # plot intrinsic motivaation
            if config["genome_type"] == "intrinsic":
                plot_intrinsic(log["Climate"].tolist(), log_niches, trial_dir)

        except IOError:
            print("No log file for trial: ", trial_dir)


    # choose which evaluation metrics to plot
    if config["only_climate"]:
        include = ["climate"]
    else:
        include = ["competition", "climate","mutate", "dispersal", "diversity",
                   "num_agents",
                   "extinct",]

        if config["genome_type"] != "intrinsic":
            include.append("sigma")
            include.append("mean")

        if config["genome_type"] == "niche-construction":
            include.append("construct")
            include.append("construct_sigma")
            include.append( "constructed")

    if not log_df.empty:
        if total:
            plotter = Plotter(project=project,
                              num_niches=config["num_niches"],
                              log=log_df,
                              log_niches=log_niches_total,
                              include=include)

            log = plotter.plot_evolution(save_name="_total")

        for trial_dir in trial_dirs:
            trial = find_index(trial_dir)
            log_trial = log_df.loc[(log_df['Trial'] == trial)]
            if total:
                # save new log data produced by plotter (includes dispersal)
                pickle.dump(log_trial, open(trial_dir + '/log_updated.pickle', 'wb'))
            else:
                # plot only for this trial and don't save
                log_niches_trial = {}
                log_niches_trial[trial] = log_niches_total[trial]
                if not log_trial.empty:
                    plotter = Plotter(project=project,
                                      num_niches=config["num_niches"],
                                      log=log_trial,
                                      log_niches=log_niches_trial,
                                      include=include)
                    log = plotter.plot_evolution(save_name="trial_" + str(trial))





if __name__ == "__main__":

    top_dir = sys.argv[1]  # choose the top directory containing the projects you want to plot (relative path to
    # "../projects")
    total = int(sys.argv[2])  # if 1 only plot average across trials, if 0 only plot independently for each trial

    projects = [os.path.join("../projects/", top_dir, o) for o in os.listdir("../projects/" + top_dir)]
    for project in projects:
        if "plots" not in project:
            run(project, total)
