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
import pickle
from utils import compute_dispersal, find_index
import numpy as np
import pandas as pd


class Plotter:

    def __init__(self, project, num_niches, log, log_niches):
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

        # ----- global configuration of plots -----
        params = {'legend.fontsize': 6,
                  "figure.autolayout": True,
                  'font.size': 8}
        plt.rcParams.update(params)

        cm = 1 / 2.54  # for converting inches to cm
        self.fig_size = (8.48 * cm, 6 * cm)  # these dimensions chosen to fit in latex column

        self.ci = 95 # for confidence intervals

        # ------------------------------------------
        self.labels = {"climate_mean_init": "$\\bar{e}$, Mean climate",
                       "num_niches": "$N$, Number of niches",
                       "period": "$T$, Period of sinusoid",
                       "amplitude": "$A$, Amplitude of sinusoid"}
        self.label_colors = {"F-selection": "blue", "N-selection": "orange", "NF-selection": "green"}

    def plot_evolution(self, include):
        """ Plot the evolution of climate and population dynamics.

        Parameters
        ----------
        include: list of string
            which evaluation metrics to include. options are ["climate", "mean", "sigma", "mutate",
            "n_agents", "n_extinctions", "fitness"]

        """
        fig, axs = plt.subplots(len(include), figsize=(self.fig_size[0], self.fig_size[1] / 3 * len(include)))
        count = 0
        first_trial = np.min(self.log['Trial'])
        unique_trials = list(set(self.log['Trial']))

        # ----- plot climate curve -----
        if "climate" in include:
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
            sns.lineplot(ax=axs[count], x=x, y=y, ci=None)

            axs[count].set(ylabel="$e_0$")
            axs[count].set(xlabel=None)

            count += 1
        # ----------------------------------------
        # ----- plot average preferred niche -----
        if "mean" in include:
            x = self.log["Generation"]
            y = self.log["Mean"]

            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(ylabel="$\\bar{\mu}$ ")
            axs[count].set(xlabel=None)

            count += 1
        # -----------------------------------
        # ----- plot average plasticity -----
        if "sigma" in include:
            x = self.log["Generation"]
            y = self.log["SD"]

            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(ylabel="$\\bar{\sigma}$")
            axs[count].set(xlabel=None)
            axs[count].set_yscale('log')

            count += 1
        # ------------------------------------
        # ----- plot average evolvability -----
        if "mutate" in include:
            x = self.log["Generation"]
            y = self.log["R"]

            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(ylabel="$\\bar{r}$")
            axs[count].set(xlabel=None)
            axs[count].set_yscale('log')
            count += 1
        # --------------------------------
        # ----- plot average fitness -----

        if "fitness" in include:
            x = self.log["Generation"]
            y = self.log["Fitness"]

            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$\\bar{f}$")
            count += 1
        # ------------------------------------
        # ----- plot average extinctions -----

        if "extinct" in include:
            x = self.log["Generation"]
            y = self.log["extinctions"]

            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$E$")
            count += 1
        # ----------------------------------
        # ----- plot number of agents  -----
        if "num_agents" in include:
            x = self.log["Generation"]
            y = self.log["num_agents"]

            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$N$, number of agents")
            count += 1
        # --------------------------
        # ----- plot genomic diversity -----
        if "diversity" in include:
            x = self.log["Generation"]
            y = self.log["diversity"]

            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$V$")
            count += 1

        # ------------------------------------------
        # ----- plot genomic diversity of preferred state -----
        if "diversity_mean" in include:
            x = self.log["Generation"]
            y = self.log["diversity_mean"]

            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$V_{\mu}$")
            count += 1
        # ------------------------------------------
        # ----- plot genomic diversity of plasticity -----
        if "diversity_sigma" in include:
            x = self.log["Generation"]
            y = self.log["diversity_sigma"]

            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$V_{\sigma}$")
            count += 1
        # ------------------------------------------
        # ----- plot genomic diversity of evolvability  -----
        if "diversity_mutate" in include:
            x = self.log["Generation"]
            y = self.log["diversity_mutate"]

            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$V_{r}$")
            count += 1
        # ------------------------------------------
        # ----- plot fixation index  -----
        if "fixation_index" in include:
            x = self.log["Generation"]
            y = self.log["fixation_index"]

            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$F_{st}$, fixation_index")
            count += 1
        # ------------------------------------------
        # ----- plot dispersal  -----
        if "dispersal" in include:
            self.log = compute_dispersal(self.log, self.log_niches, self.num_niches)
            x = self.log["Generation"]
            y = self.log["Dispersal"]

            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$D$")
            count += 1

        # all sub-plots share the same horizontal axis
        for ax in axs.flat:
            ax.label_outer()
        axs[count - 1].set(xlabel="$G$, Generation")

        # -------------------------------------------------------------
        plt.savefig("../projects/" + self.project + "/plots/evolution_" + str(unique_trials) + ".pdf", dpi=300)
        plt.clf()
        return self.log


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

    for trial_idx, trial_dir in enumerate(trial_dirs):
        try:
            log = pickle.load(open(trial_dir + '/log.pickle', 'rb'))
            log_niches = pickle.load(open(trial_dir + '/log_niches.pickle', 'rb'))

            trial = find_index(trial_dir)
            if log_df.empty:
                log_df = log
            else:
                log_df = log_df.append(log)
            log_niches_total[trial] = log_niches

        except IOError:
            print("No log file for trial: ", trial_dir)

    # ---------------------------------
    # load  project configuration
    skip_lines = 1
    with open(project + "/config.yml") as f:
        for i in range(skip_lines):
            _ = f.readline()
        config = yaml.load(f)

    # choose which evaluation metrics to plot
    if config["only_climate"]:
        include = ["climate"]
    else:
        include = ["climate", "mean", "sigma", "mutate", "dispersal", "diversity", "num_agents", "extinct"]

    if total:
        plotter = Plotter(project=project,
                          num_niches=config["num_niches"],
                          log=log_df,
                          log_niches=log_niches_total)

        log = plotter.plot_evolution(include=include)

    for trial_dir in trial_dirs:
        trial = find_index(trial_dir)
        log_trial = log.loc[(log['Trial'] == trial)]
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
                                  log_niches=log_niches_trial)
                log = plotter.plot_evolution(include=include)


if __name__ == "__main__":

    top_dir = sys.argv[1]  # choose the top directory containing the projects you want to plot (relative path to
    # "../projects")
    total = sys.argv[2]  # if 1 only plot average across trials, if 0 only plot independently for each trial

    projects = [os.path.join("../projects/", top_dir, o) for o in os.listdir("../projects/" + top_dir)]
    for project in projects:
        if "plots" not in project:
            run(project, total)
