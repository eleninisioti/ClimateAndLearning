""" This script can be used to produce plots for multiple projects under a common directory.

For each project it plots:
* the evolution of climate and population dynamics
* the SoS, Strengh of Selection plot
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
from utils import compute_SoS, compute_dispersal
import numpy as np

labels = {"climate_mean_init": "$\\bar{e}$, Mean climate",
          "num_niches": "$N$, Number of niches",
          "period": "$T$, Period of sinusoid",
          "amplitude": "$A$, Amplitude of sinusoid"}


class Plotter:

    def __init__(self, project, num_niches, log, log_niches):
        """ Class constructor
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
        include = ["climate", "mean",
                   "sigma", "mutate", "extinct",
                   "dispersal", "diversity"]
        cm = 1 / 2.54  # convert inches to cm
        scale = 1
        self.fig_size = (8.48 * cm / scale, 6 * cm / scale)  # these dimensions chosen to fit in latex column
        fig, self.axs = plt.subplots(len(include), figsize=(self.fig_size[0], self.fig_size[1] / 2 * len(include)))
        if include == ["climate"]:
            self.axs = [self.axs]  # the simulations don't include a population
        self.ci = 95
        self.y_upper_thres = 10 ** 10
        self.y_lower_thres = 1 / (10 ** 10)

        if not self.log.empty:
            # plot has at most 2000 points
            self.interval = int(len(self.log["Climate"]) / 2000)
            if self.interval < 2:
                self.interval = 1
        # ------------------------------------------

    def plot_SoS(self):
        """ Plot the evolution of the strength of selection (SoS).

        We also plot the evolution of genes (SD and mutation) to detect patterns in how they interact with SoS
        """
        fig, axs = plt.subplots(1, figsize=self.fig_size)

        log = compute_SoS(self.log, self.log_niches, self.num_niches)
        x = log["Generation"][::self.interval]
        y = log["Selection"][::self.interval]
        y = y.clip(upper=self.y_upper_thres)
        y = y.clip(lower=self.y_lower_thres)
        sns.lineplot(data=log, x=x, y=y, label="SoS", ci=self.ci)

        y = log["SD"][::self.interval]
        y = y.clip(upper=self.y_upper_thres)
        y = y.clip(lower=self.y_lower_thres)
        sns.lineplot(data=log, x=x, y=y, label="$\sigma$", ci=self.ci)

        y = log["R"][::self.interval]
        y = y.clip(upper=self.y_upper_thres)
        y = y.clip(lower=self.y_lower_thres)
        sns.lineplot(data=log, x=x, y=y, label="$r$", ci=self.ci)

        axs.set(xlabel="Time (in generations)")
        axs.set_yscale('log')

        plt.savefig("../projects/" + self.project + "/plots/SoS.png")
        plt.clf()

    def plot_evolution(self, include):
        """ Plot the evolution of climate and population dynamics.

        Parameters
        ----------


        include: list of str
            which parameters to include in the plot. options are ["climate", "mean", "sigma", "mutate",
            "n_agents", "n_extinctions", "fitness"]
        """
        fig, axs = plt.subplots(len(include), figsize=(self.fig_size[0], self.fig_size[1] / 3 * len(include)))
        count = 0

        # ----- plot climate curve -----
        if "climate" in include:
            log_trial = self.log.loc[(self.log['Trial'] == 0) ]

            # find mean climate across niches:
            climate_avg = []
            for el in list(log_trial["Climate"]):
                niches_states = [el + 0.01 * idx for idx in range(-int(self.num_niches / 2),
                                                                  int(self.num_niches / 2 + 0.5))]
                climate_avg.append(np.mean(niches_states))

            log_trial["Climate_avg"] = climate_avg
            x = log_trial["Generation"][::self.interval]
            y = log_trial["Climate_avg"][::self.interval]
            sns.lineplot(ax=axs[count], x=x, y=y, ci=None)

            axs[count].set(ylabel="$e_0$")
            axs[count].set(xlabel=None)
            count += 1
        # ----------------------------------------
        # ----- plot average preferred niche -----
        if "mean" in include:
            x = self.log["Generation"][::self.interval]
            y = self.log["Mean"][::self.interval]
            y = y.clip(upper=self.y_upper_thres)
            y = y.clip(lower=self.y_lower_thres)
            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(ylabel="$\\bar{\mu}$ ")
            axs[count].set(xlabel=None)
            count += 1
        # -----------------------------------
        # ----- plot average plasticity -----
        if "sigma" in include:
            x = self.log["Generation"][::self.interval]
            y = self.log["SD"][::self.interval]
            y = y.clip(upper=self.y_upper_thres)
            #y = y.clip(lower=self.y_lower_thres)
            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(ylabel="$\\bar{\sigma}$")
            axs[count].set(xlabel=None)
            axs[count].set_yscale('log')
            count += 1
        # ------------------------------------
        # ----- plot average evolvability -----
        if "mutate" in include:
            x = self.log["Generation"][::self.interval]
            y = self.log["R"][::self.interval]
            #y = y.clip(upper=self.y_upper_thres)
            #y = y.clip(lower=self.y_lower_thres)
            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(ylabel="$\\bar{r}$")
            axs[count].set(xlabel=None)
            axs[count].set_yscale('log')
            count += 1
        # --------------------------------
        # ----- plot average fitness -----

        if "fitness" in include:
            x = self.log["Generation"][::self.interval]
            y = self.log["Fitness"][::self.interval]
            y = y.clip(upper=self.y_upper_thres)
            y = y.clip(lower=self.y_lower_thres)
            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$\\bar{f}$")
            count += 1
        # ------------------------------------
        # ----- plot average extinctions -----

        if "extinct" in include:
            x = self.log["Generation"][::self.interval]
            y = self.log["extinctions"][::self.interval]
            y = y.clip(upper=self.y_upper_thres)
            y = y.clip(lower=self.y_lower_thres)
            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$E$")
            count += 1
        # ----------------------------------
        # ----- plot number of agents  -----
        if "num_agents" in include:
            x = self.log["Generation"][::self.interval]
            y = self.log["num_agents"][::self.interval]
            y = y.clip(upper=self.y_upper_thres)
            y = y.clip(lower=self.y_lower_thres)
            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$N$, number of agents")
            count += 1

        # --------------------------
        # ----- plot genomic diversity -----

        if "diversity" in include:
            x = self.log["Generation"][::self.interval]
            y = self.log["diversity"][::self.interval]
            y = y.clip(upper=self.y_upper_thres)
            y = y.clip(lower=self.y_lower_thres)
            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$V$")
            count += 1

        # ------------------------------------------
        # ----- plot genomic diversity of preferred state -----
        if "diversity_mean" in include:
            x = self.log["Generation"][::self.interval]
            y = self.log["diversity_mean"][::self.interval]
            y = y.clip(upper=self.y_upper_thres)
            y = y.clip(lower=self.y_lower_thres)
            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$V_{\mu}$")
            count += 1
        # ------------------------------------------
        # ----- plot genomic diversity of plasticity -----

        if "diversity_sigma" in include:
            x = self.log["Generation"][::self.interval]
            y = self.log["diversity_sigma"][::self.interval]
            y = y.clip(upper=self.y_upper_thres)
            y = y.clip(lower=self.y_lower_thres)
            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$V_{\sigma}$")
            count += 1

        # ------------------------------------------
        # ----- plot genomic diversity of evolvability  -----
        if "diversity_mutate" in include:
            x = self.log["Generation"][::self.interval]
            y = self.log["diversity_mutate"][::self.interval]
            y = y.clip(upper=self.y_upper_thres)
            y = y.clip(lower=self.y_lower_thres)
            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$V_{r}$")
            count += 1

        # ------------------------------------------
        # ----- plot fixation index  -----
        if "fixation_index" in include:
            x = self.log["Generation"][::self.interval]
            y = self.log["fixation_index"][::self.interval]
            y = y.clip(upper=self.y_upper_thres)
            y = y.clip(lower=self.y_lower_thres)
            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$F_{st}$, fixation_index")
            count += 1

        # ------------------------------------------
        # ----- plot dispersal  -----
        if "dispersal" in include:
            self.log = compute_dispersal(self.log, self.log_niches, self.num_niches)
            x = self.log["Generation"][::self.interval]
            y = self.log["Dispersal"][::self.interval]
            y = y.clip(upper=self.y_upper_thres)
            y = y.clip(lower=self.y_lower_thres)
            sns.lineplot(ax=axs[count], x=x, y=y, ci=self.ci)

            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$D$")
            count += 1
        # all sub-plots share the same horizontal axis
        for ax in axs.flat:
            ax.label_outer()
        axs[count - 1].set(xlabel="$G$, Generation")

        # -------------------------------------------------------------
        plt.savefig("../projects/" + self.project + "/plots/evolution.pdf", dpi=300)
        plt.clf()
        return self.log


def run(project):
    """ Produce plots for a single project.

    Args:

    """
    trial_dirs = [os.path.join(project + "/trials",o) for o in os.listdir(project + "/trials")]
    for trial, trial_dir in enumerate(trial_dirs):
        file_exists = os.path.exists(trial_dir + '/log_updated.pickle')
        if file_exists:
            # the project has already been plotted
            return

        # ----- load outcome of trial -----
        try:
            log = pickle.load(open(trial_dir + '/log.pickle', 'rb'))
            log_niches = pickle.load(open(trial_dir + '/log_niches.pickle', 'rb'))

        except IOError:
            print("No log file for project. ")
            return 1

        if trial == 0:
            log_df = log
            log_niches_total = [log_niches]
        else:
            log_df = log_df.append(log)
            log_niches_total.append(log_niches)
        # ---------------------------------

    # load  project configuration
    skip_lines = 1
    with open(project + "/config.yml") as f:
        for i in range(skip_lines):
            _ = f.readline()
        config = yaml.load(f)
    # choose which metrics to plot
    if config["only_climate"]:
        include = ["climate"]
    else:
        include = ["climate", "mean", "sigma", "mutate", "dispersal", "diversity","num_agents"]
    plotter = Plotter(project=project,
                      num_niches=config["num_niches"],
                      log=log_df,
                      log_niches=log_niches_total)
    log = plotter.plot_evolution(include=include)

    # save new log data produced by plotter
    for trial, trial_dir in enumerate(trial_dirs):
        log_trial = log.loc[(log['Trial'] == trial)]
        pickle.dump(log_trial, open(trial_dir + '/log_updated.pickle', 'wb'))


if __name__ == "__main__":
    top_dir = sys.argv[1]  # choose the top directory containing the projects you want to plot
    projects = [os.path.join("../projects/", top_dir, o) for o in os.listdir("../projects/" + top_dir)]
    for project in projects:
        if "plots" not in project:
            run(project)
