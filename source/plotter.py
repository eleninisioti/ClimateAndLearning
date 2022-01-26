""" This scripts contains functions for plotting information about an experiment.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle
import yaml
from source.utils import compute_SoS, compute_dispersal
import pandas as pd

labels = {"climate_mean_init": "$\\bar{e}$, Mean climate",
          "num_niches": "$N$, Number of niches",
          "period": "$T$, Period of sin",
          "amplitude": "$A$, Amplitude of sin"}

class Plotter:

    def __init__(self, project, log=pd.DataFrame(),log_niches={}, num_niches=1, env_profile={}, climate_noconf=False):
        """
        Parameters
        ----------
        project: str
            project directory

        env_profile: dict
            contains information about the climate dynamics

        climate_noconf: bool
            If True, we don't plot confidence intervals for climate, we only plot one of the trials
             (useful for highly variable periods)
        """
        self.project = project
        self.env_profile = env_profile
        self.climate_noconf = climate_noconf
        self.log = log
        self.log_niches = log_niches
        self.num_niches = num_niches

        # ----- project-wide plot config -----
        params = {'legend.fontsize': 20,
                  'legend.handlelength': 2,
                  'font.size': 20,
                  "figure.autolayout": True}
        plt.rcParams.update(params)
        self.fig_size_heatmap = (12,12)
        self.fig_size = (10,5)
        if not self.log.empty:

            self.y_upper_thres = 10**(10)
            self.y_lower_thres = 1/(10**(10))
            self.ci=None
            self.interval = int(len(self.log["Climate"])/2000)
            print("printing every", self.interval)
            if self.interval < 2:
                self.interval = 1



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


    def plot_evolution(self,include, cycles=None):
        """ Plot the evolution of climate and population dynamics.

        Parameters
        ----------
        log: dict
            results produced during simulation

        include: list of str
            which parameters to include in the plot. options are ["climate", "mean", "sigma", "mutate",
            "n_agents", "n_extinctions", "fitness"]
        """
        fig, axs = plt.subplots(len(include), figsize=(12, 3*len(include)))
        if include == ["climate"]:
            axs = [axs]

        count = 0
        start_cycle = 0
        end_cycle = cycles
        ci = self.ci
        y_upper_thres = self.y_upper_thres
        y_lower_thres = self.y_lower_thres


        ## TODO: remove
        #self.log["Generation"] = [idx for idx in range(len(self.log["Climate"]))]

        if "climate" in include:
            log_climate = self.log.loc[(self.log['Trial'] == 0)]

            # find mean across niches:
            climate_avg = []
            for el in list(log_climate["Climate"]):
                niches_states = [el + 0.01*idx for idx in range(-int(self.num_niches/2),
                                                                int(self.num_niches/2 +0.5))]
                climate_avg.append(np.mean(niches_states))
            log_climate["Climate_avg"] = climate_avg
            #sns.lineplot(, data=self.log, x="Generation", y="Climate_avg")
            x = log_climate["Generation"][::self.interval]
            y = log_climate["Climate_avg"][::self.interval]
            #y = y.clip(upper=y_upper_thres)
            #y = y.clip(lower=y_lower_thres)


            if self.climate_noconf:
                sns.lineplot(ax=axs[count], x=x, y=y, ci=None)
            else:
                sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)
            #axs[count].plot(self.log["Generation"], self.log["Climate_avg"])
            #axs[count].fill_between(x, (y - ci), (y + ci), color='b', alpha=.1)
            axs[count].set(ylabel="$\\bar{e}$")
            axs[count].set(xlabel=None)
            count += 1

        if "mean" in include:
            x = log_climate["Generation"][::self.interval]
            y = log_climate["Mean"][::self.interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="Mean")
            axs[count].set(ylabel="$\\bar{\mu}$")
            axs[count].set(xlabel=None)
            count += 1

        if "sigma" in include:
            x = log_climate["Generation"][::self.interval]
            y = log_climate["SD"][::self.interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="SD")
            axs[count].set(ylabel="$\\bar{\sigma}$")
            axs[count].set(xlabel=None)
            axs[count].set_yscale('log')
            count += 1

        if "mutate" in include:
            x = log_climate["Generation"][::self.interval]
            y = log_climate["R"][::self.interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="R")
            axs[count].set(ylabel="$\\bar{r}$")
            axs[count].set(xlabel=None)
            axs[count].set_yscale('log')
            count += 1

        if "fitness" in include:
            x = log_climate["Generation"][::self.interval]
            y = log_climate["Fitness"][::self.interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="Fitness")
            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$\\bar{f}$")
            count += 1

        if "extinct" in include:
            x = log_climate["Generation"][::self.interval]
            y = log_climate["extinctions"][::self.interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="extinctions")
            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="Extinctions")
            count += 1

        if "num_agents" in include:
            x = log_climate["Generation"][::self.interval]
            y = log_climate["num_agents"][::self.interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="num_agents")
            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$N$, number of agents")
            count += 1
        if "diversity" in include:
            x = log_climate["Generation"][::self.interval]
            y = log_climate["diversity"][::self.interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="diversity")
            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$V$, diversity")
            count += 1
        if "fixation_index" in include:
            x = log_climate["Generation"][::self.interval]
            y = log_climate["fixation_index"][::self.interval]
            y = y.clip(upper=y_upper_thres)
            y = y.clip(lower=y_lower_thres)

            sns.lineplot(ax=axs[count], x=x, y=y, ci=ci)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="fixation_index")
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

            #sns.lineplot(ax=axs[count], data=log, x="Generation", y="Dispersal")
            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$D$")
            count += 1

        axs[count - 1].set(xlabel="Time (in generations)")

        # ----- highlight periods of variability of applicable -------
        if len(self.env_profile):
            plot_regions = True
        else:
            plot_regions = False

        plot_regions = False

        if plot_regions and end_cycle:
            for subplot in range(count):
                for cycle in range(start_cycle, end_cycle):
                    axs[subplot].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle],
                                         alpha=0.2,
                                         color='gray')
                    axs[subplot].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle],
                                         alpha=0.2,
                                         color='green')
        # -------------------------------------------------------------
        plt.savefig("../projects/" + self.project + "/plots/evolution.png")
        plt.clf()
        return self.log


    def plot_heatmap(self, top_dir, x_variables, y_variables):
        """ Plots a heatmap based on all projects in a directory
        """

        # find all projects in directory
        # find all parameters for x_variables
        projects = [os.path.join("../projects/" +top_dir, o) for o in os.listdir("../projects/" + top_dir)]

        for y_var in y_variables:
            x1_values = []
            x2_values = []
            keep_projects = {}
            for p in projects:
                print(p)
                if "plots" not in p:

                    # load configuration variables
                    config_file = p + "/config.yml"

                    with open(config_file, "rb") as f:
                        config = yaml.load(f, Loader=yaml.UnsafeLoader)
                        if config.num_niches > 200:
                            print("bump")
                            return 0
                        else:

                            x1 = float(getattr(config, x_variables[0]))
                            x2 = float(getattr(config, x_variables[1]))

                            x1_values.append(x1)
                            x2_values.append(x2)
                            keep_projects[(x1,x2)] = p

            # TODO: remove this
            #x1_values =[el for el in x1_values if el <100]
            #x2_values =[el for el in x2_values if el <100]

            x1_unique = list(set(x1_values))
            x1_unique.sort()
            x2_unique = list(set(x2_values))
            x2_unique.sort()
            y_values = np.zeros((len(x1_unique), len(x2_unique)))
            for idx1, x1 in enumerate(x1_unique):
                for idx2, x2 in enumerate(x2_unique):
                    p = keep_projects[(x1, x2)]
                    # load performance variables
                    trial_dirs = list(next(os.walk(p + "/trials"))[1])
                    print(trial_dirs)

                    for trial, trial_dir in enumerate(trial_dirs):
                        if os.path.isfile( p + '/trials/trial_' + str(trial)
                                          + '/log_updated.pickle'):


                            log = pickle.load(open(p + '/trials/trial_' + str(trial)
                                                   + '/log_updated.pickle', 'rb'))
                            if trial:
                                log_df = log_df.append(log)
                            else:
                                log_df = log
                        else:
                            print("sth wron gwith ", p + '/trials/trial_' + str(trial)
                                          + '/log_updated.pickle')
                    locx = x1_unique.index(x1)
                    locy = x2_unique.index(x2)
                    y_values[locx, locy] = np.mean(log_df[y_var])

                    if len(log_df[y_var]) < config.num_gens:
                        y_values[locx, locy] =0

            fig, ax = plt.subplots(figsize=self.fig_size_heatmap)

            pos=ax.imshow(y_values)
            fig.colorbar(pos, ax=ax)
            for im in plt.gca().get_images():
                im.set_clim(0, max(np.max(y_values),0.1))
            #ax.invert_yaxis()
            ax.set_xticks(list(range(len(x2_unique))))
            ax.set_yticks(list(range(len(x1_unique))))
            #ax.set_yticks(x2_values)
            # ... and label them with the respective list entries
            ax.set_xticklabels(x2_unique)
            ax.set_yticklabels(x1_unique)
            ax.set_ylabel(labels[x_variables[0]])
            ax.set_xlabel(labels[x_variables[1]])

            ax.set_title(str(y_var))

            save_dir = "../projects/" + top_dir + "/plots"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_dir + "/heatmap_" + str(y_var) + ".png")
            plt.clf()