""" This scripts contains functions for plotting information about an experiment.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pickle
import yaml
from source.utils import compute_SoS, compute_dispersal

class Plotter:

    def __init__(self, project, log={},log_niches={}, num_niches=1, env_profile={}, climate_noconf=False):
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
        self.fig_size_heatmap = (10,10)
        self.fig_size = (10,5)



    def plot_SoS(self):
        """ Plot the evolution of the strength of selection (SoS).

        We also plot the evolution of genes (SD and mutation) to detect patterns in how they interact with SoS
        """
        fig, axs = plt.subplots(1, figsize=self.fig_size)

        log = compute_SoS(self.log, self.log_niches, self.num_niches)

        sns.lineplot(data=log, x="Generation", y="Selection", label="SoS")
        sns.lineplot(data=log, x="Generation", y="SD", label="$\sigma$")
        sns.lineplot(data=log, x="Generation", y="R", label="$r$")
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
            x = log_climate["Generation"]
            y = log_climate["Climate_avg"]
            if self.climate_noconf:
                sns.lineplot(ax=axs[count], x=x, y=y, ci=None)
            else:
                sns.lineplot(ax=axs[count], x=x, y=y, ci=80)
            #axs[count].plot(self.log["Generation"], self.log["Climate_avg"])
            #axs[count].fill_between(x, (y - ci), (y + ci), color='b', alpha=.1)
            axs[count].set(ylabel="$\\bar{e}$")
            axs[count].set(xlabel=None)
            count += 1

        if "mean" in include:
            x = self.log["Generation"]
            y = self.log["Mean"]
            sns.lineplot(ax=axs[count], x=x, y=y, ci=80)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="Mean")
            axs[count].set(ylabel="$\\bar{\mu}$")
            axs[count].set(xlabel=None)
            count += 1

        if "sigma" in include:
            x = self.log["Generation"]
            y = self.log["SD"]
            sns.lineplot(ax=axs[count], x=x, y=y, ci=80)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="SD")
            axs[count].set(ylabel="$\\bar{\sigma}$")
            axs[count].set(xlabel=None)
            axs[count].set_yscale('log')
            count += 1

        if "mutate" in include:
            x = self.log["Generation"]
            y = self.log["R"]
            sns.lineplot(ax=axs[count], x=x, y=y, ci=80)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="R")
            axs[count].set(ylabel="$\\bar{r}$")
            axs[count].set(xlabel=None)
            axs[count].set_yscale('log')
            count += 1

        if "fitness" in include:
            x = self.log["Generation"]
            y = self.log["Fitness"]
            sns.lineplot(ax=axs[count], x=x, y=y, ci=80)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="Fitness")
            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$\\bar{f}$")
            count += 1

        if "extinct" in include:
            x = self.log["Generation"]
            y = self.log["extinctions"]
            sns.lineplot(ax=axs[count], x=x, y=y, ci=80)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="extinctions")
            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="Extinctions")
            count += 1

        if "num_agents" in include:
            x = self.log["Generation"]
            y = self.log["num_agents"]
            sns.lineplot(ax=axs[count], x=x, y=y, ci=80)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="num_agents")
            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$N$, number of agents")
            count += 1
        if "diversity" in include:
            x = self.log["Generation"]
            y = self.log["diversity"]
            sns.lineplot(ax=axs[count], x=x, y=y, ci=80)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="diversity")
            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$V$, diversity")
            count += 1
        if "fixation_index" in include:
            x = self.log["Generation"]
            y = self.log["fixation_index"]
            sns.lineplot(ax=axs[count], x=x, y=y, ci=80)

            #sns.lineplot(ax=axs[count], data=self.log, x="Generation", y="fixation_index")
            axs[count].set(xlabel="Time (in generations)")
            axs[count].set(ylabel="$F_{st}$, fixation_index")
            count += 1
        if "dispersal" in include:

            self.log = compute_dispersal(self.log, self.log_niches, self.num_niches)
            x = self.log["Generation"]
            y = self.log["Dispersal"]
            sns.lineplot(ax=axs[count], x=x, y=y, ci=80)

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


    def plot_heatmap(self, top_dir, x_variables_sin, x_variables_stable, y_variables):
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
                        if config.env_type == "sin":
                            x_variables = x_variables_sin
                            x1 = int(getattr(config, x_variables[0]))
                            x2 = int(getattr(config, x_variables[1]))
                        else:
                            x_variables = x_variables_stable
                            x1 = float("Inf")
                            x1 = config.num_gens
                            x2 = int(getattr(config,x_variables[0] ))

                        x1_values.append(x1)
                        x2_values.append(x2)
                        keep_projects[(x1,x2)] = p
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
                                          + '/log.pickle'):
                            log = pickle.load(open(p + '/trials/trial_' + str(trial)
                                                   + '/log.pickle', 'rb'))
                            if trial:
                                log_df = log_df.append(log)
                            else:
                                log_df = log
                        else:
                            print("sth wron gwith ", p + '/trials/trial_' + str(trial)
                                          + '/log.pickle')
                    locx = x1_unique.index(x1)
                    locy = x2_unique.index(x2)
                    y_values[locx, locy] = np.mean(log_df[y_var])

                    if len(log_df[y_var]) < config.num_gens:
                        y_values[locx, locy] =0

            fig, ax = plt.subplots(figsize=self.fig_size_heatmap)
            extent = [x1_unique[0], x1_unique[-1], x2_unique[0], x2_unique[-1]]

            pos=ax.imshow(y_values)
            fig.colorbar(pos, ax=ax)
            for im in plt.gca().get_images():
                im.set_clim(0, max(np.max(y_values),0.1))
            ax.invert_yaxis()
            ax.set_xticks(list(range(len(x2_unique))))
            ax.set_yticks(list(range(len(x1_unique))))
            #ax.set_yticks(x2_values)
            # ... and label them with the respective list entries
            ax.set_xticklabels(x2_unique)
            ax.set_yticklabels(x1_unique)
            ax.set_ylabel("Transition scaling")
            ax.set_xlabel("$N$, Number of niches")
            ax.set_title(str(y_var))

            save_dir = "../projects/" + top_dir + "/plots"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_dir + "/heatmap_" + str(y_var) + ".png")
            plt.clf()