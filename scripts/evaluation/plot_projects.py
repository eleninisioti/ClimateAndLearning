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
from scipy import stats
from matplotlib.colors import LogNorm, Normalize

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
        step = 1
        if "dispersal" in self.include:
            self.log = compute_dispersal(self.log, self.log_niches, self.num_niches)

        new_value = self.log[self.log.Generation % step == 0]
        new_value = new_value.reset_index()
        # new_value = value[0].loc[((value[0]["Trial"]) % step)==0]
        self.log = new_value

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

            # add niche construction
            key = list(self.log_niches.keys())[0]
            constructed = self.log_niches[key]
            generations = list(set(self.log["Generation"].tolist()))
            generations.sort()

            if "constructed" in list(constructed.keys()):
                constructed=constructed["constructed"]
                constructed_mean = []
                for gen in generations:
                    constructed_mean.append(np.mean([el[1] for el in constructed[gen]]))
                y = y
            total = [sum(x) for x in zip(y, constructed_mean)]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=None, label="intrinsic")
            sns.lineplot(ax=self.axs[count], x=x, y=total, ci=None, label="total")

            #self.axs[count].ticklabel_format(useOffset=False)
            #self.axs[count].set_ylim(np.log(min(y+total)), np.log(max(y+total)))
            #print(max(constructed_mean), max(y))
            #if max(constructed_mean) > (max(y)+100):
            #    t1 = [float(el) for el in np.geomspace(min(y + total), max(y + total), num=5)]
            #     print(t1)
            #     #self.axs[count].set_yticklabels([float(el) for el in np.geomspace(min(y+total), max(y+total), num=2)])
            #     #self.axs[count].set_yticks(np.geomspace(min(y + total), max(y + total), num=2))
            #
            #     self.axs[count].set_yscale('symlog')
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
        if "construct" in self.include and ("construct" in self.log.keys()):
            x = self.log["Generation"]
            y = self.log["construct"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(ylabel="$c$, construct ")
            self.axs[count].set(xlabel=None)
            #self.axs[count].set_ylim(np.log(0.0000001),np.log(100000))

            #self.axs[count].set_yscale('symlog')


            count += 1
        # -----------------------------------
        # ----- plot average preferred niche -----
        if ("construct_sigma" in self.include) and ("construct_sigma" in self.log.keys()):
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

        if "var_constructed" in self.include:
            x = self.log["Generation"]
            y = self.log["var_constructed"]

            sns.lineplot(ax=self.axs[count], x=x, y=y, ci=self.ci)

            self.axs[count].set(ylabel="$c_{coord}$,\n Coordination in NC ")
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


    def plot_heatmap(self, log, save_dir):
        constructed = []
        trial = list(log.keys())[0]
        log = log[trial]
        for gen_idx, gen in enumerate(log["constructed"]):
            new_constructed= [el[1] for el in gen]
            if len(new_constructed)>100:
                new_constructed = new_constructed[:100]
            elif len(new_constructed) < 100:
                new_constructed = new_constructed + [0 for el in range(100-len(new_constructed)+1)]
            constructed.append(new_constructed)
        constructed_array = np.array(constructed).transpose()
        constructed_array = pd.DataFrame(constructed_array)
        plt.figure(figsize=self.fig_size)
        sns.heatmap(constructed_array,vmin=0,  cmap='Blues')
        plt.xlabel("Generations")
        plt.ylabel("Niche index")
        #plt.colorbar()

        plt.savefig(save_dir + "/plots/meanconstructed_trial_" + str(trial))
        plt.clf()

        constructed = []
        for gen_idx, gen in enumerate(log["var_constructed"]):
            new_constructed = [el[1] for el in gen]
            if len(new_constructed) > 100:
                new_constructed = new_constructed[:100]
            elif len(new_constructed) < 100:
                new_constructed = new_constructed + [0 for el in range(100 - len(new_constructed) + 1)]
            constructed.append(new_constructed)
        constructed_array = np.array(constructed).transpose()
        constructed_array = pd.DataFrame(constructed_array)
        plt.figure(figsize=self.fig_size)
        sns.heatmap(constructed_array, vmin=0, cmap='Reds',norm=LogNorm())
        plt.xlabel("Generations")
        plt.ylabel("Niche index")
        # plt.colorbar()

        plt.savefig(save_dir + "/plots/varconstructed_trial_" + str(trial))
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
    trial_dirs = [el for el in trial_dirs if ".DS" not in el]
    # ---------------------------------
    # load  project configuration
    skip_lines = 1
    with open(project + "/config.yml") as f:
        for i in range(skip_lines):
            _ = f.readline()
        config = yaml.safe_load(f)
    if "niche" in config["genome_type"]:

        for trial_idx, trial_dir in enumerate(trial_dirs):
            print(trial_dir)

            try:
                log = pickle.load(open(trial_dir + '/log.pickle', 'rb'))
                log_niches = pickle.load(open(trial_dir + '/log_niches.pickle', 'rb'))
                #for key, val in log_niches.items():
                    #if key != "constructed" or key != "inhabited_niches":
                    #    log_niches[key] = []
                #log = log.assign(Trial=trial_idx)

                trial_idx = find_index(trial_dir)
                if log_df.empty:
                    log_df = log
                else:
                    log_df = log_df.append(log)
                log_niches_total[trial_idx] = log_niches

                # plot intrinsic motivaation
                if config["genome_type"] == "intrinsic":
                    plot_intrinsic(log["Climate"].tolist(), log_niches, trial_dir)

            except (IOError, EOFError) as e:
                print("No log file for trial: ", trial_dir)


    # choose which evaluation metrics to plot
    if config["only_climate"]:
        include = ["climate"]
    else:
        include = [ "climate","mutate", "dispersal", "diversity",
                   "num_agents",
                   "extinct",]

        if config["genome_type"] != "intrinsic":
            include.append("sigma")
            include.append("mean")

        if config["genome_type"] == "niche-construction" or config["genome_type"] == "niche-construction-v2" :
            include.append("construct")
            include.append("construct_sigma")
            include.append("var_constructed")
            #include.append( "constructed")

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
                print(trial_dir, list(log_niches_total.keys()))
                if trial in list(log_niches_total.keys()):
                    log_niches_trial[trial] = log_niches_total[trial]
                    var_constructed = []
                    for gen in log_niches_trial[trial]["var_constructed"]:
                        var_constructed.append(np.mean([el[1] for el in gen]))
                    log_trial["var_constructed"] = var_constructed
                    if not log_trial.empty:
                        plotter = Plotter(project=project,
                                          num_niches=config["num_niches"],
                                          log=log_trial,
                                          log_niches=log_niches_trial,
                                          include=include)
                        log = plotter.plot_evolution(save_name="trial_" + str(trial))

                    plotter.plot_heatmap(log_niches_trial, project)





if __name__ == "__main__":

    top_dir = sys.argv[1]  # choose the top directory containing the projects you want to plot (relative path to
    # "../projects")
    total = int(sys.argv[2])  # if 1 only plot average across trials, if 0 only plot independently for each trial

    projects = [os.path.join("../projects/", top_dir, o) for o in os.listdir("../projects/" + top_dir)]

    for project in projects:
        if "plots" not in project and ".DS" not in project:
            print(project)
            run(project, total)
