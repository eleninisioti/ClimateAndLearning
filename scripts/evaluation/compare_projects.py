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
                  'font.size': 6,
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
        step = 10
        num_yticks= 5
        import matplotlib.ticker as plticker


        for key, value in results.items():
            if "dispersal" in self.include:
                value[0] = compute_dispersal(value[0], value[1], self.num_niches)
            new_value = value[0][value[0].Generation % step == 0]
            new_value = new_value.reset_index()
            #new_value = value[0].loc[((value[0]["Trial"]) % step)==0]
            results[key][0] = new_value
        # ----- plot climate curve -----
        if "climate" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                config = value[2]

                # add mean niche construction
                constructed = value[1]
                total_constructed = []
                total_generations = []
                total_climate = []

                for trial_idx, trial in constructed.items():
                    climate_trial = log.loc[log["Trial"]==trial_idx]
                    #mean_climate = log.groupby('Generation')["Climate"].mean().tolist()
                    generations = list(set(log["Generation"].tolist()))
                    generations.sort()
                    #total_generations.extend(generations)

                    #generations =generations[0::10]


                    if "constructed" in trial.keys():
                        trial["constructed"] = [trial["constructed"][el] for el in range(len(trial["constructed"])) if el < max(generations)]
                        for gen_idx, gen in enumerate(trial["constructed"][0::10]):
                            gen_idx = gen_idx*10
                            print(max(climate_trial["Generation"]), gen_idx)
                            if gen_idx <= max(climate_trial["Generation"]):
                                current_climate = climate_trial.loc[climate_trial["Generation"]==gen_idx]
                                current_climate_list = current_climate["Climate"].tolist()
                                total_constructed.append( np.mean([el[1] for el in gen]) )
                                total_climate.append(current_climate_list[0])
                                total_generations.append(gen_idx)

                if "constructed" in trial.keys():

                    #constructed_mean = [el/len(constructed) for el in constructed_mean]


                    climate_and_construct = [sum(x) for x in zip(total_climate, total_constructed)]
                else:
                    climate_and_construct = total_climate
                print(len(total_generations))
                print(len(climate_and_construct))
                print(len(total_climate))
                print(len(total_constructed ))


                log_climate = pd.DataFrame({'Generation': total_generations,
                                            "climate_and_construct": climate_and_construct})
                #sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="Climate", ci=None, label=label)
                sns.lineplot(ax=self.axs[count], data=log_climate, x="Generation", y="climate_and_construct", ci=self.ci, label=label)


            self.axs[count].ticklabel_format(useOffset=False)

            self.axs[count].set(ylabel="$e_0$, \n climate")
            self.axs[count].set(xlabel=None)
            self.axs[count].get_legend().remove()

            count += 1
            print("plotted climate")
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

                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="Mean", ci=self.ci, label=label)

            self.axs[count].set(ylabel="$\mathbb{E}(\mu)$, \npreferred state")
            self.axs[count].set(xlabel=None)
            self.axs[count].set_yscale('log')
            loc = plticker.LogLocator(base=10.0, numticks=num_yticks)  # this locator puts ticks at regular intervals
            self.axs[count].yaxis.set_major_locator(loc)

            self.axs[count].get_legend().remove()

            count += 1
            print("plotted mean")

        # ----- plot average preferred niche -----
        if ("constructed" in self.include) :
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                if ("constructed" in list(log.keys())):
                    x = log["Generation"]
                    y = log["constructed"]

                    sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="constructed", ci=self.ci, label=label)

            if ("constructed" in list(log.keys())):

                self.axs[count].set(xlabel="Time (in generations)")
                self.axs[count].set(ylabel="$C$, \n constructed")
            loc = plticker.LinearLocator(numticks=num_yticks)  # this locator puts ticks at regular intervals
            self.axs[count].yaxis.set_major_locator(loc)
            #self.axs[count].set_yscale('log')
            #self.axs[count].set(ylim=(math.pow(10,-10), math.pow(10,5)))



            self.axs[count].get_legend().remove()

            count +=1
            print("plotted construct")

        if ("var_constructed" in self.include):
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                if ("var_constructed" in list(log.keys())):
                    x = log["Generation"]
                    y = log["var_constructed"]

                    sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="var_constructed", ci=self.ci, label=label)

            if ("var_constructed" in list(log.keys())):
                self.axs[count].set(xlabel="Time (in generations)")
                self.axs[count].set(ylabel="$c_{coord}$,\n Coordination in NC ")
            loc = plticker.LinearLocator(numticks=num_yticks)  # this locator puts ticks at regular intervals
            self.axs[count].yaxis.set_major_locator(loc)
            # self.axs[count].set_yscale('log')
            # self.axs[count].set(ylim=(math.pow(10,-10), math.pow(10,5)))

            self.axs[count].get_legend().remove()

            count += 1
            print("plotted construct")

        # ----------------------------------------
        # ----- plot average preferred niche -----
        if ("construct" in self.include) :
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                if ("construct" in list(log.keys())):
                    x = log["Generation"]
                    y = log["construct"]

                    sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="construct", ci=self.ci, label=label)

            if ("construct" in list(log.keys())):

                self.axs[count].set(xlabel="Time (in generations)")
                self.axs[count].set(ylabel="$E(c)$, \n mean construction")
            loc = plticker.LinearLocator(numticks=num_yticks)  # this locator puts ticks at regular intervals
            self.axs[count].yaxis.set_major_locator(loc)
            #self.axs[count].set_yscale('log')
            #self.axs[count].set(ylim=(math.pow(10,-10), math.pow(10,5)))



            self.axs[count].get_legend().remove()

            count +=1
            print("plotted construct")

        # -----------------------------------
        # ----- plot average preferred niche -----
        print(self.include)
        if "construct_sigma" in self.include:
            max_construct_sigma = 0

            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                if ("construct_sigma" in list(log.keys())):
                    x = log["Generation"]
                    y = log["construct_sigma"]
                    if max(y) > max_construct_sigma:
                        max_construct_sigma = max(y)
                    print("min of sigma", min(y), "for ", key)

                    sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="construct_sigma", ci=self.ci, label=label)

            if ("construct_sigma" in list(log.keys())):
                self.axs[count].set(xlabel="Time (in generations)")
                self.axs[count].set(ylabel="$Var(c)$, \n construction variance")
            #self.axs[count].set_ylim((0, max_construct_sigma))
            loc = plticker.LinearLocator(numticks=num_yticks)  # this locator puts ticks at regular intervals
            self.axs[count].yaxis.set_major_locator(loc)
            self.axs[count].get_legend().remove()
            print("plotted consrtuct_sigma")


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

                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="SD", ci=self.ci, label=label)


            self.axs[count].set(ylabel="$\mathbb{E}(sigma)$,\n plasticity")
            self.axs[count].set(xlabel=None)
            self.axs[count].set_yscale('log')
            loc = plticker.LogLocator(base=10, numticks=num_yticks)  # this locator puts ticks at regular intervals
            self.axs[count].yaxis.set_major_locator(loc)
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
                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="R", ci=self.ci, label=label)


            self.axs[count].set(ylabel="$\mathbb{E}(r)$,\nevolvability")
            self.axs[count].set(xlabel=None)
            self.axs[count].set_yscale('log')
            loc = plticker.LogLocator(base=10, numticks=num_yticks)  # this locator puts ticks at regular intervals
            self.axs[count].yaxis.set_major_locator(loc)
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
                x = log["Generation"]
                y = log["extinctions"]

                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="extinctions", ci=self.ci, label=label)


            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$X$,\n extinctions")
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

                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="num_agents", ci=self.ci, label=label)


            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$K$,\n population size ")
            #self.axs[count].set_yscale('log')
            #self.axs[count].set_ylim((1, 10000000))
            #loc = plticker.LogLocator(base=10, numticks=num_yticks)  # this locator puts ticks at regular intervals
            #self.axs[count].yaxis.set_major_locator(loc)

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
                x = log["Generation"]
                y = log["competition"]

                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="competition", ci=self.ci, label=label)


            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="competition")
            self.axs[count].get_legend().remove()
            count +=1
        # # --------------------------
        # if "niche_coordination" in self.include:
        #     for key, value in results.items():
        #         label = key
        #         log = value[0]
        #         log_niches = value[1]
        #
        #         for trial_idx, trial in log_niches.items():
        #             if "constructed" in trial.keys():
        #                 for gen_idx, gen in enumerate(trial["constructed"][0::10]):
        #                     print(gen_idx, len(constructed_mean))
        #                     for niche_index, niche_constructed in gen.items():
        #
        #
        #
        #                     constructed_mean[gen_idx] += np.mean([el[1] for el in gen])
        #
        #
        #
        #         config = value[2]
        #         x = log["Generation"]
        #         y = log["diversity"]
        #
        #         sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="diversity", ci=self.ci, label=label)
        #
        #     self.axs[count].set(xlabel="Time (in generations)")
        #     self.axs[count].set(ylabel="$V$,\n diversity")
        #     self.axs[count].get_legend().remove()
        #
        #     count += 1
        # ----- plot genomic diversity -----
        if "diversity" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                x = log["Generation"]
                y = log["diversity"]

                sns.lineplot(ax=self.axs[count],data=log, x="Generation", y="diversity", ci=self.ci, label=label)


            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$V$,\n diversity")
            self.axs[count].set_yscale('log')

            self.axs[count].get_legend().remove()

            count += 1
        if "diversity_mean" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                x = log["Generation"]
                y = log["diversity_mean"]

                sns.lineplot(ax=self.axs[count],data=log, x="Generation", y="diversity_mean", ci=self.ci, label=label)


            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$V_{\mu}$,\n diversity in mean")
            self.axs[count].set_yscale('log')

            self.axs[count].get_legend().remove()

            count += 1
        if "diversity_sigma" in self.include:
            for key, value in results.items():
                label = key
                log = value[0]
                log_niches = value[1]
                config = value[2]
                x = log["Generation"]
                y = log["diversity_sigma"]

                sns.lineplot(ax=self.axs[count],data=log, x="Generation", y="diversity_sigma", ci=self.ci, label=label)


            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$V_{\sigma}$,\n diversity in $\sigma$")
            self.axs[count].set_yscale('log')

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

                sns.lineplot(ax=self.axs[count], data=log, x="Generation", y="Dispersal",ci=self.ci, label=label)


            self.axs[count].set(xlabel="Time (in generations)")
            self.axs[count].set(ylabel="$D$,\ndispersal")
            self.axs[count].get_legend().remove()
            self.axs[count].set_ylim((50, 100))
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
        order = ["evolv", "niche-construction"]
    for o in order:
        for p in projects:
            if o in p:
                ordered_projects.append(p)
    projects = ordered_projects
    #projects = [projects[0]]
    print(projects)
    for p in projects:
        if "plots" not in p and ".DS" not in p:
            trial_dirs = list(next(os.walk(p + "/trials"))[1])
            log_niches_total = {}
            log_df = pd.DataFrame()
            for trial_idx, trial_dir in enumerate(trial_dirs):
                try:
                    log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))
                    log_niches = pickle.load(open(p + "/trials/" + trial_dir + '/log_niches.pickle', 'rb'))
                    trial = trial_dir.find("trial_")
                    # trial = int(trial_dir[(trial + 6):])

                    if log_df.empty:
                        log_df = log
                    else:
                        log_df = log_df.append(log, ignore_index=True)
                    log_niches_total[trial_idx] = log_niches

                except (IOError, EOFError) as e:
                    print("No log file for project. ", trial_dir)

            skip_lines = 1
            with open(p + "/config.yml") as f:
                for i in range(skip_lines):
                    _ = f.readline()
                config = yaml.safe_load(f)
                # config = SimpleNamespace(**config)

            label = find_label(SimpleNamespace(**config), parameter)
            results[label] = [log_df, log_niches_total, config]

    if config["only_climate"]:
        include = ["climate"]
    else:
        include = ["climate", "mutate", "num_agents", "dispersal"]

        if config["genome_type"] != "intrinsic":
            include.append("sigma")
            include.append("mean")

        if config["genome_type"] == "niche-construction" and parameter != "genome":
            include.append("construct")
            include.append("construct_sigma")
            # include.append( "constructed")

    include = ["climate", "mutate", "num_agents", "dispersal", "diversity",
               "diversity_mean", "diversity_sigma", "mean", "sigma"]

    print("plotting")
    plotter = Plotter(project=results_dir,
                      num_niches=config["num_niches"],
                      log={},
                      log_niches={},
                      include=include)
    plotter.compare_evolution(results, save_name="compare_select")


    #plotter.compare_intrinsic(results, results_dir)


