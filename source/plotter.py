import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:

  def __init__(self, project, env_profile):
    self.project = project
    self.env_profile = env_profile
    plt.rcParams['font.size'] = '16'
    plt.rc('axes', labelsize='24')
    plt.ioff()


  def plot_trial(self,  log, trial):
    # plot evolution
    self.plot_evolution(log, trial)
    self.plot_species(log, trial)
    self.plot_diversity(log, trial)

  def plot_with_conf(self, log):
    self.plot_diversity_with_conf(log)
    self.plot_species_with_conf(log)
    self.plot_evolution_with_conf(log)

  def plot_evolution(self, log, trial):
    fig, axs = plt.subplots(4, figsize=(15,10))
    axs[0].plot(list(range(len(log["climate_values"]))), log["climate_values"])
    axs[0].set(ylabel="$s$")
    axs[1].plot(list(range(len(log["running_mean"]))), log["running_mean"])
    axs[1].set(ylabel="$\\bar{\mu}$")
    axs[2].plot(list(range(len(log["running_SD"]))), log["running_SD"])
    axs[2].set(ylabel="$\\bar{\sigma}$")
    axs[3].plot(list(range(len(log["running_fitness"]))), log["running_fitness"])
    axs[3].set(xlabel="Time (in generations)")
    axs[3].set(ylabel="$\\bar{f}$")

    # highlight periods of variability
    if len(self.env_profile):
      for cycle in range(len(self.env_profile["start_a"])):
        axs[0].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='red')
        axs[0].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')
        axs[1].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='red')
        axs[1].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')
        axs[2].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='red')
        axs[2].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')
        axs[3].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle] , alpha=0.5, color='red')
        axs[3].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')

    plt.savefig("../projects/" + self.project + "/trials/trial_" + str(trial) + "/plots/evolution.png")
    plt.clf()

  def plot_evolution_with_conf(self, log):
    fig, axs = plt.subplots(4, figsize=(15, 10))
    sns.lineplot(ax=axs[0], data=log, x="Generation", y="Climate")
    axs[0].set(ylabel="$s$")
    sns.lineplot(ax=axs[1], data=log, x="Generation", y="Mean")
    axs[1].set(ylabel="$\\bar{\mu}$")
    sns.lineplot(ax=axs[2], data=log, x="Generation", y="SD")
    axs[2].set(ylabel="$\\bar{\sigma}$")
    sns.lineplot(ax=axs[3], data=log, x="Generation", y="Fitness")
    axs[3].set(xlabel="Time (in generations)")
    axs[3].set(ylabel="$\\bar{f}$")
    # highlight periods of variability
    if len(self.env_profile):
      for cycle in range(len(self.env_profile["start_a"])):
        axs[0].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='red')
        axs[0].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')
        axs[1].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='red')
        axs[1].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')
        axs[2].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='red')
        axs[2].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')
        axs[3].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='red')
        axs[3].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')

    plt.savefig("../projects/" + self.project + "/plots/evolution.png")
    plt.clf()


  def plot_generation(self, agents, gen):
    mean_values = [agent.mean for agent in agents]
    SD_values = [agent.SD for agent in agents]
    plt.plot(mean_values, SD_values, 'o')
    plt.savefig("../projects/" + self.project + "/plots/generations/" + str(gen)+ ".png")
    plt.clf()

  def plot_species(self, log, trial):

    fig, axs = plt.subplots(4, figsize=(15, 10))

    axs[0].plot(list(range(len(log["climate_values"]))), log["climate_values"])
    axs[0].set(ylabel="$s$")

    # plot species type
    total_species = [el + log["generalists"]["number"][idx] for idx, el in
                         enumerate(log["specialists"]["number"])]
    axs[1].plot(range(len(log["specialists"]["extinctions"])), log["generalists"]["number"], label="generalists")
    axs[1].plot(range(len(log["specialists"]["extinctions"])), log["specialists"]["number"], label="specialists")
    axs[1].plot(range(len(log["specialists"]["extinctions"])), total_species, label="total")
    axs[1].legend(loc="upper right", fontsize=14)
    axs[1].set(ylabel="$N$")
    axs[1].set_yscale('log')

    # plot for extinctions
    extinctions_total = [el + log["generalists"]["extinctions"][idx] for idx, el in
                         enumerate(log["specialists"]["extinctions"])]
    axs[2].plot(range(len(log["generalists"]["extinctions"])), log["generalists"]["extinctions"], label="generalists")
    axs[2].plot(range(len(log["specialists"]["extinctions"])), log["specialists"]["extinctions"], label="specialists")
    axs[2].plot(range(len(extinctions_total)), extinctions_total, label="all")
    axs[2].set(ylabel="Extinctions")
    axs[2].set_yscale('log')


    # plot for diversity
    axs[3].plot(range(len(log["generalists"]["extinctions"])), log["generalists"]["diversity"], label="generalists")
    axs[3].plot(range(len(log["specialists"]["extinctions"])), log["specialists"]["diversity"], label="specialist")
    axs[3].plot(range(len(log["total_diversity"])), log["total_diversity"], label="all")
    axs[3].legend(loc="upper right", fontsize=14)
    axs[3].set(ylabel="Diversity")
    axs[3].set(xlabel="Time (in generations)")
    # highlight periods of variability
    if len(self.env_profile):
      for cycle in range(len(self.env_profile["start_a"])):
        axs[0].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='red')
        axs[0].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')
        axs[1].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='red')
        axs[1].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')
        axs[2].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='red')
        axs[2].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')
        axs[3].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle] , alpha=0.5, color='red')
        axs[3].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')

    plt.savefig("../projects/" + self.project + "/trials/trial_" + str(trial) + "/plots/species.png")
    plt.clf()

  def plot_species_with_conf(self, log):

    fig, axs = plt.subplots(4, figsize=(15, 10))
    sns.lineplot(ax=axs[0], data=log, x="Generation", y="Climate")
    axs[0].set(ylabel="$s$")

    # plot species type
    log["Total_Number"] = log["Specialists_Number"] + log["Generalists_Number"]
    sns.lineplot(ax=axs[1], data=log, x="Generation", y="Generalists_Number", label="generalists")
    sns.lineplot(ax=axs[1], data=log, x="Generation", y="Specialists_Number", label="specialists")
    sns.lineplot(ax=axs[1], data=log, x="Generation", y="Total_Number", label="total")
    axs[1].legend(loc="upper right", fontsize=14)
    axs[1].set(ylabel="$N$")
    axs[1].set_yscale('log')

    log["Total_Extinct"] = log["Specialists_Extinct"] + log["Generalists_Extinct"]
    sns.lineplot(ax=axs[2], data=log, x="Generation", y="Generalists_Extinct", label="generalists")
    sns.lineplot(ax=axs[2], data=log, x="Generation", y="Specialists_Extinct", label="specialists")
    sns.lineplot(ax=axs[2], data=log, x="Generation", y="Total_Extinct", label="total")
    axs[2].legend(loc="upper right", fontsize=14)
    axs[2].set(ylabel="Extinctions")
    axs[2].set_yscale('log')



    # plot for diversity
    sns.lineplot(ax=axs[3], data=log, x="Generation", y="Generalists_Diversity", label="generalists")
    sns.lineplot(ax=axs[3], data=log, x="Generation", y="Specialists_Diversity", label="specialists")
    sns.lineplot(ax=axs[3], data=log, x="Generation", y="Total_Diversity", label="total")
    axs[3].legend(loc="upper right", fontsize=14)
    axs[3].set(ylabel="Diversity")
    axs[3].set(xlabel="Time (in generations)")

    # highlight periods of variability
    if len(self.env_profile):
      for cycle in range(len(self.env_profile["start_a"])):
        axs[0].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='red')
        axs[0].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')
        axs[1].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='red')
        axs[1].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')
        axs[2].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='red')
        axs[2].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')
        axs[3].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle] , alpha=0.5, color='red')
        axs[3].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')

    plt.savefig("../projects/" + self.project + "/plots/species.png")
    plt.clf()


  def plot_diversity(self, log, trial):

    plt.plot(range(len(log["generalists"]["diversity_mean"])), log["generalists"]["diversity_mean"],
             label="generalists-mean")
    plt.plot(range(len(log["specialists"]["diversity_mean"])), log["specialists"]["diversity_mean"],
             label="specialist-mean")
    plt.plot(range(len(log["generalists"]["diversity_mean"])), log["generalists"]["diversity_std"],
             label="generalists-std")
    plt.plot(range(len(log["specialists"]["diversity_mean"])), log["specialists"]["diversity_std"],
             label="specialist-std")
    plt.legend()
    plt.ylabel("Diversity")

    # highlight periods of variability
    if len(self.env_profile):
      for cycle in range(len(self.env_profile["start_a"])):
        plt.axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='red')
        plt.axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')


    plt.savefig("../projects/" + self.project + "/trials/trial_" + str(trial) + "/plots/diversity.png")
    plt.clf()


  def plot_diversity_with_conf(self, log):
    sns.lineplot(data=log, x="Generation", y="Generalists_Diversity_Mean", label="generalists-mean")
    sns.lineplot(data=log, x="Generation", y="Specialists_Diversity_Mean", label="specialists-mean" )
    sns.lineplot(data=log, x="Generation", y="Generalists_Diversity_SD", label="generalists-std")
    sns.lineplot(data=log, x="Generation", y="Specialists_Diversity_SD", label="specialists-std" )
    plt.legend()
    plt.ylabel("Diversity")

    # highlight periods of variability
    if len(self.env_profile):
      for cycle in range(len(self.env_profile["start_a"])):
        plt.axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='red')
        plt.axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='blue')

    plt.savefig("../projects/" + self.project + "/plots/diversity.png")
    plt.clf()





