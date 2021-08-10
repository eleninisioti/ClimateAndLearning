import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:

  def __init__(self, project, env_profile):
    self.project = project
    self.env_profile = env_profile
    plt.rcParams['font.size'] = '16'
    plt.rc('axes', labelsize='24')
    #plt.ioff()


  def plot_trial(self,  log, trial):
    # plot evolution
    self.plot_evolution(log, trial)
    self.plot_species(log, trial)
    self.plot_diversity(log, trial)

  def plot_with_conf(self, log, include, cycles):
    self.include = include
    self.cycles = cycles
    self.plot_diversity_with_conf(log)
    self.plot_species_with_conf(log)
    self.plot_evolution_with_conf(log)



  def plot_evolution(self, log, trial, include=[1,1,1,1], cycles=None):
    fig, axs = plt.subplots(sum(include), figsize=(20,8))
    count = 0
    max_gen = int(self.env_profile["ncycles"]*self.env_profile["cycle"])


    if include[0]:
      axs[count].plot(list(range(len(log["climate_values"][:max_gen]))), log["climate_values"][:max_gen])
      axs[count].set(ylabel="$s$")

      count += 1
    if include[1]:
      axs[count].plot(list(range(len(log["running_mean"][:max_gen]))), log["running_mean"][:max_gen])
      axs[count].set(ylabel="$\\bar{\mu}$")
      count += 1
    if include[2]:
      axs[count].plot(list(range(len(log["running_SD"][:max_gen]))), log["running_SD"][:max_gen])
      axs[count].set(ylabel="$\\bar{\sigma}$")
      count += 1
    if include[3]:
      axs[count].plot(list(range(len(log["running_fitness"][:max_gen]))), log["running_fitness"][:max_gen])
      axs[count].set(ylabel="$\\bar{f}$")
      count += 1
    axs[count-1].set(xlabel="Time (in generations)")

    # highlight periods of variability
    if cycles is None:
      cycles = len(self.env_profile["start_a"])
    if len(self.env_profile):
      plot_regions = True
    else:
      plot_regions = False

    if plot_regions:
      for subplot in range(count):
        for cycle in range(cycles):
          axs[subplot].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.2, color='red')
          axs[subplot].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.2, color='blue')

    plt.savefig("../projects/" + self.project + "/trials/trial_" + str(trial) + "/plots/evolution_" + str(include) + ".png")
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

  def plot_species(self, log, trial, include=[1,1,1,1], cycles=None):
    fig, axs = plt.subplots(sum(include), figsize=(20,8))
    count = 0
    max_gen = int(self.env_profile["ncycles"]*self.env_profile["cycle"])

    if include[0]:
      axs[count].plot(list(range(len(log["climate_values"][:max_gen]))), log["climate_values"][:max_gen])
      axs[count].set(ylabel="$s$")
      count += 1
    if include[1]:

      # plot species type
      total_species = [el + log["generalists"]["number"][idx] for idx, el in
                           enumerate(log["specialists"]["number"])]
      axs[count].plot(range(len(log["specialists"]["extinctions"][:max_gen])), log["generalists"]["number"][:max_gen],
                      label="generalists")
      axs[count].plot(range(len(log["specialists"]["extinctions"][:max_gen])), log["specialists"]["number"][:max_gen],
                      label="specialists")
      axs[count].plot(range(len(log["specialists"]["extinctions"][:max_gen])), total_species[:max_gen], label="total")
      axs[count].legend(loc="upper right", fontsize=14)
      axs[count].set(ylabel="$N$")
      axs[count].set_yscale('log')
      count += 1

    # plot for extinctions
    if include[2]:
      extinctions_total = [el + log["generalists"]["extinctions"][idx] for idx, el in
                           enumerate(log["specialists"]["extinctions"])]
      axs[count].plot(range(len(log["generalists"]["extinctions"][:max_gen])), log["generalists"]["extinctions"][:max_gen], label="generalists")
      axs[count].plot(range(len(log["specialists"]["extinctions"][:max_gen])), log["specialists"]["extinctions"][:max_gen], label="specialists")
      axs[count].plot(range(len(extinctions_total[:max_gen])), extinctions_total[:max_gen], label="all")
      axs[count].set(ylabel="Extinctions")
      axs[count].set_yscale('log')
      count += 1
    if include[3]:


      # plot for diversity
      axs[count].plot(range(len(log["generalists"]["extinctions"][:max_gen])), log["generalists"]["diversity"][:max_gen], label="generalists")
      #axs[count].plot(range(len(log["specialists"]["extinctions"][:max_gen])), log["specialists"]["diversity"][:max_gen], label="specialist")
      #axs[count].plot(range(len(log["total_diversity"][:max_gen])), log["total_diversity"][:max_gen], label="all")
      axs[count].legend(loc="upper right", fontsize=14)
      axs[count].set(ylabel="Diversity")
      count += 1

    axs[count - 1].set(xlabel="Time (in generations)")

    # highlight periods of variability
    if cycles is None:
      cycles = len(self.env_profile["start_a"])
    if len(self.env_profile):
      plot_regions = True
    else:
      plot_regions = False

    if plot_regions:
      for subplot in range(count):
        for cycle in range(cycles):
          axs[subplot].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.2,
                               color='red')
          axs[subplot].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.2,
                               color='blue')

    plt.savefig(
      "../projects/" + self.project + "/trials/trial_" + str(trial) + "/plots/species_" + str(include) + ".png")
    plt.clf()

  def plot_species_with_conf(self, log):

    fig, axs = plt.subplots(sum(self.include), figsize=(20, 8))
    count = 0
    max_gen = int(self.env_profile["ncycles"] * self.env_profile["cycle"])
    log = log.loc[(log['Generation'] <= max_gen)]

    if self.include[0]:
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Climate")
      axs[count].set(ylabel="$s$")
      count +=1
    if self.include[1]:

      # plot species type
      log["Total_Number"] = log["Specialists_Number"] + log["Generalists_Number"]
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Generalists_Number", label="generalists")
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Specialists_Number", label="specialists")
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Total_Number", label="total")
      axs[count].legend(loc="upper right", fontsize=14)
      axs[count].set(ylabel="$N$")
      axs[count].set_yscale('log')
      count += 1
    if self.include[2]:

      log["Total_Extinct"] = log["Specialists_Extinct"] + log["Generalists_Extinct"]
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Generalists_Extinct", label="generalists")
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Specialists_Extinct", label="specialists")
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Total_Extinct", label="total")
      axs[count].legend(loc="upper right", fontsize=14)
      axs[count].set(ylabel="Extinctions")
      axs[count].set_yscale('log')
      count += 1

    if self.include[3]:
      # plot for diversity
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Generalists_Diversity", label="generalists")
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Specialists_Diversity", label="specialists")
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Total_Diversity", label="total")
      axs[count].legend(loc="upper right", fontsize=14)
      axs[count].set(ylabel="Diversity")
      count += 1

    axs[count - 1].set(xlabel="Time (in generations)")

    # highlight periods of variability
    if self.cycles is None:
      cycles = len(self.env_profile["start_a"])
    else:
      cycles = self.cycles
    if len(self.env_profile):
      plot_regions = True
    else:
      plot_regions = False

    if plot_regions:
      for subplot in range(count):
        for cycle in range(cycles):
          axs[subplot].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.2,
                               color='red')
          axs[subplot].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.2,
                               color='blue')

    plt.savefig(
      "../projects/" + self.project + "/plots/species_" + str(self.include) + ".png")
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





