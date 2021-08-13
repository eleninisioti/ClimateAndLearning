import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:

  def __init__(self, project, env_profile, climate_noconf):
    """
    Args:
      climate_noconf (bool): don't plot confidence intervals for climate
      """
    self.project = project
    self.env_profile = env_profile
    self.climate_noconf = climate_noconf
    plt.rcParams['font.size'] = '16'
    plt.rc('axes', labelsize='24')
    #plt.ioff()


  def plot_trial(self,  log, trial):
    # plot evolution
    self.plot_evolution(log, trial)
    self.plot_species(log, trial)
    self.plot_diversity(log, trial)

  def plot_with_conf(self, log):
    #self.plot_species_with_conf(log)
    self.plot_evolution_with_conf(log)
    #self.plot_diversity_with_conf(log)



  def plot_evolution(self, log, trial, include=[1,1,1,1], cycles=None):
    fig, axs = plt.subplots(sum(include), figsize=(20,10))
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
    #axs[count-1].set(xlabel="Time (in generations)")

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
          axs[subplot].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.2, color='gray')
          axs[subplot].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.2, color='green')

    plt.savefig("../projects/" + self.project + "/trials/trial_" + str(trial) + "/plots/evolution_" + str(include) + ".png")
    plt.clf()

  def plot_evolution_with_conf(self, log, include, cycles=None):
    fig, axs = plt.subplots(sum(include), figsize=(20, 10))
    if cycles is None:
      cycles = len(self.env_profile["start_a"])
    count = 0
    start_cycle = cycles - 2 # which cycles to plot?
    end_cycle = cycles
    #max_gen = int(cycles * self.env_profile["cycle"])
    log = log[(start_cycle * self.env_profile["cycle"]) <= log['Generation'] ]
    log = log[log['Generation'] <= (end_cycle * self.env_profile["cycle"])]

    if include[0]:
      if self.climate_noconf:

        log_climate = log.loc[(log['Trial'] == 1)]
      else:
        log_climate = log
      sns.lineplot(ax=axs[count], data=log_climate, x="Generation", y="Climate")
      axs[count].set(ylabel="$s$")
      axs[count].set(xlabel=None)

      count += 1
    if include[1]:
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Mean")
      axs[count].set(ylabel="$\\bar{\mu}$")
      axs[count].set(xlabel=None)

      count += 1
    if include[2]:
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="SD")
      axs[count].set(ylabel="$\\bar{\sigma}$")
      axs[count].set(xlabel=None)

      count += 1
    if include[3]:
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Fitness")
      axs[count].set(xlabel="Time (in generations)")
      axs[count].set(ylabel="$\\bar{f}$")
      count +=1
    axs[count - 1].set(xlabel="Time (in generations)")

    # highlight periods of variability

    if len(self.env_profile):
      plot_regions = True
    else:
      plot_regions = False

    if plot_regions:
      for subplot in range(count):
        for cycle in range(start_cycle, end_cycle):
          axs[subplot].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.2,
                               color='gray')
          axs[subplot].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.2,
                               color='green')
    plt.savefig("../projects/" + self.project + "/plots/evolution_" + str(include) + "_" + str(self.climate_noconf)
                + ".png")
    plt.clf()


  def plot_generation(self, agents, gen):
    mean_values = [agent.mean for agent in agents]
    SD_values = [agent.SD for agent in agents]
    plt.plot(mean_values, SD_values, 'o')
    plt.savefig("../projects/" + self.project + "/plots/generations/" + str(gen)+ ".png")
    plt.clf()

  def plot_species(self, log, trial, include=[1,1,1,1], cycles=None):
    fig, axs = plt.subplots(sum(include), figsize=(20,10))
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
      #axs[count].plot(range(len(log["specialists"]["extinctions"][:max_gen])), total_species[:max_gen], label="total")
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
      axs[count].plot(range(len(log["specialists"]["extinctions"][:max_gen])), log["specialists"]["diversity"][:max_gen], label="specialist")
      #axs[count].plot(range(len(log["total_diversity"][:max_gen])), log["total_diversity"][:max_gen], label="all")
      axs[count].legend(loc="upper right", fontsize=14)
      axs[count].set(ylabel="Diversity")
      count += 1

    #axs[count - 1].set(xlabel="Time (in generations)")

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
                               color='gray')
          axs[subplot].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.2,
                               color='green')

    plt.savefig(
      "../projects/" + self.project + "/trials/trial_" + str(trial) + "/plots/species_" + str(include) + ".png")
    plt.clf()

  def plot_species_with_conf(self, log, include, cycles=None):

    fig, axs = plt.subplots(sum(include), figsize=(20, 10))
    if cycles is None:
      cycles = len(self.env_profile["start_a"])
    count = 0
    start_cycle = cycles - 2  # which cycles to plot?
    end_cycle = cycles
    # max_gen = int(cycles * self.env_profile["cycle"])
    log = log[(start_cycle * self.env_profile["cycle"]) <= log['Generation'] ]
    log = log[log['Generation'] <= (end_cycle * self.env_profile["cycle"])]

    if include[0]:
      if self.climate_noconf:

        log_climate = log.loc[(log['Trial'] == 1)]
      else:
        log_climate = log
      sns.lineplot(ax=axs[count], data=log_climate, x="Generation", y="Climate")
      axs[count].set(ylabel="$s$")
      axs[count].set(xlabel=None)

      count +=1
    if include[1]:

      # plot species type
      log["Total_Number"] = log["Specialists_Number"] + log["Generalists_Number"]
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Generalists_Number", label="generalists")
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Specialists_Number", label="specialists")
      #sns.lineplot(ax=axs[count], data=log, x="Generation", y="Total_Number", label="total")
      axs[count].legend(loc="upper right", fontsize=14)
      axs[count].set(ylabel="$N$")
      axs[count].set_yscale('log')
      axs[count].set(xlabel=None)

      count += 1
    if include[2]:

      log["Total_Extinct"] = log["Specialists_Extinct"] + log["Generalists_Extinct"]
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Generalists_Extinct", label="generalists")
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Specialists_Extinct", label="specialists")
      #sns.lineplot(ax=axs[count], data=log, x="Generation", y="Total_Extinct", label="total")
      axs[count].legend(loc="upper right", fontsize=14)
      axs[count].set(ylabel="Extinctions")
      axs[count].set(xlabel=None)

      axs[count].set_yscale('log')
      count += 1

    if include[3]:
      # plot for diversity
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Generalists_Diversity", label="generalists")
      sns.lineplot(ax=axs[count], data=log, x="Generation", y="Specialists_Diversity", label="specialists")
      #sns.lineplot(ax=axs[count], data=log, x="Generation", y="Total_Diversity", label="total")
      axs[count].legend(loc="upper right", fontsize=14)
      axs[count].set(ylabel="Diversity")
      count += 1
    print(count, len(axs))
    axs[count - 1].set(xlabel="Time (in generations)")

    if len(self.env_profile):
      plot_regions = True
    else:
      plot_regions = False

    if plot_regions:
      for subplot in range(count):
        for cycle in range(start_cycle, end_cycle):
          axs[subplot].axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.2,
                               color='gray')
          axs[subplot].axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.2,
                               color='green')

    plt.savefig(
      "../projects/" + self.project + "/plots/species_" + str(include) + "_" + str(self.climate_noconf) + ".png")
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
        plt.axvspan(self.env_profile["start_a"][cycle], self.env_profile["end_a"][cycle], alpha=0.5, color='gray')
        plt.axvspan(self.env_profile["start_b"][cycle], self.env_profile["end_b"][cycle], alpha=0.5, color='green')

    plt.savefig("../projects/" + self.project + "/plots/diversity.png")
    plt.clf()





