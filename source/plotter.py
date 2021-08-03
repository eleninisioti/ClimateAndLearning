import matplotlib.pyplot as plt

class Plotter:

  def __init__(self, project):
    self.project = project
    plt.rcParams['font.size'] = '16'
    plt.rc('axes', labelsize='24')

  def plot_evolution(self):
    fig, axs = plt.subplots(4, figsize=(15,10))
    axs[0].plot(list(range(len(self.log["climate_values"]))), self.log["climate_values"])
    axs[0].set(ylabel="$s$")
    axs[1].plot(list(range(len(self.log["running_mean"]))), self.log["running_mean"])
    axs[1].set(ylabel="$\\bar{\mu}$")
    axs[2].plot(list(range(len(self.log["running_SD"]))), self.log["running_SD"])
    axs[2].set(ylabel="$\\bar{\sigma}$")
    axs[3].plot(list(range(len(self.log["running_fitness"]))), self.log["running_fitness"])
    axs[3].set(xlabel="Time (in generations)")
    axs[3].set(ylabel="$\\bar{f}$")

    plt.savefig("../projects/" + self.project + "/plots/evolution.png")
    plt.clf()

  def plot_project(self,  log):
    self.log = log
    # plot evolution
    self.plot_evolution()

  def plot_generation(self, agents, gen):
    mean_values = [agent.mean for agent in agents]
    SD_values = [agent.SD for agent in agents]
    plt.plot(mean_values, SD_values, 'o')
    plt.savefig("../projects/" + self.project + "/plots/generations/" + str(gen)+ ".png")
    plt.clf()

  def plot_species(self, log):

    fig, axs = plt.subplots(4, figsize=(15, 10))

    axs[0].plot(list(range(len(self.log["climate_values"]))), self.log["climate_values"])
    axs[0].set(ylabel="$s$")

    # plot species type
    total_species = [el + log["generalists"]["number"][idx] for idx, el in
                         enumerate(log["specialists"]["number"])]
    axs[1].plot(range(len(log["specialists"]["extinctions"])), log["generalists"]["number"], label="generalists")
    axs[1].plot(range(len(log["specialists"]["extinctions"])), log["specialists"]["number"], label="specialists")
    axs[1].plot(range(len(log["specialists"]["extinctions"])), total_species, label="total")
    axs[1].legend(loc="upper right", fontsize=14)
    axs[1].set(ylabel="$N$")

    # plot for extinctions
    extinctions_total = [el + log["generalists"]["extinctions"][idx] for idx, el in
                         enumerate(log["specialists"]["extinctions"])]
    axs[2].plot(range(len(log["generalists"]["extinctions"])), log["generalists"]["extinctions"], label="generalists")
    axs[2].plot(range(len(log["specialists"]["extinctions"])), log["specialists"]["extinctions"], label="specialists")
    axs[2].plot(range(len(extinctions_total)), extinctions_total, label="all")
    axs[2].legend(loc="upper right", fontsize=14)
    axs[2].set(ylabel="Extinctions")

    # plot for diversity
    axs[3].plot(range(len(log["generalists"]["extinctions"])), log["generalists"]["diversity"], label="generalists")
    axs[3].plot(range(len(log["specialists"]["extinctions"])), log["specialists"]["diversity"], label="specialist")
    axs[3].plot(range(len(self.log["total_diversity"])), self.log["total_diversity"], label="all")
    axs[3].legend(loc="upper right", fontsize=14)
    axs[3].set(ylabel="Diversity")
    axs[3].set(xlabel="Time (in generations)")
    plt.savefig("../projects/" + self.project + "/plots/species.png")
    plt.clf()


  def plot_diversity(self, log):
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
    #plt.set(ylabel="Diversity")
    plt.savefig("../projects/" + self.project + "/plots/diversity.png")
    plt.clf()



