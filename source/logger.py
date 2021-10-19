
import numpy as np

class Logger:

    def __init__(self):
        self.log = {"running_fitness": [],
                    "running_mean": [],
                    "running_SD": [],
                    "running_mutate": [],
                    "total_diversity": [],
                    "specialists": {"speciations": [], "extinctions": [], "number": [], "diversity": [],
                                    "diversity_mean": [], "diversity_std": []},
                    "generalists": {"speciations": [], "extinctions": [], "number": [], "diversity": [],
                                    "diversity_mean": [], "diversity_std": []}}

    def log_gen(self, population):
        """ Compute metrics characterizing the generation.
        """
        # compute generation averages
        fitness_values = []
        mean_values = []
        SD_values = []
        mutate_values = []
        for agent in population.agents:
            fitness_values.append(agent.fitness)
            mean_values.append(agent.genome.genes["mean"])
            SD_values.append(agent.genome.genes["sigma"])
            mutate_values.append(agent.genome.genes["r"])
        mean_fitness = np.mean(fitness_values)
        mean_mean = np.mean(mean_values)
        mean_SD = np.mean(SD_values)
        mean_mutate = np.mean(mutate_values)
        self.log["running_fitness"].append(mean_fitness)
        self.log["running_mean"].append(mean_mean)
        self.log["running_SD"].append(mean_SD)
        self.log["running_mutate"].append(mean_mutate)

    def final_log(self, env):
        self.log["climate_values"] = env.climate_values

        # convert log for plotting
        self.log = { "Climate": self.log["climate_values"],
                     'Fitness': self.log["running_fitness"],
                     "Mean": self.log["running_mean"],
                     "SD": self.log["running_SD"],
                     "R": self.log["running_mutate"],
                     "climate_values" : self.log["climate_values"],
                     "Total_Diversity": self.log["total_diversity"],
                     "Specialists_Extinct": self.log["specialists"]["extinctions"],
                     "Specialists_Number": self.log["specialists"]["number"],
                     "Specialists_Diversity": self.log["specialists"]["diversity"],
                     "Specialists_Diversity_Mean": self.log["specialists"]["diversity_mean"],
                     "Specialists_Diversity_SD": self.log["specialists"]["diversity_std"],
                     "Generalists_Extinct": self.log["generalists"]["extinctions"],
                     "Generalists_Number": self.log["generalists"]["number"],
                     "Generalists_Diversity": self.log["generalists"]["diversity"],
                     "Generalists_Diversity_Mean": self.log["generalists"]["diversity_mean"],
                     "Generalists_Diversity_SD": self.log["generalists"]["diversity_std"]}

        if env.type == "change":
            self.log["env_profile"] = {"start_a": env.b1, "end_a": env.b2,
                                       "start_b": env.b3, "end_b": env.b4,
                                       "cycles": env.cycles}
        elif env.type == "sin":
            self.log["env_profile"] = {}

        elif env.type == "combined":
            self.log["env_profile"] = {"start_a": env.b1_values, "end_a": env.b2_values,
                                       "start_b": env.b3_values, "end_b": env.b4_values,
                                       "ncycles": env.cycles, "cycle": env.b5_values[0]}




