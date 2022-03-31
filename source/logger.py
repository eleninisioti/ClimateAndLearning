""" Contains class responsible for logging information during an experiment
"""
import numpy as np
import pandas as pd


class Logger:

    def __init__(self, trial, env):
        self.trial = trial
        self.log = {"running_fitness": [],
                    "running_mean": [],
                    "running_SD": [],
                    "running_mutate": [],
                    "diversity": [],
                    "diversity_mean": [],
                    "scale_diversity_mean": [],
                    "diversity_sigma": [],
                    "scale_diversity_sigma": [],
                    "diversity_mutate": [],
                    "scale_diversity_mutate": [],
                    "fixation_index": [],
                    "extinctions": [],
                    "num_agents": [],
                    "specialists": {"speciations": [], "extinctions": [], "number": [], "diversity": [],
                                    "diversity_mean": [], "diversity_std": []},
                    "generalists": {"speciations": [], "extinctions": [], "number": [], "diversity": [],
                                    "diversity_mean": [], "diversity_std": []}}
        self.log_FPindex = False # indicates whether to compute the fixation index (https://en.wikipedia.org/wiki/Fixation_index) of the population
        self.env = env
        self.log_niches = {"inhabited_niches": []}

    def log_gen(self, population):
        """ Compute metrics characterizing the generation.

        Parameters
        ----------
        population: Population
            the population
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
        print("running_Sd", mean_SD)
        self.log["running_mutate"].append(mean_mutate)
        self.log["extinctions"].append(population.num_extinctions)
        self.log["num_agents"].append(len(population.agents))
        #print("from log", len(population.agents))

        inhabited_niches = []
        for agent in population.agents:
            inhabited_niches.extend(agent.niches)
        inhabited_niches = list(set(inhabited_niches))
        self.log_niches["inhabited_niches"].append(inhabited_niches)

        # compute population diversity
        self.log["diversity_mean"].append(np.std(mean_values))
        self.log["diversity_sigma"].append(np.std(SD_values))
        self.log["diversity_mutate"].append(np.std(mutate_values))
        if not len(mean_values):
            mean_values = SD_values = mutate_values = [0]
        self.log["scale_diversity_mean"].append(max(mean_values))
        self.log["scale_diversity_sigma"].append(max(SD_values))
        self.log["scale_diversity_mutate"].append(max(mutate_values))
        self.log["diversity"].append(np.std(mean_values) + np.std(SD_values) + np.std(mutate_values))

        # compute population FP-index
        if self.log_FPindex:
            within_var = 0
            for niche in inhabited_niches:
                # find population within niche
                niche_pop = []
                for agent in population.agents:
                    if niche in agent.niches:
                        niche_pop.append(agent)
                mean_values = []
                SD_values = []
                mutate_values = []
                for agent in niche_pop:
                    mean_values.append(agent.genome.genes["mean"])
                    SD_values.append(agent.genome.genes["sigma"])
                    mutate_values.append(agent.genome.genes["r"])
                niche_mean = np.mean(mean_values) + np.mean(SD_values) + np.mean(mutate_values)
                within_var += len(niche_pop) * niche_mean * (1 - niche_mean)

            pop_mean = mean_mean + mean_SD + mean_mutate
            pop_var = pop_mean * (1 - pop_mean)
            self.log["fixation_index"].append((pop_var - within_var) / (pop_var))

    def final_log(self):
        env = self.env
        self.log["climate_values"] = env.climate_values

        # ----- adapt for  plotting -----
        self.log = {"Generation": [idx for idx in range(len(self.log["climate_values"]))],
                    "Climate": self.log["climate_values"],
                    'Fitness': self.log["running_fitness"],
                    "Mean": self.log["running_mean"],
                    "SD": self.log["running_SD"],
                    "R": self.log["running_mutate"],
                    "extinctions": self.log["extinctions"],
                    "num_agents": self.log["num_agents"],
                    "diversity": self.log["diversity"],
                    "diversity_mean": self.log["diversity_mean"],
                    "diversity_sigma": self.log["diversity_sigma"],
                    "diversity_mutate": self.log["diversity_mutate"],
                    "scale_diversity_mean": self.log["scale_diversity_mean"],
                    "scale_diversity_sigma": self.log["scale_diversity_sigma"],
                    "scale_diversity_mutate": self.log["scale_diversity_mutate"],
                    "fixation_index": self.log["fixation_index"],
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



        # convert to dataframe and save trial data
        for step in range(len(self.log["Fitness"])):
            trial_log = {'Generation': [step], 'Trial': [self.trial]}
            for key in self.log.keys():
                if len(self.log[key]):
                    trial_log[key] = self.log[key][step]
            if step:
                log_df = log_df.append(pd.DataFrame.from_dict(trial_log))
            else:
                log_df = pd.DataFrame.from_dict(trial_log)
        self.log = log_df
