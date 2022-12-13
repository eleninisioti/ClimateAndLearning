import numpy as np
import pandas as pd

class Logger:
    """ Class responsible for logging information during an experiment.
    """

    def __init__(self, trial, env, max_pop):
        self.trial = trial
        self.max_population = max_pop

        # log contains information about the population
        self.log = {"running_fitness": [],
                    "running_mean": [],
                    "running_SD": [],
                    "running_mutate": [],
                    "capacity": [],
                    "constructed": [],
                    "mean_constructed": [],
                    "sigma_constructed": [],

                    "construct": [],
                    "construct_sigma": [],
                    "diversity": [],
                    "diversity_mean": [],
                    "scale_diversity_mean": [],
                    "diversity_sigma": [],
                    "scale_diversity_sigma": [],
                    "diversity_mutate": [],
                    "scale_diversity_mutate": [],
                    "fixation_index": [],
                    "extinctions": [],
                    "competition":[],
                    "not_reproduced":[],
                    "num_agents": [],
                    "specialists": {"speciations": [], "extinctions": [], "number": [], "diversity": [],
                                    "diversity_mean": [], "diversity_std": []},
                    "generalists": {"speciations": [], "extinctions": [], "number": [], "diversity": [],
                                    "diversity_mean": [], "diversity_std": []}}
        self.env = env

        # log_niches contains information for niches
        self.log_niches = {"inhabited_niches": [],
                           "movements": [],
                           "intrinsic_curves": [],
                           "histories": [],
                           "construct": [],
                           "pop_niches": [],
                           "constructed": []}

    def log_gen(self, population, env):
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
        construct_values = []
        for agent in population.agents:
            fitness_values.append(np.mean(list(agent.fitnesses.values())))
            if "mean" in agent.genome.genes.keys():
                mean_values.append(agent.genome.genes["mean"])
            SD_values.append(agent.genome.genes["sigma"])
            mutate_values.append(agent.genome.genes["r"])
            if "c" in agent.genome.genes.keys():
                construct_values.append(agent.genome.genes["c"])
            else:
                construct_values.append(0)
        mean_fitness = np.mean(fitness_values)
        mean_mean = np.mean(mean_values)
        mean_SD = np.mean(SD_values)
        mean_mutate = np.mean(mutate_values)
        mean_construct = np.mean(construct_values)
        sigma_construct = np.std(construct_values)
        self.log["running_fitness"].append(mean_fitness)
        self.log["running_mean"].append(mean_mean)
        self.log["running_SD"].append(mean_SD)
        self.log["running_mutate"].append(mean_mutate)
        self.log["construct"].append(mean_construct)
        self.log["construct_sigma"].append(sigma_construct)
        self.log["extinctions"].append(population.num_extinctions)
        self.log["num_agents"].append(len(population.agents))
        self.log["competition"].append(population.competition)
        self.log["not_reproduced"].append(population.not_reproduced)

        self.log["capacity"].append(np.sum([el["capacity"] for key, el in env.niches.items()]))
        self.log["constructed"].append(np.sum([el["constructed"] for key, el in env.niches.items()]))
        self.log["mean_constructed"].append(np.mean([el["constructed"] for key, el in env.niches.items()]))
        self.log["sigma_constructed"].append(np.std([el["constructed"] for key, el in env.niches.items()]))


        print("capacity ", self.log["capacity"][-1])
        print("sigma", self.log["running_SD"][-1])
        print("constructed ", self.log["constructed"][-1])
        print("construct ", self.log["construct"][-1])
        print("extinctions", self.log["extinctions"][-1])


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

        # which niches are inhabited by at least one agent?
        inhabited_niches = []
        for agent in population.agents:
            inhabited_niches.extend(agent.niches_lat)
        set_inhabited_niches = list(set(inhabited_niches))
        self.log_niches["inhabited_niches"].append(set_inhabited_niches)

        pop_niches = {}
        for niche in set_inhabited_niches:
            for el in inhabited_niches:
                if el == niche:
                    pop_niches[el] = (pop_niches[el] + 1) if (el in pop_niches.keys()) else 0

        for el in list(env.niche_constructions.keys()):
            if el not in pop_niches.keys():
                pop_niches[el]= 0

        print("pop_niches", pop_niches)

        self.log_niches["pop_niches"].append(pop_niches)


        movements = [agent.movement for agent in population.agents]
        while len(movements) < self.max_population:
            movements.append(999)
        self.log_niches["movements"].append(movements)

        histories = [agent.history for agent in population.agents]
        self.log_niches["histories"].append(histories)

        if len(population.agents) and "intrinsic_curves" in population.agents[0].genome.genes.keys():
            self.log_niches["intrinsic_curves"].append([agent.genome.genes["intrinsic_curves"] for agent in population.agents])

        self.log_niches["construct"].append(construct_values)
        self.log_niches["constructed"].append({(key, el["constructed"]) for key, el in env.niches.items()})


    def final_log(self):
        """ Prepare log for final saving.
        """
        env = self.env
        self.log["climate_values"] = env.climate_values

        # ----- reformat for plotting functions -----
        self.log = {"Generation": [idx for idx in range(len(self.log["climate_values"]))],
                    "Climate": self.log["climate_values"],
                    'Fitness': self.log["running_fitness"],
                    "Mean": self.log["running_mean"],
                    "SD": self.log["running_SD"],
                    "R": self.log["running_mutate"],
                    "construct": self.log["construct"],
                    "capacity": self.log["capacity"],
                    "constructed": self.log["constructed"],
                    "mean_constructed": self.log["mean_constructed"],
                    "sigma_constructed": self.log["sigma_constructed"],

                    "construct_sigma": self.log["construct_sigma"],
                    "extinctions": self.log["extinctions"],
                    "competition": self.log["competition"],
                    "not_reproduced": self.log["not_reproduced"],
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

        # convert to dataframe for saving current trial
        log_df = pd.DataFrame()
        for step in range(len(self.log["Fitness"])):
            trial_log = {'Generation': [step], 'Trial': [self.trial]}
            for key in self.log.keys():
                if len(self.log[key]):
                    trial_log[key] = self.log[key][step]
            if log_df.empty:
                log_df = pd.DataFrame.from_dict(trial_log)
            else:
                log_df = log_df.append(pd.DataFrame.from_dict(trial_log))

        self.log = log_df
