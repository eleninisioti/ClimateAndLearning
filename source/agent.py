from scipy import stats
import numpy as np
import math
from utils import compute_survival

class Agent:

    def __init__(self, genome):
        self.genome = genome
        self.ancestry = []

    def mutate(self):
        self.genome.mutate()

    def compute_fitness(self, env_mean):
        #self.is_extinct(env)
        self.fitness = stats.norm(self.genome.genes["mean"], self.genome.genes["sigma"]).pdf(env_mean)
        return self.fitness

    def is_generalist(self):
        if self.genome["sigma"] > 0.1:
            self.generalist = True
        else:
            self.generalist = False
        return self.generalist

    def is_extinct_old(self, env_mean):
        if ((self.genome.genes["mean"] - 2 * self.genome.genes["sigma"]) > env_mean) \
                or (self.genome.genes["mean"] + 2 * self.genome.genes["sigma"]) < env_mean:
            return True
        else:
            return False

    def is_extinct(self, env):
        num_latitudes = env.num_niches
        survival = 0
        self.niches = []
        self.capacities = []
        total_capacity = 0
        self.fitness_values = []
        for lat in range(-int(num_latitudes/2), int(num_latitudes/2) + 1):
            lat_climate = env.mean + 0.01 * lat

            if ((self.genome.genes["mean"] - 2 * self.genome.genes["sigma"]) < lat_climate) \
                    and (lat_climate < self.genome.genes["mean"] + 2 * self.genome.genes["sigma"]):
                survival += 1
                self.niches.append(lat_climate)
                self.capacities.append(np.floor(lat_climate*env.orig_capacity))
                self.fitness_values.append(self.compute_fitness(lat_climate))

            total_capacity += lat_climate*env.orig_capacity
        if survival and len(self.niches)==0:
            print("check")

        if len(self.fitness_values):
            self.fitness = np.mean(self.fitness_values)
        else:
            self.fitness = 0
        #print(survival)
        #if not survival:
            #print("agent dieed with ", self.genome.genes["mean"], self.genome.genes["sigma"], lat_climate)
        return (not survival), self


    def has_speciated(self, agents):
        thres_mean = 0.05
        thres_SD = 0.01

        if len(self.ancestry) > 50:
            current_gene = self.ancestry[-1]
            previous_gene = self.ancestry[-50]

            same = False
            if np.abs(current_gene[0] - previous_gene[0]) < thres_mean:
                if np.abs(current_gene[1] - previous_gene[1]) < thres_SD:
                    same = True
            if same:
                return False
            else:
                return True
        else:
            return False
