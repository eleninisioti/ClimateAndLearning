import numpy as np
from numpy.random import normal
import random

class Genome:

    def __init__(self, type, env_mean, init_SD, init_mutate, mutate_rate=0.05):
        """

        Parameters
        ---------
        type: str
            choose between '1d', '1d_mutate', '2d'

        """
        self.type = type

        if self.type == "1D" or self.type == "1D_mutate":
            mu = normal(env_mean, init_SD)
            sigma = np.abs(normal(0, init_SD))
            r = init_mutate
            self.genes = {"mean": mu, "sigma": sigma, "r": r}
            self.mutate_rate = mutate_rate



    def mutate(self):
        if self.type == "1D":
            self.genes["mean"] = self.genes["mean"] + normal(0, self.genes["r"])
            self.genes["sigma"] = np.abs(self.genes["sigma"] + normal(0, self.genes["r"]))
        elif self.type == "1D_mutate":
            self.genes["mean"] = self.genes["mean"] + normal(0, self.genes["r"])
            self.genes["sigma"] = np.abs(self.genes["sigma"] + normal(0, self.genes["r"]))
            self.genes["r"] = np.abs(self.genes["r"] + normal(0, self.mutate_rate))

    def cross(self, genomes):
        for key in self.genes.keys():
            parent_idx = random.randint(0,1)
            self.genes[key] = genomes[parent_idx].genes[key]
