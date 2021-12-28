""" This script contains the definition of genomes, which are objects assigned to agents.
"""

import numpy as np
from numpy.random import normal
import random


class Genome:

    def __init__(self, type, env_mean, init_SD, init_mutate, mutate_mutate_rate=0.05):
        """ Constructor of Class Genome

        Parameters
        ---------
        type: str
            determines the structure and mutation rule of the genome. choose between '1D',
             '1D_mutate','1D_mutate_fixed'

        env_mean: float
            the state of the environment upon initialization of the genoem

        init_SD: float
            determines the range around env_mean for genome initialization

        init_mutate: float
            the mutation rate upon initialisation

        mutate_mutate_rate: float
            the mutation rate of the mutation rate

        """
        self.type = type

        if self.type == "1D" or self.type == "1D_mutate" or self.type == "1D_mutate_fixed":
            mu = normal(env_mean, init_SD)
            sigma = np.abs(normal(0, init_SD))
            r = init_mutate
            self.genes = {"mean": mu, "sigma": sigma, "r": r}
            self.mutate_mutate_rate = mutate_mutate_rate

    def mutate(self):
        """ Applies mutations to the genome.
        """
        if self.type == "1D":
            # constant mutation rate
            self.genes["mean"] = self.genes["mean"] + normal(0, self.genes["r"])
            self.genes["sigma"] = np.abs(self.genes["sigma"] + normal(0, self.genes["r"]))
        elif self.type == "1D_mutate_fixed":
            # mutation rate evolves with constant mutation rate
            self.genes["mean"] = self.genes["mean"] + normal(0, self.genes["r"])
            self.genes["sigma"] = np.abs(self.genes["sigma"] + normal(0, self.genes["r"]))
            self.genes["r"] = np.abs(self.genes["r"] + normal(0, self.mutate_mutate_rate))
        elif self.type == "1D_mutate":
            # all genes evolve with the same evolving mutation rate
            self.genes["mean"] = self.genes["mean"] + normal(0, self.genes["r"])
            self.genes["sigma"] = np.abs(self.genes["sigma"] + normal(0, self.genes["r"]))
            self.genes["r"] = np.abs(self.genes["r"] + normal(0, self.genes["r"]))

    def cross(self, genomes):
        """ Crossing multiple genomes during reproduction.

        Each gene is inhereted from one of the parents, chosen at random
        """
        for key in self.genes.keys():
            parent_idx = random.randint(0, 1)
            self.genes[key] = genomes[parent_idx].genes[key]
