import numpy as np
from numpy.random import normal
import random


class Genome:

    def __init__(self, genome_type, env_mean, init_sigma, init_mutate, mutate_mutate_rate):
        """ Constructor of Class Genome

        Parameters
        ---------
        genome_type: str
            determines the structure and mutation rule of the genome. choose between '1D',
             '1D_mutate','1D_mutate_fixed'

        env_mean: float
            the state of the environment upon initialization of the genoem

        init_sd: float
            determines the range around env_mean for genome initialization

        init_mutate: float
            the mutation rate upon initialisation

        mutate_mutate_rate: float
            the mutation rate of the mutation rate

        """
        self.type = genome_type
        self.mutate_mutate_rate = mutate_mutate_rate
        mu = normal(env_mean, init_sigma)
        sigma = np.abs(normal(0, init_sigma))
        r = init_mutate
        self.genes = {"mean": mu, "sigma": sigma, "r": r}

    def mutate(self):
        """ Applies mutations to the genome.
        """
        if self.type == "no-evolv":
            # constant mutation rate
            self.genes["mean"] = self.genes["mean"] + normal(0, self.genes["r"])
            self.genes["sigma"] = np.abs(self.genes["sigma"] + normal(0, self.genes["r"]))
        elif self.type == "evolv-fixed":
            # mutation rate evolves with constant mutation rate
            self.genes["mean"] = self.genes["mean"] + normal(0, self.genes["r"])
            self.genes["sigma"] = np.abs(self.genes["sigma"] + normal(0, self.genes["r"]))
            self.genes["r"] = np.abs(self.genes["r"] + normal(0, self.mutate_mutate_rate))
        elif self.type == "evolv":
            # all genes evolve with the same (evolving) mutation rate
            self.genes["mean"] = self.genes["mean"] + normal(0, self.genes["r"])
            self.genes["sigma"] = np.abs(self.genes["sigma"] + normal(0, self.genes["r"]))
            self.genes["r"] = np.abs(self.genes["r"] + normal(0, self.genes["r"]))

    def cross(self, genomes):
        """ Crossing two genomes (during reproduction)

        Each gene is inherited from one of the parents, chosen at random.

        Parameters
        ----------
        genomes: list of Gene
            the parents' genomes
        """
        for key in self.genes.keys():
            parent_idx = random.randint(0, 1)
            self.genes[key] = genomes[parent_idx].genes[key]
