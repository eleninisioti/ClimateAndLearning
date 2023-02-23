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

        init_sigma: float
            determines the range around env_mean for genome initialization

        init_mutate: float
            the mutation rate upon initialisation

        mutate_mutate_rate: float
            the mutation rate of the mutation rate

        """
        self.max_c = 100/5000 #number of niches/max population
        self.type = genome_type
        self.mutate_mutate_rate = mutate_mutate_rate
        mu = normal(env_mean, init_sigma)
        r = init_mutate

        if self.type == "intrinsic":
            sigma = np.abs(normal(0, 0.001))

            num_intrinsic_curves = 10
            intrinsic_curves = [mu for _ in range(num_intrinsic_curves)]
            self.genes = {"intrinsic_curves": intrinsic_curves, "r": r, "sigma": sigma}

        elif self.type == "niche-construction" or self.type == "niche-construction-v2":
            sigma = np.abs(normal(0, init_sigma))
            self.genes = {"mean": mu, "sigma": sigma, "r": r, "c": 0}
        else:
            sigma = np.abs(normal(0, init_sigma))
            self.genes = {"mean": mu, "sigma": sigma, "r": r}

    def mutate(self):
        """ Applies mutations to the genome.
        """
        if self.type == "no-evolv":
            # constant mutation rate
            self.genes["mean"] = self.genes["mean"] + normal(0, self.genes["r"])
            self.genes["sigma"] = np.abs(self.genes["sigma"] + normal(0, self.genes["r"]))

        elif self.type == "evolv":
            # all genes evolve with the same (evolving) mutation rate
            self.genes["mean"] = self.genes["mean"] + normal(0, self.genes["r"])
            self.genes["sigma"] = np.abs(self.genes["sigma"] + normal(0, self.genes["r"]))
            self.genes["r"] = np.abs(self.genes["r"] + normal(0, self.genes["r"]))

        elif self.type == "intrinsic":
            self.genes["r"] = np.abs(self.genes["r"] + normal(0, self.genes["r"]))
            self.genes["intrinsic_curves"] = [el + normal(0, self.genes["r"])  for el in self.genes["intrinsic_curves"]]

        elif self.type == "niche-construction" or self.type == "niche-construction-v2":
            # all genes evolve with the same (evolving) mutation rate
            self.genes["mean"] = self.genes["mean"] + normal(0, self.genes["r"])
            self.genes["r"] = np.abs(self.genes["r"] + normal(0, self.genes["r"]))
            self.genes["c"] = np.min([self.genes["c"] + normal(0, 0.0001), self.max_c])
            self.genes["sigma"] = np.abs(self.genes["sigma"] + normal(0, self.genes["r"]))



    def cross(self, genomes):
        """ Crossing two genomes (during reproduction).

        Each gene is inherited from one of the parents, chosen at random.

        Parameters
        ----------
        genomes: list of Gene
            the parents' genomes
        """
        for key in self.genes.keys():
            parent_idx = random.randint(0, 1)
            self.genes[key] = genomes[parent_idx].genes[key]
