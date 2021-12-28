from scipy import stats
import numpy as np


class Agent:

    def __init__(self, genome):
        self.genome = genome
        self.ancestry = []

    def mutate(self):
        self.genome.mutate()

    def compute_fitness(self, env_mean):
        """ Compute the fitness of the individual in a given niche.

        A genome is a gaussian that we sample at the given environment. For more info, see 'Grove et al, 
        Evolution and dispersal under climatic instability: a simple evolutionary algorithm'.

        Parameters
        ----------
        env_mean: float
            the state of the environment
        """
        self.fitness = stats.norm(self.genome.genes["mean"], self.genome.genes["sigma"]).pdf(env_mean)
        return self.fitness

    def is_extinct(self, env):
        """ Detect whether an agent goes extinct in a given environment.

        This happens when an agent's preferred niche is more than two standard deviations away from the
        actual niche. For the latitudial model employed here see 'Grove et al,
        Evolution and dispersal under climatic instability: a simple evolutionary algorithm'

        Parameters
        ----------
        env: Env
            the environment
        """
        survival = 0
        self.niches = []
        self.fitness_values = []
        num_latitudes = env.num_niches
        for lat in range(-int(num_latitudes/2), int(num_latitudes/2) + 1):
            lat_climate = env.mean + 0.01 * lat

            if ((self.genome.genes["mean"] - 2 * self.genome.genes["sigma"]) < lat_climate) \
                    and (lat_climate < self.genome.genes["mean"] + 2 * self.genome.genes["sigma"]):
                survival += 1
                self.niches.append(lat_climate)
                self.fitness_values.append(self.compute_fitness(lat_climate))

        # the fitness is the average over its fitnesses in niches where the agent survives
        if len(self.fitness_values):
            self.fitness = np.mean(self.fitness_values)
        else:
            self.fitness = 0

        return not survival



