from scipy import stats
import numpy as np


class Agent:
    """ Class implementing an agent/individual in the population.
    """

    def __init__(self, genome):
        """ Class constructor
        
        Parameters
        ----------

        genome: Genome
            the agent's genome
        """
        self.genome = genome
        self.reproduced = 0

    def mutate(self):
        self.genome.mutate()

    def compute_fitness(self, env_state):
        """ Compute the fitness of the individual in a given environmental state.

        A genome is a gaussian distribution that we sample at the given environmental state.

        Parameters
        ----------
        env_state: float
            the state of the environment
        """
        return stats.norm(self.genome.genes["mean"], self.genome.genes["sigma"]).pdf(env_state)

    def is_extinct(self, env):
        """ Detect whether an agent goes extinct in a given environment. Also compute the fitness of the agent in
        each niche.

        An agent goes extinct when its preferred state is more than two standard deviations away from the
        environment's state. We examine survival in each niche independently employ a latitudinal model with N/2
        northern and N/2 southern niches

        Parameters
        ----------
        env: Environment
            the environment
        """
        survival = 0
        self.niches = []
        self.fitnesses = {}

        for niche_idx, niche_info in env.niches.items():
            niche_climate = niche_info["climate"]
            if ((self.genome.genes["mean"] - 2 * self.genome.genes["sigma"]) < niche_climate) \
                    and (niche_climate < self.genome.genes["mean"] + 2 * self.genome.genes["sigma"]):
                survival += 1
                self.niches.append(niche_climate)
                fitness = self.compute_fitness(niche_climate)

                self.fitnesses[niche_climate] = fitness
            else:
                self.fitnesses[niche_climate] = 0

        return not survival



