from scipy import stats
import numpy as np


class Agent:
    """ Class implementing an agent, modeling an individual in the population.
    """

    def __init__(self, genome):
        """ Class constructor
        
        Parameters
        ----------

        genome: Genome
            the agent's genome
        """
        self.genome = genome
        self.reproduced = False

    def mutate(self):
        self.genome.mutate()

    def compute_fitness(self, env_state):
        """ Compute the fitness of the individual in a given environmental state.

        A genome is a gaussian distribution that we sample at the given environmental state.

        Parameters
        ----------
        env_mean: float
            the state of the environment
        """
        return stats.norm(self.genome.genes["mean"], self.genome.genes["sigma"]).pdf(env_state)

    def is_extinct(self, env):
        """ Detect whether an agent goes extinct in a given environment.

        This happens when an agent's preferred state is more than two standard deviations away from the
        environment's state. We examine survival in each niche independently mploy a latitudinal model with N/2 
        northern and N/2 southern niches

        Parameters
        ----------
        env: Environment
            the environment
        """
        survival = 0
        self.niches = []
        self.fitness_values = []
        for niche_idx, niche_info in env.niches.items():
            niche_climate = niche_info["climate"]
            if ((self.genome.genes["mean"] - 2 * self.genome.genes["sigma"]) < niche_climate) \
                    and (niche_climate < self.genome.genes["mean"] + 2 * self.genome.genes["sigma"]):
                survival += 1
                self.niches.append(niche_climate)
                self.fitness_values.append(self.compute_fitness(niche_climate))

        # the fitness of an agent is the average over its fitnesses in niches where the agent survives
        if len(self.fitness_values):
            self.fitness = np.mean(self.fitness_values)
        else:
            self.fitness = 0

        return not survival



