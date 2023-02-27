from scipy import stats
import numpy as np
import random

class Agent:
    """ Class implementing an agent/individual in the population.
    """

    def __init__(self, genome, history_window):
        """ Class constructor
        
        Parameters
        ----------

        genome: Genome
            the agent's genome
        """
        self.genome = genome
        self.reproduced = 0
        self.movement = 999
        self.closest_niche = 999
        self.history = []
        self.history_window = history_window


    def mutate(self, stopped_NC):
        self.genome.mutate(stopped_NC)

    def set_history(self, history):
        self.history = history[-self.history_window:]

    def compute_fitness(self, env_state):
        """ Compute the fitness of the individual in a given environmental state.

        A genome is a gaussian distribution that we sample at the given environmental state.

        Parameters
        ----------
        env_state: float
            the state of the environment
        """
        if self.genome.type == "intrinsic":
            fitness = 0
            for el in self.genome.genes["intrinsic_curves"]:
                fitness += stats.norm(el, self.genome.genes["sigma"]).pdf(env_state)

            return fitness

        else:
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
        self.niches_lat = []

        closest_niche = 999
        min_distance = 999

        for niche_idx, niche_info in env.niches.items():
            niche_climate = niche_info["climate"]

            if self.genome.type == "intrinsic":
                # find curve closest to this niche
                min_distance_intrinsic = 999
                for el in self.genome.genes["intrinsic_curves"]:
                    if np.abs(niche_climate-el) < min_distance_intrinsic:
                        min_distance_intrinsic =np.abs(niche_climate-el)
                        main_niche = el
            else:
                main_niche = self.genome.genes["mean"]

            if ((main_niche - 2 * self.genome.genes["sigma"]) < niche_climate) \
                    and (niche_climate < main_niche+ 2 * self.genome.genes["sigma"]):
                survival += 1
                self.niches.append(niche_climate)
                self.niches_lat.append(niche_info["lat"])

                fitness = self.compute_fitness(niche_climate)

                self.fitnesses[niche_climate] = fitness
            else:
                self.fitnesses[niche_climate] = 0

            if np.abs(main_niche- niche_climate) < min_distance:
                closest_niche = niche_info["lat"]
                min_distance = np.abs(main_niche- niche_climate)
        if len(self.niches):

            self.realized_niche = random.choice(self.niches_lat)
        else:
            self.realized_niche = None
        self.closest_niche = closest_niche

        return not survival



