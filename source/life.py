from sin_environment import SinEnv
from stable_environment import StableEnv
from noisy_environment import NoisyEnv
import numpy as np
from population import Population
from logger import Logger


class Life:

    def __init__(self, args):
        self.config = args  # contains all configuration necessary for the experiment

    def setup(self):
        # ----- set up environment -----
        if self.config.env_type == "sin":
            self.env = SinEnv(mean=self.config.climate_mean_init,
                              ref_capacity=self.config.capacity,
                              num_niches=self.config.num_niches,
                              period=self.config.period,
                              amplitude=self.config.amplitude)

        elif self.config.env_type == "stable":
            self.env = StableEnv(mean=self.config.climate_mean_init,
                                 ref_capacity=self.config.capacity,
                                 num_niches=self.config.num_niches)

        elif self.config.env_type == "noisy":
            self.env = NoisyEnv(mean=self.config.climate_mean_init,
                                std=self.config.noise_std,
                                ref_capacity=self.config.capacity,
                                num_niches=self.config.num_niches)
        # -------------------------------------------------------------------------
        # ----- set up population -----
        pop_size = int(np.min([self.config.init_num_agents, self.env.current_capacity * self.config.num_niches]))
        self.population = Population(pop_size=pop_size,
                                     selection_type=self.config.selection_type,
                                     genome_type=self.config.genome_type,
                                     env_mean=self.env.climate,
                                     init_sigma=self.config.init_sigma,
                                     init_mutate=self.config.init_mutate,
                                     mutate_mutate_rate=self.config.mutate_mutate_rate,
                                     extinctions=self.config.extinctions,
                                     mean_fitness = self.config.mean_fitness,
                                     reproduce_once=self.config.reproduce_once)
        # -------------------------------------------------------------------------
        self.logger = Logger(trial=self.config.trial, env=self.env)

    def run(self):
        """ Main routine that simulates the evolution of a population in a varying environment.
        """

        # prepare environment and population
        self.setup()

        # ----- run generations ------
        for gen in range(self.config.num_gens):

            # update environment
            self.env.step(gen)

            if not self.config.only_climate:

                # compute fitness of population
                self.population.survive(self.env)

                # compute metrics for new generation
                self.logger.log_gen(self.population)

                if self.population.has_mass_extinction():
                    break

                # reproduce population
                self.population.reproduce(self.env)

                if gen % 1 == 0:
                    # print progress
                    print("Generation: ", gen, len(self.population.agents), " agents")

        # collect all information for logging
        self.logger.final_log()

        return self.logger.log, self.logger.log_niches
