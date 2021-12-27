""" Contains the main simulation.
"""
from change_environment import ChangeEnv
from sin_environment import SinEnv
from combined_environment import CombEnv
from agent import Agent
from plotter import Plotter
from numpy.random import normal
import random
import numpy as np
from numpy.random import choice
from species import Species
from genome import Genome
from population import Population
from logger import Logger


class Life:

    def __init__(self, args):
        self.config = args

    def setup(self):
        # ----- set up ----
        if self.config.env_type == "change":
            self.env = ChangeEnv(self.config.climate_mean_init, self.config.capacity/2,
                                 num_niches=self.config.num_niches,
                                 factor_time_abrupt=self.config.factor_time_abrupt,
                                 factor_time_steady=self.config.factor_time_steady
                                 )

        elif self.config.env_type == "sin":
            self.env = SinEnv(self.config.climate_period,
                              self.config.capacity/2,
                              num_niches=self.config.num_niches)

        elif self.config.env_type == "combined":
            self.env = CombEnv(orig_capacity=self.config.capacity/2, model=self.config.model,
                               factor_time_abrupt=self.config.factor_time_abrupt,
                               factor_time_variable=self.config.factor_time_variable,
                               factor_time_steady=self.config.factor_time_steady,
                               var_freq=self.config.var_freq, var_SD=self.config.var_SD,
                               irregular=self.config.irregular, low_value= self.config.low_value,
                               num_niches=self.config.num_niches)

        pop_size = int(np.min([self.config.num_agents, self.env.capacity]))
        self.population = Population(pop_size=pop_size,
                                     survival_type=self.config.survival_type,
                                     genome_type=self.config.genome_type,
                                     env_mean=self.env.mean,
                                     init_SD=self.config.init_SD,
                                     init_mutate=self.config.init_mutate,
                                     mutate_rate=self.config.mutate_rate,
                                     extinctions=self.config.extinctions,
                                     scale_weights=self.config.scale_weights)
        self.logger = Logger()

    def run(self):

        self.setup()

        # ----- run generations ------
        for gen in range(self.config.first_gen, self.config.num_gens):

            # update environment
            self.env.step(gen)

            if not self.config.only_climate:

                # compute fitness of population
                self.population.survive(self.env)

                # compute metrics for new generation
                self.logger.log_gen(self.population)

                if self.population.has_mass_extinction():
                    # compute metrics for new generation
                    self.logger.log_gen(self.population)
                    break

                self.population.reproduce(self.env)



                if gen % 10 == 0:
                    print("Generation: ", gen, len(self.population.agents), " agents with R:", self.logger.log[
                        "running_mutate"][-1], " extinctions ", self.logger.log[
                        "extinctions"][-1])

        self.logger.final_log(self.env)

        return self.logger.log
