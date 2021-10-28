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
            self.env = ChangeEnv(self.config.climate_mean_init, self.config.capacity)

        elif self.config.env_type == "sin":
            self.env = SinEnv(self.config.climate_period, self.config.capacity)

        elif self.config.env_type == "combined":
            self.env = CombEnv(orig_capacity=self.config.capacity, model=self.config.model,
                               factor_time_abrupt=self.config.factor_time_abrupt,
                               factor_time_variable=self.config.factor_time_variable,
                               factor_time_steady=self.config.factor_time_steady,
                               var_freq=self.config.var_freq, var_SD=self.config.var_SD,
                               irregular=self.config.irregular)

        pop_size = int(np.min([self.config.num_agents, self.env.capacity]))
        self.population = Population(pop_size=pop_size,
                                     survival_type=self.config.survival_type,
                                     genome_type=self.config.genome_type,
                                     env_mean=self.env.mean,
                                     init_SD=self.config.init_SD,
                                     init_mutate=self.config.init_mutate,
                                     mutate_rate=self.config.mutate_rate)
        self.logger = Logger()

    def run(self):

        self.setup()

        # ----- run generations ------
        for gen in range(self.config.num_gens):

            # update environment
            self.env.step(gen)

            if not self.config.only_climate:

                # compute fitness of population
                self.population.survive(self.env)

                if self.population.has_mass_extinction():
                    # compute metrics for new generation
                    self.logger.log_gen(self.population)
                    break

                self.population.reproduce(self.env)

                # compute metrics for new generation
                self.logger.log_gen(self.population)

            if gen % 100 == 0:
                print("Generation: ", gen)

        self.logger.final_log(self.env)

        return self.logger.log
