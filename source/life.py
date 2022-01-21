from change_environment import ChangeEnv
from sin_environment import SinEnv
from maslin_environment import MaslinEnv
import numpy as np
from population import Population
from logger import Logger


class Life:

    def __init__(self, args):
        self.config = args  # contains all configuration necessary for the experiment

    def setup(self):
        # ----- set up environment ----
        if self.config.env_type == "change":
            self.env = ChangeEnv(mean=self.config.climate_mean_init,
                                 ref_capacity=self.config.capacity,
                                 num_niches=self.config.num_niches,
                                 factor_time_abrupt=self.config.factor_time_abrupt,
                                 factor_time_steady=self.config.factor_time_steady)

        elif self.config.env_type == "sin":
            self.env = SinEnv(period=self.config.climate_period,
                              orig_capacity=self.config.capacity / 2,
                              num_niches=self.config.num_niches)

        elif self.config.env_type == "combined":
            self.env = MaslinEnv(mean=self.config.climate_mean_init,
                                 ref_capacity=self.config.capacity ,
                                 num_niches=self.config.num_niches,
                                 factor_time_abrupt=self.config.factor_time_abrupt,
                                 factor_time_variable=self.config.factor_time_variable,
                                 factor_time_steady=self.config.factor_time_steady,
                                 var_freq=self.config.var_freq,
                                 var_SD=self.config.var_SD)
        # -------------------------------------------------------------------------
        # ----- set up population ----
        pop_size = int(np.min([self.config.init_num_agents, self.env.current_capacity*self.config.num_niches]))
        self.population = Population(pop_size=pop_size,
                                     selection_type=self.config.selection_type,
                                     genome_type=self.config.genome_type,
                                     env_mean=self.env.mean,
                                     init_SD=self.config.init_SD,
                                     init_mutate=self.config.init_mutate,
                                     mutate_mutate_rate=self.config.mutate_mutate_rate,
                                     extinctions=self.config.extinctions)
        # -------------------------------------------------------------------------
        self.logger = Logger(trial=self.config.trial, env=self.env)

    def run(self):
        """ Main routine that simulates the evolution of a population in a varying environment.
        """

        # prepare environment and population
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
                    break

                # reproduce population
                self.population.reproduce(self.env)

                if gen % 10 == 0:
                    # print progress
                    print("Generation: ", gen, len(self.population.agents), " agents")

        # collect all information for logging
        self.logger.final_log()

        return self.logger.log, self.logger.env_profile, self.logger.log_niches
