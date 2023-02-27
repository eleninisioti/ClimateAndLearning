from sin_environment import SinEnv
from stable_environment import StableEnv
from noisy_environment import NoisyEnv
import numpy as np
from population import Population
from logger import Logger
import time


class Life:
    """ Class responsible for simulating the evolution of a population in a changing environment.
    """

    def __init__(self, args):
        self.config = args

        # ----- set up environment -----
        if self.config.env_type == "sin":
            self.env = SinEnv(mean=self.config.climate_mean_init,
                              ref_capacity=self.config.capacity,
                              num_niches=self.config.num_niches,
                              period=self.config.period,
                              amplitude=self.config.amplitude,
                              decay_construct=self.config.decay_construct)

        elif self.config.env_type == "stable":
            self.env = StableEnv(mean=self.config.climate_mean_init,
                                 ref_capacity=self.config.capacity,
                                 num_niches=self.config.num_niches,
                              decay_construct=self.config.decay_construct)

        elif self.config.env_type == "noisy":
            self.env = NoisyEnv(mean=self.config.climate_mean_init,
                                std=self.config.noise_std,
                                ref_capacity=self.config.capacity,
                                num_niches=self.config.num_niches,
                              decay_construct=self.config.decay_construct)
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
                                     history_window=self.config.history_window)
        # -------------------------------------------------------------------------
        self.logger = Logger(trial=self.config.trial,
                             env=self.env,
                             max_pop=int(self.config.capacity*self.env.max_value))


    def run(self):
        """ Simulates the evolution of a population in a varying environment.
        """
        start_time = time.time()
        niche_constructions = self.env.niche_constructions

        stopped_NC = False
        # ----- run generations ------
        for gen in range(self.config.num_gens):
            start_time_gen = time.time()

            # update environment
            self.env.step(gen, niche_constructions)

            if not self.config.only_climate:

                if self.config["stop_NC_every"]:
                    if gen%self.config["stop_NC_every"] ==0:
                        print("stopping NC at ", str(gen))
                        stop_NC_counter = 0
                        stopped_NC = True
                        keep_NC = [agent.genome.genes["c"] for agent in self.population.agents]
                        for agent_idx, agent in enumerate(self.population.agents):
                            agent.genomes.genes["c"] = 0
                            self.population.agents[agent_idx] = agent

                if stopped_NC:
                    stop_NC_counter += 1
                    if stop_NC_counter == self.config["stop_NC_for"]:
                        print("continuing NC at ", str(gen))
                        for agent_idx, agent in enumerate(self.population.agents):
                            agent.genomes.genes["c"] = keep_NC[agent_idx]
                            self.population.agents[agent_idx] = agent

                # compute fitness of population
                self.population.survive(self.env)

                # compute metrics for new generation
                self.logger.log_gen(self.population, self.env)

                print("this gen took ", (time.time() - start_time_gen))

                time_out = (time.time() - start_time) > self.config.time_budget

                if self.population.has_mass_extinction() or time_out:
                    self.logger.log_gen(self.population, self.env)
                    if time_out:
                        print("Time out. Exiting simulation")
                    break

                # reproduce population
                niche_constructions = self.population.reproduce(self.env)

                if gen % 1 == 0:
                    # print progress
                    print("Generation: ", gen, len(self.population.agents), " agents")

        # collect all information for logging
        self.logger.final_log()

        return self.logger.log, self.logger.log_niches
