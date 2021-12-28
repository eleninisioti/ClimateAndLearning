from agent import Agent
from genome import Genome
from random import choices, randrange
import random
import numpy as np
import math
import copy

class Population:

    def __init__(self, pop_size, survival_type, genome_type, env_mean, init_SD, init_mutate,
                 mutate_rate, extinctions, scale_weights=0):
        self.agents = []
        self.survival_type = survival_type
        self.genome_type = genome_type
        self.extinctions = extinctions
        self.init_SD = init_SD
        self.init_mutate=init_mutate
        self.mutate_rate = mutate_rate
        self.env_mean = env_mean
        self.scale_weights = scale_weights
        for _ in range(pop_size):
            agent_genome = Genome(type=genome_type, env_mean=env_mean, init_SD=init_SD,
                                  init_mutate=init_mutate, mutate_rate=mutate_rate )

            self.agents.append(Agent(genome=agent_genome))

    def compute_fitness(self, agents, env_mean):
        for agent in agents:
            agent.compute_fitness(env_mean)

    def reproduce(self, env):
        if self.survival_type == "FP-Grove":

            # only the half-best agents reproduce and are selected based on their fitness
            self.agents = self.order_agents(self.agents)
            self.agents_reproduce = self.agents[:int(len(self.agents) / 2)]
            weights = [agent.fitness for agent in self.agents_reproduce]
            if self.scale_weights:
                weights =[math.exp(w) for w in weights]
            self.agents_reproduce = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))
            weights = [agent.fitness for agent in self.agents_reproduce]
            if self.scale_weights:
                weights =[math.exp(w) for w in weights]
            # match pairs

            partners_a = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))

            new_agents = []
            for idx, agent in enumerate(self.agents_reproduce):
                agent_genome = Genome(type=self.genome_type, env_mean=self.env_mean, init_SD=self.init_SD,
                                      init_mutate=self.init_mutate, mutate_rate=self.mutate_rate)

                agent_genome.cross([agent.genome, partners_a[idx].genome])
                new_agent = Agent(genome=agent_genome)
                new_agent.mutate()
                #new_agent.is_extinct()
                #new_agent.compute_fitness(env.mean)
                new_agents.append(new_agent)

                if len(self.agents) < env.capacity:
                    # if extinctions are on, fill till maximum population
                    self.agents.append(new_agent)
                else:
                    # replace the worst agents
                    self.agents[idx + int(len(self.agents) / 2)] = new_agent

        elif self.survival_type == "limited-capacity":


            # everyone reproduces
            self.agents_reproduce = self.agents

            # match pairs
            partners_a = choices(self.agents_reproduce, k=len(self.agents_reproduce))
            partners_b = choices(self.agents_reproduce, k=len(self.agents_reproduce))

            if (partners_a != self.agents_reproduce) or (partners_b != self.agents_reproduce):
                print(len(set(partners_b)), len(set(partners_a)))
                print("wrong selection")
            self.agents = []
            for idx, agent in enumerate(self.agents_reproduce):
                agent_genome = Genome(type=self.genome_type, env_mean=self.env_mean, init_SD=self.init_SD,
                                      init_mutate=self.init_mutate, mutate_rate=self.mutate_rate)

                # first child
                agent_genome.cross([agent.genome, partners_a[idx].genome])
                new_agent = Agent(genome=agent_genome)
                new_agent.mutate()
                #new_agent.is_extinct()
                #new_agent.compute_fitness(env)
                #new_agents.append(new_agent)
                if len(self.agents) < (env.capacity):
                    self.agents.append(new_agent)

                # second child
                agent_genome.cross([agent.genome, partners_b[idx].genome])
                new_agent.mutate()
                #new_agent.compute_fitness(env.mean)
                #new_agents.append(new_agent)

                if len(self.agents) < (env.capacity):
                    self.agents.append(new_agent)

        elif self.survival_type == "capacity-fitness":
            self.agents = self.order_agents(self.agents)

            for agent in self.agents:
                agent.assigned = False
            new_agents = []
            keep_agents = copy.copy(self.agents)
            self.agents = []


            # the capacity determines how many will reproduce (in a niche)
            for lat in range(-int(env.num_niches/2) , int(env.num_niches/2) + 1):
                lat_climate = env.mean + 0.01 * lat
                current_agents = []
                for agent in keep_agents:
                    if lat_climate in agent.niches:
                        if not agent.assigned:
                            current_agents.append(agent)

                lat_capacity = int(lat_climate * env.orig_capacity)
                if lat_capacity > 1 and len(current_agents) > 2:
                    current_agents = self.order_agents(current_agents)
                    self.agents_reproduce = current_agents[:int(lat_capacity/2)]
                    #self.agents_reproduce = self.agents_reproduce[:int(len(self.agents_reproduce )/2)]
                    #self.compute_fitness(self.agents_reproduce, lat_climate)

                    weights = [agent.fitness for agent in self.agents_reproduce]

                    self.agents_reproduce = choices(self.agents_reproduce, weights=weights,
                                                    k=len(self.agents_reproduce))

                    for agent in self.agents_reproduce:
                        agent.assigned = True

                    # match pairs
                    weights = [agent.fitness for agent in self.agents_reproduce]
                    partners_a = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))
                    partners_b = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))

                    for idx, agent in enumerate(self.agents_reproduce):
                        agent_genome = Genome(type=self.genome_type, env_mean=self.env_mean, init_SD=self.init_SD,
                                              init_mutate=self.init_mutate, mutate_rate=self.mutate_rate)

                        # first child
                        agent_genome.cross([agent.genome, partners_a[idx].genome])
                        new_agent = Agent(genome=agent_genome)
                        new_agent.mutate()
                        new_agents.append(new_agent)
                        agent.reproduced = True

                        # second child
                        agent_genome.cross([agent.genome, partners_b[idx].genome])
                        new_agent.mutate()
                        new_agents.append(new_agent)
                        agent.reproduced = True

                        if len(new_agents)> lat_capacity:
                            print("error")
                            quit()
            self.agents = new_agents

        elif self.survival_type == "capacity-fitness-v2":
            self.agents = self.order_agents(self.agents)

            # the capacity determines how many will reproduce (in a niche)
            for lat in range(-int(env.num_niches / 2), int(env.num_niches / 2) + 1):
                lat_climate = env.mean + 0.01 * lat
                current_agents = []
                for agent in self.agents:
                    if lat_climate in agent.niches:
                        current_agents.append(agent)


                lat_capacity = int(lat_climate * env.orig_capacity)
                if lat_capacity > 1 and len(current_agents) > 1:

                    current_agents = self.order_agents(current_agents)

                    self.agents_reproduce = current_agents[:lat_capacity]
                    self.agents_reproduce = self.agents_reproduce[:int(len(self.agents_reproduce)/2)]

                    weights = [agent.fitness for agent in self.agents_reproduce]
                    if self.scale_weights:
                        weights = [math.exp(w) for w in weights]
                    self.agents_reproduce = choices(self.agents_reproduce, weights=weights,
                                                    k=len(self.agents_reproduce))


                    if len(self.agents_reproduce) > 2:

                        # match pairs
                        weights = [agent.fitness for agent in self.agents_reproduce]
                        if self.scale_weights:
                            weights = [math.exp(w) for w in weights]
                        partners_a = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))

                        for idx, agent in enumerate(self.agents_reproduce):
                            agent_genome = Genome(type=self.genome_type, env_mean=lat_climate, init_SD=self.init_SD,
                                                  init_mutate=self.init_mutate, mutate_rate=self.mutate_rate)

                            # first child

                            agent_genome.cross([agent.genome, partners_a[idx].genome])
                            new_agent = Agent(genome=agent_genome)
                            new_agent.mutate()
                            # new_agent.compute_fitness(lat_climate)
                            if len(self.agents) < lat_capacity:
                                # if extinctions are on, fill till maximum population
                                self.agents.append(new_agent)
                            else:
                                # replace the worst agents
                                self.agents[idx + int(len(self.agents) / 2)] = new_agent

        elif self.survival_type == "capacity-fitness-v3":
            self.agents = self.order_agents(self.agents)

            for agent in self.agents:
                agent.assigned = False
            new_agents = []
            keep_agents = copy.copy(self.agents)
            #self.agents = []


            # the capacity determines how many will reproduce (in a niche)
            for lat in range(-int(env.num_niches/2) , int(env.num_niches/2) + 1):
                lat_climate = env.mean + 0.01 * lat
                current_agents = []
                for agent in keep_agents:
                    if lat_climate in agent.niches:
                        if not agent.assigned:
                            current_agents.append(agent)

                lat_capacity = int(lat_climate * env.orig_capacity)
                if lat_capacity > 1 and len(current_agents) > 2:
                    current_agents = self.order_agents(current_agents)
                    self.agents_reproduce = current_agents[:int(lat_capacity/2)]
                    #self.agents_reproduce = self.agents_reproduce[:int(len(self.agents_reproduce )/2)]
                    #self.compute_fitness(self.agents_reproduce, lat_climate)

                    weights = [agent.fitness for agent in self.agents_reproduce]

                    self.agents_reproduce = choices(self.agents_reproduce, weights=weights,
                                                    k=len(self.agents_reproduce))

                    for agent in self.agents_reproduce:
                        agent.assigned = True

                    # match pairs
                    weights = [agent.fitness for agent in self.agents_reproduce]
                    partners_a = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))
                    partners_b = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))

                    for idx, agent in enumerate(self.agents_reproduce):
                        agent_genome = Genome(type=self.genome_type, env_mean=self.env_mean, init_SD=self.init_SD,
                                              init_mutate=self.init_mutate, mutate_rate=self.mutate_rate)

                        # first child
                        agent_genome.cross([agent.genome, partners_a[idx].genome])
                        new_agent = Agent(genome=agent_genome)
                        new_agent.mutate()
                        #new_agents.append(new_agent)
                        if len(self.agents) < lat_capacity:
                            # if extinctions are on, fill till maximum population
                            self.agents.append(new_agent)
                        else:
                            # replace the worst agents
                            self.agents[idx + int(len(self.agents) / 2)] = new_agent
                        agent.reproduced = True


            #self.agents = new_agents





    def reproduce_outdated(self, env):
        # choose which ones will reproduce

        if self.survival_type == "mixed":
            new_agents = []

            # the capacity determines how many will reproduce (in a niche)
            for lat in range(-int(env.num_niches / 2), int(env.num_niches / 2)):
                lat_climate = env.mean + 0.01 * lat
                current_agents = []
                for agent in self.agents:
                    if lat_climate in agent.niches:
                        current_agents.append(agent)
                lat_capacity = int(lat_climate*env.orig_capacity)

                if lat_capacity > 1 and len(current_agents) > 1:
                    self.agents_reproduce = current_agents[:lat_capacity]
                    # match pairs
                    weights = [agent.fitness for agent in self.agents_reproduce]
                    partners_a = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))

                    for idx, agent in enumerate(self.agents_reproduce):
                        agent_genome = Genome(type=self.genome_type, env_mean=self.env_mean, init_SD=self.init_SD,
                                              init_mutate=self.init_mutate, mutate_rate=self.mutate_rate)

                        # first child
                        agent_genome.cross([agent.genome, partners_a[idx].genome])
                        new_agent = Agent(genome=agent_genome)
                        new_agent.mutate()
                        #new_agent.compute_fitness(env)
                        new_agents.append(new_agent)

            self.agents = new_agents

        if self.survival_type == "no-pressure":
            self.agents_reproduce = self.agents

            # match pairs
            weights = [agent.fitness for agent in self.agents_reproduce]
            partners_a = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))
            partners_b = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))

            new_agents = []
            for idx, agent in enumerate(self.agents_reproduce):
                agent_genome = Genome(type=self.genome_type, env_mean=self.env_mean, init_SD=self.init_SD,
                                  init_mutate=self.init_mutate, mutate_rate=self.mutate_rate)

                # first child
                agent_genome.cross([agent.genome, partners_a[idx].genome])
                new_agent = Agent(genome=agent_genome)
                new_agents.append(new_agent)

                # second child
                agent_genome.cross([agent.genome, partners_b[idx].genome])
                new_agent = Agent(genome=agent_genome)
                new_agents.append(new_agent)

            self.agents = []
            for new_agent in new_agents:
                new_agent.mutate()
                new_agent.compute_fitness(env)

                for idx, niche in enumerate(new_agent.niches):
                    remain_cap = new_agent.capacities[idx]
                    for agent in self.agents:
                        if niche in agent.niches:
                            remain_cap -= 1
                    if remain_cap > 1:
                        self.agents.append(new_agent)
                        break

            if len(self.agents) > (env.orig_capacity*env.num_niches):
                print("error")

        elif self.survival_type == "FP-global":
            # competition across all niches
            self.agents_reproduce = self.agents[:int(len(self.agents)/2*3)]

            # match pairs
            weights = [agent.fitness for agent in self.agents_reproduce]
            partners_a = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))
            partners_b = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))
            new_agents = []
            for idx, agent in enumerate(self.agents_reproduce):
                agent_genome = Genome(type=self.genome_type, env_mean=self.env_mean, init_SD=self.init_SD,
                                      init_mutate=self.init_mutate, mutate_rate=self.mutate_rate)

                # first child
                agent_genome.cross([agent.genome, partners_a[idx].genome])
                new_agent = Agent(genome=agent_genome)
                new_agents.append(new_agent)

                # second child
                agent_genome.cross([agent.genome, partners_b[idx].genome])
                new_agent = Agent(genome=agent_genome)
                new_agents.append(new_agent)

            inserted = 0
            for new_agent in new_agents:
                new_agent.mutate()
                new_agent.compute_fitness(env)


                for idx, niche in enumerate(new_agent.niches):

                    if inserted < int(len(self.agents)):
                        self.agents[inserted] = new_agent
                        inserted += 1
                    else:
                        remain_cap = new_agent.capacities[idx]
                        for agent in self.agents:
                            if niche in agent.niches:
                                remain_cap -= 1
                        if remain_cap > 1:
                            self.agents.append(new_agent)
                            break

            if len(self.agents) > (env.orig_capacity*env.num_niches):
                print("error")

        elif self.survival_type == "no-pressure-random":
            self.agents_reproduce = self.agents[:int(len(self.agents)/2)]

            # match pairs
            weights = [agent.fitness for agent in self.agents_reproduce]
            partners_a = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))
            partners_b = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))

            new_agents = []
            for idx, agent in enumerate(self.agents_reproduce):
                agent_genome = Genome(type=self.genome_type, env_mean=self.env_mean, init_SD=self.init_SD,
                                  init_mutate=self.init_mutate, mutate_rate=self.mutate_rate)


                # second child
                agent_genome.cross([agent.genome, partners_b[idx].genome])
                new_agent = Agent(genome=agent_genome)
                new_agent.mutate()
                new_agent.compute_fitness(env)
                new_agents.append(new_agent)

            for new_agent in new_agents:
                self.agents.append(new_agent)

            while len(self.agents) > env.capacity:
                self.agents.pop(randrange(len(self.agents)))

        elif self.survival_type == "FP-Grove":
            self.agents_reproduce = self.agents[:int(len(self.agents)/2)]

            # match pairs
            weights = [agent.fitness for agent in self.agents_reproduce]
            partners_a = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))

            new_agents = []
            for idx, agent in enumerate(self.agents_reproduce):
                agent_genome = Genome(type=self.genome_type, env_mean=self.env_mean, init_SD=self.init_SD,
                                  init_mutate=self.init_mutate, mutate_rate=self.mutate_rate)

                # first child
                agent_genome.cross([agent.genome, partners_a[idx].genome])
                new_agent = Agent(genome=agent_genome)
                new_agent.mutate()
                new_agent.compute_fitness(env)
                new_agents.append(new_agent)

                # second child
                #agent_genome.cross([agent.genome, partners_b[idx].genome])
                #new_agent = Agent(genome=agent_genome)
                #new_agents.append(new_agent)

                # replace the worst agents


                if len(self.agents) < env.capacity:
                    self.agents.append(agent)
                else:
                    self.agents[idx + int(len(self.agents) / 2)] = new_agent

            #self.agents = []
            #for new_agent in new_agents:
                #if len(self.agents) < (env.orig_capacity*env.num_niches):
                    #self.agents.append(new_agent)


    #elif self.survival_type == "FP-niched":
            # competition only within niche
            # only half within a


    def order_agents(self, agents):
        # order agents based on fitness
        fitness_values = [agent.fitness for agent in agents]
        keydict = dict(zip(agents, fitness_values))
        agents.sort(key=keydict.get, reverse=True)
        return agents

    def survive(self, env):
        """ Compute the fitness of a generation and rank individuals.
        """
        self.num_extinctions = 0
        new_agents = []
        for agent in self.agents:
            #agent.compute_fitness(env)
            extinct, agent = agent.is_extinct(env)
            if extinct and self.extinctions:
                #self.agents.remove(agent)
                self.num_extinctions += 1
            else:
                new_agents.append(agent)

        self.agents = new_agents
        for idx, agent in enumerate(self.agents):
            if len(agent.niches) == 0:
                print("check")

    def has_mass_extinction(self):
        if len(self.agents) < 2:
            print("Mass extinction. Terminating program.")
            return True
        return False



