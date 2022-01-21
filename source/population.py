""" This script contains the implementation of a population.
"""
from agent import Agent
from genome import Genome
from random import choices
import copy

class Population:

    def __init__(self, pop_size, selection_type, genome_type, env_mean, init_SD, init_mutate,
                 mutate_mutate_rate, extinctions):
        """ Class Population constructor.

        Parameters
        ----------
        pop_size: int
            number of agents

        selection_type: str
            determines how agents are selected for reproduction at the end of a generation

        genome_type: str
            type of genome

        env_mean: float
            environmental state upon initialization (even if there are multiple niches, everyone is initialized at
            niche 0)

        init_SD: float
            determines the range around env_mean for genome initialization

        init_mutate: float
            the mutation rate upon initialisation

        mutate_mutate_rate: float
            the mutation rate of the mutation rate

        extinctions: int
            If one, extinct individuals are discarded from the population

        """
        self.agents = []
        self.selection_type = selection_type

        # keep these variables for initializing future agents
        self.genome_type = genome_type
        self.extinctions = extinctions
        self.init_SD = init_SD
        self.init_mutate=init_mutate
        self.mutate_mutate_rate = mutate_mutate_rate
        self.env_mean = env_mean
        for _ in range(pop_size):
            agent_genome = Genome(type=genome_type,
                                  env_mean=env_mean,
                                  init_SD=init_SD,
                                  init_mutate=init_mutate,
                                  mutate_mutate_rate=mutate_mutate_rate)

            self.agents.append(Agent(genome=agent_genome))

    def compute_fitness(self, agents, env_mean):
        """ Compute the fitness of agents in a niche.

        Parameters
        ----------
        agents: list of Agent
            agents to compute fitness for
        env_mean: float
            the state of the environment
        """
        for agent in agents:
            agent.compute_fitness(env_mean)

    def reproduce(self, env):
        """ Population reproduction at the end of a generation.

        Parameters
        ----------
        env: Env
            the current environment
        """
        if self.selection_type == "FP-Grove":

            # only consider the half-best agents, selected proportionally to their fitness
            self.agents = self.order_agents(self.agents)
            self.agents_reproduce = self.agents[:int(len(self.agents) / 2)]
            weights = [agent.fitness for agent in self.agents_reproduce]
            self.agents_reproduce = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))
            weights = [agent.fitness for agent in self.agents_reproduce]

            # find partners
            partners_a = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))

            new_agents = []
            for idx, agent in enumerate(self.agents_reproduce):
                agent_genome = Genome(type=self.genome_type,
                                      env_mean=self.env_mean,
                                      init_SD=self.init_SD,
                                      init_mutate=self.init_mutate,
                                      mutate_mutate_rate=self.mutate_mutate_rate) # could
                # initialize with any genome here
                agent_genome.cross([agent.genome, partners_a[idx].genome]) # sexual crossing
                new_agent = Agent(genome=agent_genome)
                new_agent.mutate()
                new_agents.append(new_agent)

                if len(self.agents) < env.current_capacity*env.num_niches:
                    # if there is still room, fill till maximum population
                    self.agents.append(new_agent)
                else:
                    # replace the worst agents
                    self.agents[idx + int(len(self.agents) / 2)] = new_agent

        elif self.selection_type == "limited-capacity":

            # everyone is considered for reproduction
            self.agents_reproduce = self.agents

            # find two partners
            partners_a = choices(self.agents_reproduce, k=len(self.agents_reproduce))
            partners_b = choices(self.agents_reproduce, k=len(self.agents_reproduce))

            self.agents = []
            for idx, agent in enumerate(self.agents_reproduce):
                agent_genome = Genome(type=self.genome_type,
                                      env_mean=self.env_mean,
                                      init_SD=self.init_SD,
                                      init_mutate=self.init_mutate,
                                      mutate_mutate_rate=self.mutate_mutate_rate) # could initialize with any genome here

                # first child
                agent_genome.cross([agent.genome, partners_a[idx].genome])
                new_agent = Agent(genome=agent_genome)
                new_agent.mutate()

                if len(self.agents) < env.current_capacity*env.num_niches:
                    self.agents.append(new_agent)

                # second child
                agent_genome.cross([agent.genome, partners_b[idx].genome])
                new_agent.mutate()

                if len(self.agents) < env.current_capacity:
                    self.agents.append(new_agent)

        elif self.selection_type == "capacity-fitness":
            self.agents = self.order_agents(self.agents)

            for agent in self.agents:
                agent.reproduced = False
            new_agents = []
            keep_agents = copy.copy(self.agents)
            self.agents = []

            # agents reproduce within the niches they belong to
            for lat in range(-int(env.num_niches/2) , int(env.num_niches/2 + 0.5)):
                lat_climate = env.mean + 0.01 * lat
                current_agents = []
                for agent in keep_agents:
                    if lat_climate in agent.niches:
                        if not agent.reproduced: # ensure that an agent reproduces in at most one niche
                            current_agents.append(agent)

                lat_capacity = int(lat_climate * env.niche_capacity)
                if lat_capacity > 1 and len(current_agents) > 2:
                    current_agents = self.order_agents(current_agents)

                    self.agents_reproduce = current_agents[:int(lat_capacity/2)]
                    weights = [agent.fitness for agent in self.agents_reproduce]
                    self.agents_reproduce = choices(self.agents_reproduce, weights=weights,
                                                    k=len(self.agents_reproduce))

                    # find partners
                    weights = [agent.fitness for agent in self.agents_reproduce]
                    partners_a = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))
                    partners_b = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))

                    for idx, agent in enumerate(self.agents_reproduce):
                        agent_genome = Genome(type=self.genome_type, env_mean=self.env_mean, init_SD=self.init_SD,
                                              init_mutate=self.init_mutate, mutate_mutate_rate=self.mutate_mutate_rate)

                        # first child
                        agent_genome.cross([agent.genome, partners_a[idx].genome])
                        new_agent = Agent(genome=agent_genome)
                        new_agent.mutate()
                        new_agents.append(new_agent)
                        agent.reproduced = True

                        # second child
                        agent_genome.cross([agent.genome, partners_b[idx].genome])
                        new_agent = Agent(genome=agent_genome)
                        new_agent.mutate()
                        new_agents.append(new_agent)
                        agent.reproduced = True

            self.agents = new_agents

        elif self.selection_type == "limited-capacityv2":

            for agent in self.agents:
                agent.reproduced = False
            new_agents = []
            keep_agents = copy.copy(self.agents)
            self.agents = []

            # agents reproduce within the niches they belong to
            for lat in range(-int(env.num_niches/2) , int(env.num_niches/2 + 0.5)):
                lat_climate = env.mean + 0.01 * lat
                current_agents = []
                for agent in keep_agents:
                    if lat_climate in agent.niches:
                        if not agent.reproduced: # ensure that an agent reproduces in at most one niche
                            current_agents.append(agent)

                lat_capacity = int(lat_climate * env.niche_capacity)
                if lat_capacity > 1 and len(current_agents) > 2:

                    self.agents_reproduce = current_agents[:int(lat_capacity/2)]
                    weights = [agent.fitness for agent in self.agents_reproduce]
                    self.agents_reproduce = choices(self.agents_reproduce, weights=weights,
                                                    k=len(self.agents_reproduce))

                    # find partners
                    weights = [agent.fitness for agent in self.agents_reproduce]
                    partners_a = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))
                    partners_b = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))

                    for idx, agent in enumerate(self.agents_reproduce):
                        agent_genome = Genome(type=self.genome_type, env_mean=self.env_mean, init_SD=self.init_SD,
                                              init_mutate=self.init_mutate, mutate_mutate_rate=self.mutate_mutate_rate)

                        # first child
                        agent_genome.cross([agent.genome, partners_a[idx].genome])
                        new_agent = Agent(genome=agent_genome)
                        new_agent.mutate()
                        new_agents.append(new_agent)
                        agent.reproduced = True

                        # second child
                        agent_genome.cross([agent.genome, partners_b[idx].genome])
                        new_agent = Agent(genome=agent_genome)
                        new_agent.mutate()
                        new_agents.append(new_agent)
                        agent.reproduced = True

            self.agents = new_agents


    def order_agents(self, agents):
        # order agents based on fitness
        fitness_values = [agent.fitness for agent in agents]
        keydict = dict(zip(agents, fitness_values))
        agents.sort(key=keydict.get, reverse=True)
        return agents

    def survive(self, env):
        """ Discard extinct agents. Also updates their fitness

        Parameters
        ----------
        env: Env
            the current environment
        """
        self.num_extinctions = 0
        new_agents = []
        for agent in self.agents:
            extinct = agent.is_extinct(env)
            if extinct and self.extinctions:
                self.num_extinctions += 1
            else:
                new_agents.append(agent)
        self.agents = new_agents


    def has_mass_extinction(self):
        """ Detects if a mass extinction has happened.
        """
        if len(self.agents) < 2:
            print("Mass extinction. Terminating program.")
            return True
        return False



