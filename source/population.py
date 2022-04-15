from agent import Agent
from genome import Genome
import random
import numpy as np

class Population:
    """ A population is a collection of agents that reproduces based on the desired selection mechanism
    """

    def __init__(self, pop_size, selection_type, genome_type, env_mean, init_sigma, init_mutate,
                 mutate_mutate_rate):
        """ Class constructor.

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

        init_sigma: float
            determines the range around env_mean for genome initialization

        init_mutate: float
            the mutation rate upon initialisation

        mutate_mutate_rate: float
            the mutation rate of the mutation rate
        """
        self.agents = []
        self.selection_type = selection_type

        # keep these variables for initializing future agents
        self.genome_type = genome_type
        self.init_sigma = init_sigma
        self.init_mutate = init_mutate
        self.mutate_mutate_rate = mutate_mutate_rate
        self.env_mean = env_mean
        for _ in range(pop_size):
            agent_genome = Genome(genome_type=genome_type,
                                  env_mean=env_mean,
                                  init_sigma=init_sigma,
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
        # ------ which agents belong to each niche? -----
        if "N" in self.selection_type:
            # competition is niche-limited
            for_reproduction = []
            for niche_idx, niche_info in env.niches.items():
                niche_climate = niche_info["climate"]

                niche_capacity = niche_info["capacity"]
                niche_pop = []

                for agent in self.agents:
                    if niche_climate in agent.niches:
                        niche_pop.append(agent)

                for_reproduction.append({"population": niche_pop, "capacity": niche_capacity, "climate": niche_climate})
        else:
            niche_capacity = int(env.current_capacity * env.num_niches)
            for_reproduction = [{"population": self.agents, "capacity": niche_capacity, "climate": env.mean}]
        # -----------------------------------------------
        random.shuffle(for_reproduction)
        new_agents = []
        capacity_now = 0
        added_agents = 0
        for niche_idx, niche_data in enumerate(for_reproduction):
            niche_new_agents = []
            niche_pop = niche_data["population"]
            niche_capacity = niche_data["capacity"]
            niche_climate = niche_data["climate"]
            random.shuffle(niche_pop)

            if "N" in self.selection_type:
                self.mean_fitness = 0
            else:
                self.mean_fitness = 1

            if "F" in self.selection_type:
                niche_pop = self.order_agents(niche_pop, niche_climate)

            niche_pop = [el for el in niche_pop[:int(niche_capacity / 2)]]

            if "F" in self.selection_type:
                if self.mean_fitness:
                    weights = [np.mean(list(agent.fitnesses.values())) for agent in niche_pop]
                else:
                    weights = [agent.fitnesses[niche_climate] for agent in niche_pop]
            else:
                weights = [1 for _ in niche_pop]
            capacity_now += niche_data["capacity"]

            if len(niche_pop):
                agents_reproduce = random.choices(niche_pop, weights=weights,
                                           k=len(niche_pop))

                partners_a = random.choices(agents_reproduce, weights=weights, k=len(agents_reproduce))
                partners_b = random.choices(agents_reproduce, weights=weights, k=len(agents_reproduce))

                for idx, agent in enumerate(agents_reproduce):
                    agent_genome = Genome(genome_type=self.genome_type, env_mean=self.env_mean,
                                          init_sigma=self.init_sigma,
                                          init_mutate=self.init_mutate, mutate_mutate_rate=self.mutate_mutate_rate)

                    # first child
                    agent_genome.cross([agent.genome, partners_a[idx].genome])
                    new_agent = Agent(genome=agent_genome)
                    new_agent.mutate()
                    if len(niche_new_agents) < niche_capacity:
                        niche_new_agents.append(new_agent)
                        agent.reproduced = True
                        added_agents += 1

                    # second child
                    agent_genome.cross([agent.genome, partners_b[idx].genome])
                    new_agent = Agent(genome=agent_genome)
                    new_agent.mutate()
                    if len(niche_new_agents) < niche_capacity:
                        niche_new_agents.append(new_agent)
                        added_agents += 1

            new_agents.extend(niche_new_agents)

        self.agents = new_agents

    def order_agents(self, agents, niche_climate=0):
        # order agents based on fitness
        if self.mean_fitness:
            fitness_values = [np.mean(list(agent.fitnesses.values())) for agent in agents]
        else:
            fitness_values = [agent.fitnesses[niche_climate] for agent in agents]
        keydict = dict(zip(agents, fitness_values))
        agents.sort(key=keydict.get, reverse=True)
        return agents

    def survive(self, env):
        """ Discard extinct agents

        Parameters
        ----------
        env: Env
            the current environment
        """
        self.num_extinctions = 0
        new_agents = []
        for agent in self.agents:
            extinct = agent.is_extinct(env)
            if extinct:
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
