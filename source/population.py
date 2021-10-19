from agent import Agent
from genome import Genome
from random import choices, randrange

class Population:

    def __init__(self, pop_size, survival_type, genome_type, env_mean, init_SD, init_mutate, mutate_rate):
        self.agents = []
        self.survival_type = survival_type
        self.genome_type = genome_type
        self.init_SD = init_SD
        self.init_mutate=init_mutate
        self.mutate_rate = mutate_rate
        self.env_mean = env_mean
        for _ in range(pop_size):
            agent_genome = Genome(type=genome_type, env_mean=env_mean, init_SD=init_SD,
                                  init_mutate=init_mutate, mutate_rate=mutate_rate )

            self.agents.append(Agent(genome=agent_genome))

    def compute_fitness(self, env_mean):
        for agent in self.agents:
            agent.compute_fitness(env_mean)

    def reproduce(self, env):
    # choose which ones will reproduce
        if self.survival_type == "FP":
            self.agents_reproduce = self.agents[:int(len(self.agents)/2)]

            # match pairs
            weights = [agent.fitness for agent in self.agents_reproduce]
            partners = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))
            for idx, agent in enumerate(self.agents_reproduce):
                agent_genome = Genome(type=self.genome_type, env_mean=self.env_mean, init_SD=self.init_SD,
                                  init_mutate=self.init_mutate, mutate_rate=self.mutate_rate)
                agent_genome.cross([agent.genome, partners[idx].genome])
                new_agent = Agent(genome=agent_genome)

                #mutations
                new_agent.mutate()
                new_agent.compute_fitness(env.mean)

                self.agents.append(new_agent)

            while len(self.agents) > env.capacity:
                self.agents.pop(randrange(len(self.agents)))

    def order_agents(self, env_mean):
        # order agents based on fitness
        fitness_values = []
        for agent in self.agents:
            fitness_values.append(agent.compute_fitness(env_mean))

        keydict = dict(zip(self.agents, fitness_values))
        self.agents.sort(key=keydict.get, reverse=True)

    def survive(self, env):
        """ Compute the fitness of a generation and rank individuals.
        """
        for agent in self.agents:
            agent.compute_fitness(env.mean)
            if agent.is_extinct(env.mean):
                self.agents.remove(agent)
        self.order_agents(env.mean)

    def has_mass_extinction(self):
        if len(self.agents) < 2:
            print("Mass extinction. Terminating program.")
            return True
        return False



