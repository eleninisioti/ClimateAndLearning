from agent import Agent
from genome import Genome
from random import choices
import random
import copy


class Population:
    """ A population is a collections of agents with functionality for reproducing them based on a selection
    mechanism
    """

    def __init__(self, pop_size, selection_type, genome_type, env_mean, init_sigma, init_mutate,
                 mutate_mutate_rate, extinctions, mean_fitness=True, reproduce_once=True):
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

        extinctions: int
            If one, extinct individuals are discarded from the population

        mean_fitness: bool
            If True, an agent's fitness is the mean of its fitnesses in which it survives. If False,
            the agent reproduces with a different fitness in each niche

        """
        self.agents = []
        self.selection_type = selection_type
        self.mean_fitness = mean_fitness
        self.reproduce_once = reproduce_once

        # keep these variables for initializing future agents
        self.genome_type = genome_type
        self.extinctions = extinctions
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

    def reproduce_v2(self, env):
        self.agents = self.order_agents(self.agents)

        for agent in self.agents:
            agent.reproduced = False
        new_agents = []
        keep_agents = copy.copy(self.agents)
        self.agents = []

        # agents reproduce within the niches they belong to
        for lat in range(-int(env.num_niches / 2), int(env.num_niches / 2 + 0.5)):
            lat_climate = env.mean + 0.01 * lat
            current_agents = []
            for agent in keep_agents:
                if lat_climate in agent.niches:
                    if not agent.reproduced:  # ensure that an agent reproduces in at most one niche
                        current_agents.append(agent)
                    else:
                        print("reproduced")

            for agent in current_agents:
                if agent.reproduced:
                    print("wrong")
            lat_capacity = int(lat_climate * env.niche_capacity)
            if lat_capacity > 1 and len(current_agents) > 2:
                current_agents = self.order_agents(current_agents)

                self.agents_reproduce = current_agents[:int(lat_capacity / 2)]
                print("v2: climate", lat_climate, len(self.agents_reproduce))

                weights = [agent.fitness for agent in self.agents_reproduce]
                self.agents_reproduce = choices(self.agents_reproduce, weights=weights,
                                                k=len(self.agents_reproduce))

                # find partners
                weights = [agent.fitness for agent in self.agents_reproduce]
                partners_a = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))
                partners_b = choices(self.agents_reproduce, weights=weights, k=len(self.agents_reproduce))
                for agent in self.agents_reproduce:
                    if agent.reproduced:
                        print("wrong")

                for idx, agent in enumerate(self.agents_reproduce):
                    agent_genome = Genome(genome_type=self.genome_type, env_mean=self.env_mean,
                                          init_sigma=self.init_sigma,
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

    def reproduce(self, env):
        """ Population reproduction at the end of a generation.

        Parameters
        ----------
        env: Env
            the current environment
        """
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

        # pair agents for reproduction
        random.shuffle(for_reproduction)
        new_agents = []

        for niche_data in for_reproduction:
            niche_new_agents = []
            niche_pop = niche_data["population"]
            niche_capacity = niche_data["capacity"]
            niche_climate = niche_data["climate"]
            random.shuffle(niche_pop)

            if "F" in self.selection_type:
                niche_pop = self.order_agents(niche_pop, niche_climate)

            if self.reproduce_once:
                niche_pop = [el for el in niche_pop if el.reproduced < 1]
            else:
                #print(niche_capacity, len(niche_pop))
                niche_pop = [el for el in niche_pop]

            if "F" in self.selection_type:
                if self.mean_fitness:
                    weights = [agent.fitness for agent in niche_pop]
                else:
                    weights = [agent.fitnesses[niche_climate] for agent in niche_pop]
            else:
                weights = [1 for agent in niche_pop]

            if len(niche_pop):
                agents_reproduce = choices(niche_pop, weights=weights,
                                                k=len(niche_pop))

                # find partners
                weights = [agent.fitness for agent in agents_reproduce]
                partners_a = choices(agents_reproduce, weights=weights, k=len(agents_reproduce))
                partners_b = choices(agents_reproduce, weights=weights, k=len(agents_reproduce))

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

                    # second child
                    agent_genome.cross([agent.genome, partners_b[idx].genome])
                    new_agent = Agent(genome=agent_genome)
                    new_agent.mutate()
                    if len(niche_new_agents) < niche_capacity:
                        niche_new_agents.append(new_agent)

                    if len(new_agents) > sum([niche_data["capacity"] for niche_data in for_reproduction]):
                        print("new agents", len(new_agents),
                              sum([niche_data["capacity"] for niche_data in for_reproduction]))

            new_agents.extend(niche_new_agents)
            #print("after reproduce", niche_capacity, len(niche_new_agents))
        print("current_capacity", sum([niche_data["capacity"] for niche_data in for_reproduction]))
        if len(new_agents) > sum([niche_data["capacity"] for niche_data in for_reproduction]):
            print("new agents", len(new_agents), sum([niche_data["capacity"] for niche_data in for_reproduction]))
            quit()
        self.agents = new_agents

    def order_agents(self, agents, niche_climate=0):
        # order agents based on fitness
        if self.mean_fitness:
            fitness_values = [agent.fitness for agent in agents]
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
