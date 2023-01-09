from agent import Agent
from genome import Genome
import random
import numpy as np
import copy
import math

class Population:
    """ A population is a collection of agents that reproduces based on the desired selection mechanism
    """

    def __init__(self, pop_size, selection_type, genome_type, env_mean, init_sigma, init_mutate,
                 mutate_mutate_rate, history_window):
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
        self.max_population = 5000

        # keep these variables for initializing future agents
        self.genome_type = genome_type
        self.init_sigma = init_sigma
        self.init_mutate = init_mutate
        self.mutate_mutate_rate = mutate_mutate_rate
        self.env_mean = env_mean
        self.competition = 0
        self.not_reproduced = 0
        for _ in range(pop_size):
            agent_genome = Genome(genome_type=genome_type,
                                  env_mean=env_mean,
                                  init_sigma=init_sigma,
                                  init_mutate=init_mutate,
                                  mutate_mutate_rate=mutate_mutate_rate)

            self.agents.append(Agent(genome=agent_genome, history_window=history_window))

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


    def reproduce_xland(self, env):
        for agent in self.agents:
            # get fitness in each niche
            fitnesses = list(agent.fitnesses.values())
            # make a cdf
            # compute first 51 percentiles
            agent.percentiles = []
            for percentage in range(51):
                n = len(fitnesses)
                p = n * percentage / 100
                if p.is_integer():
                    percentile = sorted(fitnesses)[int(p)]
                else:
                    percentile = sorted(fitnesses)[int(math.ceil(p)) - 1]
                agent.percentiles.append(percentile)


        for agent_a in copy.copy(self.agents):
            for agent_b in copy.copy(self.agents):
                agent_a_dominant = all(map(lambda p, q: p > q, agent_a.percentiles, agent_b.percentiles))
                agent_b_dominant = all(map(lambda p, q: p > q, agent_b.percentiles, agent_a.percentiles))

                if agent_a_dominant:
                    # remove agent b
                    if agent_b in self.agents:
                        self.agents.remove(agent_b)
                        # mutate agent_b
                        agent_a.mutate()
                        self.agents.append(agent_a)

                elif agent_b_dominant:
                    # remove agent b
                    if agent_a in self.agents:
                        self.agents.remove(agent_a)
                        # mutate agent_b
                        agent_b.mutate()
                        self.agents.append(agent_b)


    def reproduce(self, env):
        """ Population reproduction at the end of a generation.

        Parameters
        ----------
        env: Env
            the current environment
        """
        if self.selection_type == "xland":
            self.reproduce_xland(env)
            return
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

                for_reproduction.append({"population": niche_pop, "lat": niche_info["lat"], "capacity": niche_capacity,
                                         "climate": \
                    niche_climate})
        else:
            niche_capacity = int(env.current_capacity * env.num_niches)
            for_reproduction = [{"population": self.agents,  "capacity": niche_capacity,
            "climate": env.mean}]
        # -----------------------------------------------
        random.shuffle(for_reproduction)
        new_agents = []
        self.competition = 0

        niche_constructions = {}


        for niche_idx, niche_data in enumerate(for_reproduction):
            niche_new_agents = []
            niche_pop = niche_data["population"]
            niche_capacity = niche_data["capacity"]
            niche_climate = niche_data["climate"]
            random.shuffle(niche_pop)

            if "N" in self.selection_type:
                self.mean_fitness = 0
            else:
                self.mean_fitness = 1 # agent reproduces based on its mean fitness across niches

            if "F" in self.selection_type:
                niche_pop = self.order_agents(niche_pop, niche_climate)

            # only as many agents as fit in the niche will reproduce
            if niche_capacity < len(niche_pop)*2:
                self.competition += (len(niche_pop)*2 - niche_capacity)

            niche_pop = [el for el in niche_pop[:int(niche_capacity / 2)]]

            # agents chosen for participating in a pair based on their fitness
            if "F" in self.selection_type:
                if self.mean_fitness:
                    weights = [np.mean(list(agent.fitnesses.values())) for agent in niche_pop]
                else:
                    weights = [agent.fitnesses[niche_climate] for agent in niche_pop]
            else:
                weights = [1 for _ in niche_pop]

            if len(niche_pop):
                agents_reproduce = random.choices(niche_pop, weights=weights, k=len(niche_pop))
                partners_a = random.choices(agents_reproduce, weights=weights, k=len(agents_reproduce))
                partners_b = random.choices(agents_reproduce, weights=weights, k=len(agents_reproduce))
                for idx, agent in enumerate(agents_reproduce):
                    agent_genome = Genome(genome_type=self.genome_type, env_mean=self.env_mean,
                                          init_sigma=self.init_sigma,
                                          init_mutate=self.init_mutate, mutate_mutate_rate=self.mutate_mutate_rate)

                    # first child
                    agent_genome.cross([agent.genome, partners_a[idx].genome])
                    new_agent = Agent(genome=agent_genome, history_window=agent.history_window)
                    new_agent.mutate()

                    if "lat" in niche_data:
                        new_agent.movement = niche_data["lat"]
                        #new_agent.movement = agent.realized_niche

                    else:
                        new_agent.movement = agent.realized_niche

                    new_agent.set_history(agent.history + partners_a[idx].history + [niche_climate])

                    if len(niche_new_agents) < niche_capacity:
                        niche_new_agents.append(new_agent)
                        agent.reproduced = True

                    # second child
                    agent_genome.cross([agent.genome, partners_b[idx].genome])
                    new_agent = Agent(genome=agent_genome, history_window=agent.history_window)
                    new_agent.mutate()
                    if "lat" in niche_data:
                        new_agent.movement = niche_data["lat"]
                        #new_agent.movement = agent.realized_niche

                    else:
                        new_agent.movement = agent.realized_niche

                    new_agent.set_history(agent.history + partners_b[idx].history +
                                        [niche_climate])

                    if len(niche_new_agents) < niche_capacity:
                        niche_new_agents.append(new_agent)
                    # apply niche construction
                    # remove old data about niche
                    if "lat" not in niche_data:
                        niche_index = agent.realized_niche
                    else:
                        niche_index = niche_data["lat"]
                        #niche_index= agent.realized_niche

                    if agent.genome.type == "niche-construction":
                        #print("updating niche with index", niche_index)
                        if niche_index is not None:

                            niche_constructions[niche_index] = agent.genome.genes["c"] + partners_a[idx].genome.genes["c"] +\
                            partners_b[idx].genome.genes["c"]

                    else:
                        niche_constructions[niche_index] = env.niches[niche_index]["constructed"]


            else:
                if "lat" not in niche_data:
                    niche_index = agent.realized_niche
                else:
                    niche_index = niche_data["lat"]
                niche_constructions[niche_index] = env.niches[niche_index]["constructed"]

            new_agents.extend(niche_new_agents)

        self.not_reproduced = len([el for el in self.agents if not agent.reproduced])

        keep_pop = min([len(new_agents), self.max_population])

        self.agents = random.choices(new_agents, k=keep_pop)
        #self.agents = new_agents
        return niche_constructions

    def order_agents(self, agents, niche_climate=0):
        """ Sort agents in ascenting order based on their fitness.
        """
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
