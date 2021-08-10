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

class Life:

  def __init__(self, args):
    self.config = args
    self.log = {"running_fitness": [],
                "running_mean": [],
                "running_SD": [],
                "total_diversity": [],
                "specialists": {"speciations": [], "extinctions": [], "number": [], "diversity": [],
                                "diversity_mean": [], "diversity_std": []},
                "generalists": {"speciations": [], "extinctions": [], "number": [], "diversity": [],
                                "diversity_mean": [], "diversity_std": []}}
    self.species = []

  def order_agents(self):
    # order agents based on fitness
    fitness_values = []
    for agent in self.agents:
      fitness_values.append(agent.compute_fitness(self.env.mean))

    keydict = dict(zip(self.agents, fitness_values))
    self.agents.sort(key=keydict.get, reverse=True)

  def new_gen(self, mutate):
    """ Computes new generation of agents
    """

    # order based on fitness
    if self.config.model in ["A", "hybrid", "hybrid_nocapac", "hybrid_noextinct"]:
      self.order_agents()

      if self.process_agents(gen=0):
        return False

      # reproduce half of the fittest
      half_agents = round(len(self.agents)/2)-1
      fittest = self.agents[:half_agents]
      num_clones = np.min([half_agents, np.abs(int(normal(half_agents/2, np.sqrt(half_agents/4))))])
      max_fitness = sum([agent.fitness for agent in fittest])
      weights = [agent.fitness/max_fitness for agent in fittest]

      # sample new agents that are copies of existing ones
      sum_nonzero = 0
      for el in weights:
        if el>0:
          sum_nonzero += 1
      if sum_nonzero < num_clones:
        replace = True
      else:
        replace = False
      new_indices = (choice(list(range(half_agents)), num_clones, p=weights, replace=replace)).tolist()
      new_agents = [agent for idx, agent in enumerate(self.agents) if idx in new_indices]

      for idx, agent in enumerate(new_agents):
        agent.ancestry.append([fittest[new_indices[idx]].mean, fittest[new_indices[idx]].SD])

      # sample new agents that get their mean and SDs from different agents
      indices_means = choice(list(range(half_agents)), half_agents - num_clones, p=weights, replace=replace)
      new_means = [fittest[idx].mean for idx in indices_means]
      indices_SDs = choice(list(range(half_agents)), half_agents - num_clones, p=weights, replace=replace)
      new_SDs = [fittest[idx].SD for idx in indices_SDs]

      for idx in range(half_agents-num_clones):
        new_agent = Agent(new_means[idx], new_SDs[idx])
        new_agent.ancestry.append([indices_means[idx], indices_SDs[idx]])
        new_agents.append(new_agent)

      # apply mutation
      replace_idx = 0
      for idx, agent in enumerate(new_agents):
        new_mean = agent.mean + normal(0, mutate)
        new_SD = np.abs(agent.SD + normal(0, mutate))
        new_agent = Agent(new_mean, new_SD)
        new_agent.ancestry = agent.ancestry
        if self.config.model == "A":
          self.agents[half_agents + idx] = new_agent
        else:
          if len(self.agents) < self.env.capacity:
            self.agents.append(new_agent)
          else:
            self.agents[half_agents + replace_idx] = new_agent
            #self.agents[0] = new_agent
            replace_idx += 1

      # reorder agents
      self.order_agents()
      return True
    elif self.config.model == "MC":

      if self.process_agents(gen=0):
        return False

      # individuals that don't satisfy the MC get extinct
      for agent in self.agents:
        agent.compute_fitness(self.env.mean)
        if agent.is_extinct(self.env.mean):
          self.agents.remove(agent)

      # --- the rest reproduce ---
      # pair survivors
      shuffled_agents = random.sample(self.agents, k=len(self.agents))
      childs = []
      for idx, agent in enumerate(self.agents):
        pair = shuffled_agents[idx]
        child_mean = random.choice([agent.mean, pair.mean])  + normal(0, mutate)
        child_sd = np.abs(random.choice([agent.SD, pair.SD])  + normal(0, mutate))
        child= Agent(child_mean, child_sd)
        childs.append(child)
      for child in childs:
        if len(self.agents) < self.env.capacity:
          self.agents.append(child)
        else:
          self.agents[0] = child
      return True



  def evaluate_gen(self):
    """ Compute metrics characterizing the generation.
    """
    # compute generation averages
    fitness_values = []
    mean_values = []
    SD_values = []
    for agent in self.agents:
      fitness_values.append(agent.compute_fitness(self.env.mean))
      mean_values.append(agent.mean)
      SD_values.append(agent.SD)
    mean_fitness = np.mean(fitness_values)
    mean_mean = np.mean(mean_values)
    mean_SD= np.mean(SD_values)
    self.log["running_fitness"].append(mean_fitness)
    self.log["running_mean"].append(mean_mean)
    self.log["running_SD"].append(mean_SD)

  # def determine_species(self, gen):
  #   # plotter = Plotter(self.config.project)
  #   # plotter.plot_generation(self.agents, gen)
  #   species = {}
  #   thres_mean = 0.05
  #   mean_centroids = list(np.arange(-5,5,0.2))
  #   new_species = []
  #   for mean in mean_centroids:
  #     species[mean] = []
  #     for ag in self.agents:
  #       if np.abs(ag.mean - mean) < thres_mean:
  #         species[mean].append(ag)
  #
  #     if len(species[mean]) >0:
  #
  #       SDs_range = [np.min([agent.SD for agent in species[mean]]), np.max([agent.SD for agent in species[mean]])]
  #       new_species.append(Species(mean=mean, SDs_range= SDs_range, agents=species[mean]))
  #
  #   return new_species


  def process_agents(self, gen):


    # detect extinction events
    num_extinctions_gen = 0
    num_extinctions_spec = 0
    num_generalists = 0
    num_specialists = 0
    gen_means = []
    spec_means = []
    gen_stds = []
    spec_stds = []
    total_means = []
    total_stds = []
    for idx,agent in enumerate(self.agents):
      if agent.is_generalist():
        num_extinctions_gen += agent.is_extinct(self.env.mean)
        num_generalists += 1
        gen_means.append(agent.mean)
        gen_stds.append(agent.SD)
      else:
        num_extinctions_spec += agent.is_extinct(self.env.mean)
        spec_means.append(agent.mean)
        spec_stds.append(agent.SD)
        num_specialists += 1
      total_means.append(agent.mean)
      total_stds.append(agent.SD)

    if len(gen_means) >1:
      diversity_gen = np.std(gen_means) + np.std(gen_stds)
      diversity_gen_mean = np.std(gen_means)
      diversity_gen_std = np.std(gen_stds)
    else:
      diversity_gen = 0
      diversity_gen_mean = 0
      diversity_gen_std = 0

    if len(spec_means) >1:
      diversity_spec = np.std(spec_means) + np.std(spec_stds)
      diversity_spec_mean = np.std(spec_means)
      diversity_spec_std = np.std(spec_stds)
    else:
      diversity_spec = 0
      diversity_spec_mean = 0
      diversity_spec_std = 0



    total_diversity = np.std(total_means) + np.std(total_stds)


    # remove from population agents belonging to extinct species
    if self.config.model == "hybrid" or self.config.model == "hybrid_nocapac":
      for agent in self.agents:
        if agent.is_extinct(self.env.mean):
          self.agents.remove(agent)

    # remove oldest individuals if buffer is over-flown
    while len(self.agents) > self.env.capacity:
      self.agents.pop(0)


    if len(self.agents) < 3:
      print("Mass extinction.Terminating.")
      return True

    self.log["generalists"]["extinctions"].append(num_extinctions_gen)
    self.log["specialists"]["extinctions"].append(num_extinctions_spec )
    self.log["generalists"]["number"].append(num_generalists)
    self.log["specialists"]["number"].append(num_specialists)
    self.log["specialists"]["diversity"].append(diversity_spec)
    self.log["generalists"]["diversity"].append(diversity_gen)
    self.log["specialists"]["diversity_mean"].append(diversity_spec_mean)
    self.log["generalists"]["diversity_mean"].append(diversity_gen_mean)
    self.log["specialists"]["diversity_std"].append(diversity_spec_std)
    self.log["generalists"]["diversity_std"].append(diversity_gen_std)
    self.log["total_diversity"].append(total_diversity)


  def run(self):
    # Initialize population
    if self.config.env_type == "change":
      self.env = ChangeEnv(self.config.climate_mean_init, self.config.capacity)
      self.log["env_profile"] = {"start_a": self.env.b1, "end_a": self.env.b2,
                                 "start_b": self.env.b3, "end_b": self.env.b4}
    elif self.config.env_type == "sin":
      self.env = SinEnv(self.config.climate_period, self.config.capacity)
      self.log["env_profile"] = {}

    elif self.config.env_type == "combined":
      self.env = CombEnv(self.config.capacity, self.config.scale_time, self.config.model)
      self.log["env_profile"] = {"start_a": self.env.b1_values, "end_a": self.env.b2_values,
                                 "start_b": self.env.b3_values, "end_b": self.env.b4_values}

    self.agents = []



    num_agents_init = int(np.min([self.config.num_agents, self.env.capacity]))
    for _ in range(num_agents_init):
      agent_mean = normal(self.env.mean, self.config.init_SD)
      agent_SD = np.abs(normal(0, self.config.init_SD))
      self.agents.append(Agent(agent_mean, agent_SD))

    self.evaluate_gen()
    self.process_agents(gen=0)


    print(random.randint(1,10))
    print(np.random.normal())


    # run generations
    for gen in range(self.config.num_gens):
      # update environment
      self.env.step(gen)

      # find new generation
      if not self.new_gen(mutate=self.config.mutate):
        return self.log

      # compute metrics for new generation
      self.evaluate_gen()

      if gen%100==0:
        print("Generation: ", gen)

      self.log["climate_values"] = self.env.climate_values

      # Initialize population
      if self.config.env_type == "change":
        self.log["env_profile"] = {"start_a": self.env.b1, "end_a": self.env.b2,
                                   "start_b": self.env.b3, "end_b": self.env.b4,
                                   "cycles": self.cycles}
      elif self.config.env_type == "sin":
        self.log["env_profile"] = {}

      elif self.config.env_type == "combined":
        self.log["env_profile"] = {"start_a": self.env.b1_values, "end_a": self.env.b2_values,
                                   "start_b": self.env.b3_values, "end_b": self.env.b4_values,
                                   "ncycles": self.cycles, "cycle": self.b5}

    return self.log


