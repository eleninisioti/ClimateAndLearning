""" Deprecated file.
"""

import numpy as np

class Species:

  def __init__(self, mean, SDs_range, agents):
    self.mean = mean
    self.SDs_range = SDs_range
    self.agents = agents
    self.generalist_thres = 0.1
    self.generalist = False

  def is_generalist(self):
    if self.SDs_range[0] > self.generalist_thres:
      self.generalist = True
    else:
      self.generalist = False

  def is_extinct(self):
    avg_fitness = np.mean([agent.fitness for agent in self.agents])
    if avg_fitness < (0.1 /len(self.agents)):
      return True
    else:
      return False

  def has_speciated(self, species):
    thres_mean = 0.1
    thres_SD = 0.01

    same = False
    for spec in species:
      if np.abs(spec.mean - self.mean) < thres_mean:
        if np.abs(spec.SDs_range[0] - self.SDs_range[0]) < thres_SD:
          if np.abs(spec.SDs_range[1] - self.SDs_range[1]) < thres_SD:
            same = True
    if same:
      return False
    else:
      return True

