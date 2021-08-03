from scipy import stats
import numpy as np

class Agent:

  def __init__(self, mean, SD):
    self.mean = mean
    self.SD = SD
    self.generalist = self.is_generalist()
    self.ancestry = []

  def compute_fitness(self, env_mean):
    self.fitness = stats.norm(self.mean, self.SD).pdf(env_mean)
    return self.fitness

  def is_generalist(self):
    if self.SD > 0.1:
      self.generalist = True
    else:
      self.generalist = False
    return self.generalist

  def is_extinct(self, env_mean):
    if ((self.mean - 2*self.SD) > env_mean) or (self.mean + 2*self.SD) < env_mean:
      return True
    else:
      return False

  def has_speciated(self, agents):
    thres_mean = 0.05
    thres_SD = 0.01

    if len(self.ancestry) > 50:
      current_gene = self.ancestry[-1]
      previous_gene = self.ancestry[-50]

      same = False
      if np.abs(current_gene[0] - previous_gene[0]) < thres_mean:
        if np.abs(current_gene[1] - previous_gene[1]) < thres_SD:
              same = True
      if same:
        return False
      else:
        return True
    else:
      return False
