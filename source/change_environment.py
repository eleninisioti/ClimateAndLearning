"""
Implements an environment where the climate curve is a pulse, as in Figure 2 of 'Evolution and dispersal under climatic
 instability: a simple evolutionary algorithm'
"""

from environment import Env


class ChangeEnv(Env):

  def __init__(self, mean, orig_capacity):
    self.mean = mean

    # define breakpoints
    self.b1 = 200
    self.b2 = 250
    self.b3 = 850
    self.b4 = 900
    self.v1 = self.mean
    self.rate = 0.01
    self.climate_values = []
    self.orig_capacity = orig_capacity
    self.capacity = self.orig_capacity
    self.type = "change"


  def climate_func(self, gen):
    if not gen:
      climate = self.mean
    elif gen > self.b1 and gen < self.b2:
      climate = self.climate_values[-1] + self.rate
    elif gen > self.b3 and gen < self.b4:
      climate = self.climate_values[-1] - self.rate
    else:
      climate = self.climate_values[-1]
    self.capacity = climate*self.orig_capacity
    self.climate_values.append(climate)
    return climate


  def step(self, gen):
    self.mean = self.climate_func(gen)