"""
Implements an environment where the climate curve is a pulse, as in Figure 2 of 'Evolution and dispersal under climatic
 instability: a simple evolutionary algorithm'
"""

from environment import Env


class ChangeEnv(Env):

  def __init__(self, mean, orig_capacity, num_niches, factor_time_abrupt, factor_time_steady):
    self.mean = mean
    self.low = mean
    self.high = mean + 0.5
    self.num_niches = num_niches

    # define breakpoints
    self.b1 = 200
    self.b2 = self.b1 + 50*factor_time_abrupt
    self.b3 = self.b2 + 50*factor_time_steady
    self.b4 = self.b3 + 50*factor_time_abrupt
    self.v1 = self.mean
    self.climate_values = []
    self.orig_capacity = int(orig_capacity/(num_niches))
    self.keep_capacity = orig_capacity
    self.capacity = self.keep_capacity
    self.type = "change"
    print(self.b1, self.b2, self.b3, self.b4)
    self.rate = (self.high-self.low)/(self.b2-self.b1)


  def climate_func(self, gen):
    if not gen:
      climate = self.mean
    elif gen > self.b1 and gen < self.b2:
      climate = self.climate_values[-1] + self.rate
    elif gen > self.b3 and gen < self.b4:
      climate = self.climate_values[-1] - self.rate
    else:
      climate = self.climate_values[-1]
    self.capacity = climate*self.keep_capacity
    self.climate_values.append(climate)
    return climate


  def step(self, gen):
    self.mean = self.climate_func(gen)