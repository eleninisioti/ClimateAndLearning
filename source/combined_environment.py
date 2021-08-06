""" Implements the climate model described in Maslin."""


from environment import Env
from numpy.random import normal
import numpy as np

class CombEnv(Env):

  def __init__(self, orig_capacity, scale_factor, model):
    self.low = 1.0
    self.high = 1.5
    # self.low = 1
    # self.high = 1
    self.mean = self.low
    self.model = model
    self.generation_duration = 15 # how many years a generation is

    # define breakpoints
    self.factor = scale_factor
    self.b1 = int(8000/self.generation_duration)
    self.b2 = self.b1 + int(300/self.generation_duration)*self.factor
    self.b3 = self.b2 + int(8000/self.generation_duration)*2
    self.b4 = self.b3 + int(2000/self.generation_duration)
    self.b5 = self.b4 + int(8000/self.generation_duration)
    self.SD = 0.2
    self.rate1 = (self.high-self.low)/(self.b2-self.b1)
    self.rate2 =  (self.high-self.low)/(self.b4-self.b3)
    self.climate_values = [self.low]
    self.climate_values_clean = [self.high]
    self.orig_capacity = orig_capacity
    self.capacity = self.orig_capacity*self.low
    self.cycles = 1
    self.b1_values = [self.b1]
    self.b2_values = [self.b2]
    self.b3_values = [self.b3]
    self.b4_values = [self.b4]




  def climate_func(self, gen):

    if gen < self.b1:
      climate = self.low
      capacity = climate*self.orig_capacity

    elif gen >= self.b1 and gen < self.b2:
      climate = self.climate_values[-1] + self.rate1
      capacity = climate*self.orig_capacity

    elif gen >= self.b2 and gen < self.b3:
      climate = self.high
      capacity = climate*self.orig_capacity
      self.steps_var = 0

    elif gen >= self.b3 and gen < self.b4:
      climate = self.climate_values_clean[-1] - self.rate2
      capacity = climate*self.orig_capacity
      # add high variance
      self.climate_values_clean.append(climate)
      if self.steps_var % 5:
        climate = climate + normal(0, self.SD)
      self.steps_var += 1
      climate = np.min([self.high, climate ])
      climate = np.max([self.low, climate])
    elif gen >= self.b4 and gen < self.b5:
      climate = self.low
      capacity = climate*self.orig_capacity

    elif gen == self.b5:
      self.climate_values_clean.append(self.high)

      climate = self.low
      capacity = climate*self.orig_capacity
      self.cycles += 1
      self.b1 = gen + int(8000 / self.generation_duration)
      self.b2 = self.b1 + int(300 / self.generation_duration) * self.factor
      self.b3 = self.b2 + int(8000 / self.generation_duration) * 2
      self.b4 = self.b3 + int(2000 / self.generation_duration)
      self.b5 = self.b4 + int(8000 / self.generation_duration)
      print(self.b1, self.b2, self.b3,self.b4, self.b5)
      self.b1_values.append(self.b1)
      self.b2_values.append(self.b2)
      self.b3_values.append(self.b3)
      self.b4_values.append(self.b4)



    # TODO: this is for debugging
    if self.model == "hybrid_nocapac" or self.model == "A":
      self.capacity = self.orig_capacity
    else:
      self.capacity = capacity

    self.climate_values.append(climate)
    return climate


  def step(self, gen):
    self.mean = self.climate_func(gen)