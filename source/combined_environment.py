""" Implements the climate model described in Maslin."""


from environment import Env
from numpy.random import normal
import numpy as np
from scipy.interpolate import interp1d



class CombEnv(Env):

  def __init__(self, orig_capacity, model, irregular, factor_time_abrupt=1, factor_time_variable=1, 
               factor_time_steady=1, var_freq=5, var_SD=0.2):
    self.type = "combined"
    self.low = 1.0 # mean of environment  during low-resources periods
    self.high = 1.5 # mean of environment  during high-resources periods
    self.mean = self.low
    self.model = model
    self.generation_duration = 15 # how many years a generation is
    self.factor_time_abrupt = factor_time_abrupt # scales abrupt transition in time
    self.factor_time_variable = factor_time_variable # scales abrupt transition in time
    self.factor_time_steady = factor_time_steady # scales abrupt transition in time
    self.var_freq = var_freq # scales abrupt transition in size
    self.irregular = irregular
    self.orig_capacity = orig_capacity
    self.capacity = self.orig_capacity*self.low
    self.cycles = 1


    # breakpoints for determining the different phases
    self.b1 = int(8000/self.generation_duration)
    self.b2 = int(self.b1 + int(300/self.generation_duration)*self.factor_time_abrupt)
    self.b3 = int(self.b2 + int(8000/self.generation_duration)*self.factor_time_steady)
    self.b4 = int(self.b3 + int(2000/self.generation_duration)*self.factor_time_variable)
    self.b5 = int(self.b4 + int(8000/self.generation_duration)*self.factor_time_steady)
    self.SD = var_SD
    self.rate1 = (self.high-self.low)/(self.b2-self.b1)
    self.rate2 =  (self.high-self.low)/(self.b4-self.b3)




    # logging
    self.b1_values = [self.b1]
    self.b2_values = [self.b2]
    self.b3_values = [self.b3]
    self.b4_values = [self.b4]
    self.b5_values = [self.b5]
    self.climate_values = []
    print(self.b1, self.b2, self.b3, self.b4)

  def climate_func(self, gen):

    if gen < self.b1: # low period
      climate = self.low

    elif gen >= self.b1 and gen < self.b2: # abrupt transition
      climate = self.climate_values[-1] + self.rate1

    elif gen >= self.b2 and gen < self.b3: # high period
      climate = self.high
      self.steps_var = 0

    elif gen == self.b3: # step at which highly-variable period begins
      # generations at which there is variability using desired frequency
      x_points = [self.b3]
      while x_points[-1] < self.b4:
        x_points.append(int(x_points[-1] + self.var_freq + normal(0,20)))
      x_points.append(self.b4)

      y_points = []
      for el in x_points:
        offset = normal(0, self.SD)
        y_points.append(self.high -self.rate2*(el-self.b3) + offset)

      y_points = [max([self.low, min([el, self.high])]) for el in y_points]
      self.interp = interp1d(x_points, y_points)
      climate = self.high


    elif gen >= self.b3 and gen < self.b4: # highly-variable period
      climate = float(self.interp(gen))
      climate = np.min([self.high, climate ])
      climate = np.max([self.low, climate])

    elif gen >= self.b4 and gen < self.b5: # low-resources period
      climate = self.low
      capacity = climate*self.orig_capacity

    elif gen == self.b5: # end of current cycle

      climate = self.low
      self.cycles += 1
      self.b1 = int(gen + int(8000 / self.generation_duration))
      self.b2 = int(self.b1 + int(300 / self.generation_duration) * self.factor_time_abrupt)
      self.b3 = int(self.b2 + int(8000 / self.generation_duration))
      self.b4 = int(self.b3 + int(2000 / self.generation_duration)*self.factor_time_variable)
      self.b5 = int(self.b4 + int(8000 / self.generation_duration))
      self.b1_values.append(self.b1)
      self.b2_values.append(self.b2)
      self.b3_values.append(self.b3)
      self.b4_values.append(self.b4)
      self.b5_values.append(self.b5)

    capacity = climate * self.orig_capacity

    # TODO: this is for debugging
    if self.model == "hybrid_nocapac" or self.model == "A":
      self.capacity = self.orig_capacity
    else:
      self.capacity = capacity

    self.climate_values.append(climate)
    return climate


  def step(self, gen):
    self.mean = self.climate_func(gen)