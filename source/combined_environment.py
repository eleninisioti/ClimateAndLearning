""" Implements the climate model described in Maslin."""


from environment import Env
from numpy.random import normal
import numpy as np
from scipy.interpolate import interp1d



class CombEnv(Env):

  def __init__(self, orig_capacity, model, factor_time_abrupt=1, factor_time_variable=1, var_freq=5, var_SD=0.2):
    self.low = 1.0
    self.high = 1.5
    # self.low = 1
    # self.high = 1
    self.mean = self.low
    self.model = model
    self.generation_duration = 15 # how many years a generation is

    # define breakpoints
    self.factor_time_abrupt = factor_time_abrupt # scales abrupt transition in time
    self.factor_time_variable = factor_time_variable # scales abrupt transition in time
    self.var_freq = var_freq # scales abrupt transition in size

    self.b1 = int(8000/self.generation_duration)
    self.b2 = self.b1 + int(300/self.generation_duration)*self.factor_time_abrupt
    self.b3 = self.b2 + int(8000/self.generation_duration)*2
    self.b4 = self.b3 + int(2000/self.generation_duration)*self.factor_time_variable
    self.b5 = self.b4 + int(8000/self.generation_duration)
    self.SD = var_SD
    self.rate1 = (self.high-self.low)/(self.b2-self.b1)
    self.rate2 =  (self.high-self.low)/(self.b4-self.b3)
    #self.rate2 = 0.1
    self.climate_values = [self.low]
    self.climate_values_clean = [self.high]
    self.orig_capacity = orig_capacity
    self.capacity = self.orig_capacity*self.low
    self.cycles = 1
    self.b1_values = [self.b1]
    self.b2_values = [self.b2]
    self.b3_values = [self.b3]
    self.b4_values = [self.b4]
    self.b5_values = [self.b5]
    print(self.b1, self.b2, self.b3, self.b4, self.b5)

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

    elif gen == self.b3:
      x_points = [self.b3]
      while x_points[-1] < self.b4:
        x_points.append(int(x_points[-1] + self.var_freq + normal(0,20)))
      x_points.append(self.b4)

      y_points = []
      previous_offset = 1
      pos_offset = 1
      neg_offset = -1
      for el in x_points:
        offset = normal(0, self.SD)
        if previous_offset>0:
          small_enough = (np.abs(offset) < np.abs(neg_offset))
        else:
          small_enough = (np.abs(offset) < np.abs(pos_offset))
        while (np.abs(offset) < 0.05) or (offset*previous_offset >0) or (not small_enough):
          offset = normal(0, self.SD)
          if previous_offset > 0:
            small_enough = (np.abs(offset) < np.abs(neg_offset))
          else:
            small_enough = (np.abs(offset) < np.abs(pos_offset))

        if previous_offset > 0:
          neg_offset = offset
        else:
          pos_offset = offset

        previous_offset = offset


        y_points.append(self.high -self.rate2*(el-self.b3)  + offset)

      y_points = [max([self.low, min([el, self.high])]) for el in y_points]
      self.interp = interp1d(x_points, y_points)
      climate= self.high
      capacity = climate*self.orig_capacity


    elif gen >= self.b3 and gen < self.b4:
      climate = float(self.interp(gen))
      climate = np.min([self.high, climate ])
      climate = np.max([self.low, climate])
      #climate = self.climate_values[-1] - self.rate2
      capacity = climate*self.orig_capacity
      # add high variance
      #self.steps_var += 1
      # if self.steps_var == self.var_freq:
      #   climate = climate + np.abs(normal(0, self.SD))
      #   self.steps_var = 0

    elif gen >= self.b4 and gen < self.b5:
      climate = self.low
      capacity = climate*self.orig_capacity

    elif gen == self.b5:
      #self.climate_values.append(self.high)

      climate = self.low
      capacity = climate*self.orig_capacity
      self.cycles += 1
      self.b1 = gen + int(8000 / self.generation_duration)
      self.b2 = self.b1 + int(300 / self.generation_duration) * self.factor_time_abrupt
      self.b3 = self.b2 + int(8000 / self.generation_duration) * 2
      self.b4 = self.b3 + int(2000 / self.generation_duration)*self.factor_time_variable
      self.b5 = self.b4 + int(8000 / self.generation_duration)
      print(self.b1, self.b2, self.b3,self.b4, self.b5)
      self.b1_values.append(self.b1)
      self.b2_values.append(self.b2)
      self.b3_values.append(self.b3)
      self.b4_values.append(self.b4)
      self.b5_values.append(self.b5)



    # TODO: this is for debugging
    if self.model == "hybrid_nocapac" or self.model == "A":
      self.capacity = self.orig_capacity
    else:
      self.capacity = capacity

    self.climate_values.append(climate)
    return climate


  def step(self, gen):
    self.mean = self.climate_func(gen)