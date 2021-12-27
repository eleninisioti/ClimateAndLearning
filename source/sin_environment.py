""" Implements an environment that changes based on a sine-wave. Default parameters taken from experiments in Section 4.2
of 'Evolution and dispersal under climatic instability: a simple evolutionary algorithm'
"""

from environment import Env
import numpy as np
import math

class SinEnv(Env):

  def __init__(self, period, orig_capacity, num_niches):
    self.omega = 2*math.pi/period

    # determine breakpoints and amplitudes based on the paper
    self.b1 = 500
    self.b2 = 1000
    self.v1 = 0.2 # low amplitude
    self.v2 = self.v1 + 1  # high amplitude
    self.mean = self.v1*np.sin(0*self.omega) +1
    self.climate_values = [self.mean]
    self.orig_capacity = int(orig_capacity / (num_niches))
    self.capacity = orig_capacity
    self.type = "sin"
    self.num_niches = num_niches



  def climate_func(self, gen):

    if gen > self.b1 and gen < self.b2:
      A = self.v2
    else:
      A = self.v1

    # A is the amplitude and omega is the frequency in rad
    climate = A*np.sin(gen*self.omega) + 1
    self.capacity = climate*self.orig_capacity
    self.climate_values.append(climate)
    return climate


  def step(self, gen):
    self.mean = self.climate_func(gen)