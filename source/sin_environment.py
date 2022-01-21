""" Implements an environment that changes based on a sine-wave. Default parameters taken from experiments in Section 4.2
of 'Evolution and dispersal under climatic instability: a simple evolutionary algorithm'
"""

from environment import Env
import numpy as np
import math

class SinEnv(Env):

  def __init__(self, mean, ref_capacity,num_niches, period, amplitude):

    # determine breakpoints and amplitudes based on the paper
    self.type = "sin"
    self.num_niches = num_niches
    self.omega = 2*math.pi/period
    self.amplitude = amplitude
    self.ref_capacity = ref_capacity  # capacity of a niche if there was only one
    self.niche_capacity = int(ref_capacity / num_niches)  # capacity of a niche normalized for number of niches
    self.mean = mean
    self.current_capacity = self.niche_capacity * self.mean
    self.low = amplitude + mean # low amplitude
    self.climate = self.low
    self.climate_values = []

  def step(self, gen):

    # A is the amplitude and omega is the frequency in rad
    self.mean = self.amplitude*np.sin(gen*self.omega) + self.low
    self.current_capacity = self.mean*self.niche_capacity
    self.climate_values.append(self.mean)
    return self.mean


