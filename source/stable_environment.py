""" Implements an environment that changes based on a sine-wave. Default parameters taken from experiments in Section 4.2
of 'Evolution and dispersal under climatic instability: a simple evolutionary algorithm'
"""

from environment import Env
import numpy as np
import math

class StableEnv(Env):

  def __init__(self, mean, ref_capacity,num_niches):

    # determine breakpoints and amplitudes based on the paper
    self.type = "stable"
    self.num_niches = num_niches
    self.mean = mean
    self.ref_capacity = ref_capacity  # capacity of a niche if there was only one
    self.niche_capacity = int(ref_capacity / num_niches)  # capacity of a niche normalized for number of niches
    self.current_capacity = self.niche_capacity * self.mean
    self.climate = self.mean
    self.climate_values = []

  def step(self, gen):

    self.climate_values.append(self.mean)
    return self.mean


