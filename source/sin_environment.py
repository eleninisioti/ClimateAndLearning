""" Implements an environment that changes based on a sine-wave. Default parameters taken from experiments in Section
4.2
of 'Evolution and dispersal under climatic instability: a simple evolutionary algorithm'
"""

from environment import Env
import numpy as np
import math


class SinEnv(Env):

    def __init__(self, mean, ref_capacity, num_niches, period, amplitude):
        self.type = "sin"
        self.omega = 2 * math.pi / period
        self.amplitude = amplitude
        self.low = amplitude + mean  # low amplitude
        super().__init__(mean, ref_capacity, num_niches, self.low)

    def step(self, gen):
        # A is the amplitude and omega is the frequency in rad
        self.mean = self.amplitude * np.sin(gen * self.omega) + self.low
        super().step()
        return self.mean
