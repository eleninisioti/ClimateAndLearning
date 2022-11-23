

from environment import Env
import numpy as np
import math


class SinEnv(Env):
    """ Implements a sinusoidal environment characterized by the amplitude and period.
    """

    def __init__(self, mean, ref_capacity, num_niches, period, amplitude, decay_construct):
        self.type = "sin"
        self.omega = 2 * math.pi / period # natural frequency of sinusoid
        self.amplitude = amplitude
        self.low = amplitude + mean  # lowest amplitude the sinusoid will reach
        self.max_value = self.amplitude + self.low
        super().__init__(mean, ref_capacity, num_niches, self.low, decay_construct)

    def step(self, gen, niche_constructions):
        # A is the amplitude and omega is the frequency in rad
        self.mean = self.amplitude * np.sin(gen * self.omega) + self.low
        super().step(niche_constructions)
        return self.mean
