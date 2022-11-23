from environment import Env
import numpy as np


class NoisyEnv(Env):
    """ Implements a noisy environment characterized by the standard deviation.
    """

    def __init__(self, mean, ref_capacity, num_niches, std, decay_construct):
        self.type = "noisy"
        self.noise_mean = mean
        self.noise_std = std
        self.low = mean
        self.max_value = self.noise_mean + self.noise_std*4

        super().__init__(mean, ref_capacity, num_niches, self.low, decay_construct)


    def step(self, gen, niche_constructions):
        # A is the amplitude and omega is the frequency in rad
        self.mean = np.random.normal(self.noise_mean, self.noise_std, 1)[0]
        super().step(niche_constructions)
        return self.mean

