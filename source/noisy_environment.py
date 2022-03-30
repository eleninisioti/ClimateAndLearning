from environment import Env
import numpy as np


class NoisyEnv(Env):

    def __init__(self, mean, ref_capacity, num_niches, std):
        self.type = "noisy"
        self.noise_mean = mean
        self.noise_std = std
        self.low = mean
        super().__init__(mean, ref_capacity, num_niches, self.low)


    def step(self, gen):
        # A is the amplitude and omega is the frequency in rad
        self.mean = np.random.normal(self.noise_mean, self.noise_std, 1)[0]
        super().step()
        return self.mean

