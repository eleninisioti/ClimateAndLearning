from environment import Env


class StableEnv(Env):

    def __init__(self, mean, ref_capacity, num_niches):
        self.type = "stable"
        self.low = mean
        super().__init__(mean, ref_capacity, num_niches, self.low)

    def step(self, gen):
        super().step()
        return self.mean
