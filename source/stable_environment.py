from environment import Env


class StableEnv(Env):
    """ Implements a stable environment where the reference environmental state does not change with generations.
    """

    def __init__(self, mean, ref_capacity, num_niches):
        self.type = "stable"
        self.low = mean
        self.max_value = self.low

        super().__init__(mean, ref_capacity, num_niches, self.low)

    def step(self, gen):
        super().step()
        return self.mean
