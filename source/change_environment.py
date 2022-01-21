"""
Implements an environment where the climate curve is a pulse, as in Figure 2 of 'Evolution and dispersal under climatic
 instability: a simple evolutionary algorithm'
"""

from environment import Env


class ChangeEnv(Env):

    def __init__(self, mean, ref_capacity, num_niches, factor_time_abrupt, factor_time_steady):
        """ Constructor for Class ChangeEnv.

        Parameters
        ----------
        mean: float
            environmental state at initialization

        ref_capacity: int
            the capacity of a niche (before multiplying with the environmental state) if there was a single niche

        num_niches: int
            number of niches

        factor_time_abrupt: float
            scale transition

        factor_time_steady: float
            scale stable period
        """
        super().__init__(mean)
        self.type = "change"
        self.num_niches = num_niches
        self.ref_capacity = ref_capacity  # capacity of a niche if there was only one
        self.niche_capacity = int(ref_capacity / num_niches)  # capacity of a niche normalized for number of niches
        self.low = mean
        if mean < 1:
            self.high = mean + 2
        else:
            self.high = mean + 0.5

        self.mean = self.low  # current state of climate
        self.current_capacity = self.niche_capacity * self.mean
        self.factor_time_abrupt = factor_time_abrupt
        self.factor_time_steady = factor_time_steady

        # ------ define variability ------
        self.b1 = 200  # low-> high start point
        self.b2 = int(self.b1 + 50 * factor_time_abrupt)  # low-> high end point
        self.b3 = int(self.b2 + 50 * factor_time_steady) # high -> low start point
        self.b4 = int(self.b3 + 50 * factor_time_abrupt)  # high -> low end point
        self.b5 = self.b4 + 200
        self.rate = (self.high - self.low) / (self.b2 - self.b1)  # symmetrical transitions

        # logging
        self.climate_values = []
        self.cycles = 0
        self.b1_values = [self.b1]
        self.b2_values = [self.b2]
        self.b3_values = [self.b3]
        self.b4_values = [self.b4]
        self.b5_values = [self.b5]

    def step(self, gen):
        """ Environment proceeds by one generation

        Parameters
        ----------

        gen: int
          current generation index
        """
        if gen < self.b1:
            climate = self.mean
        elif self.b1 <= gen < self.b2:
            climate = self.climate_values[-1] + self.rate
        elif self.b2 <= gen < self.b3:
            climate = self.climate_values[-1]
        elif self.b3 <= gen < self.b4:
            climate = self.climate_values[-1] - self.rate
        elif self.b4 <= gen < self.b5:
            climate = self.climate_values[-1]
        elif gen == self.b5:
            self.b1 = gen + 200  # low-> high start point
            self.b2 = int(self.b1 + 50 * self.factor_time_abrupt)  # low-> high end point
            self.b3 = int(self.b2 + 50 * self.factor_time_steady)  # high -> low start point
            self.b4 = int(self.b3 + 50 * self.factor_time_abrupt)
            self.b5 = self.b4 + 200
            self.rate = (self.high - self.low) / (self.b2 - self.b1)  # symmetrical transitions
            climate = self.climate_values[-1]
            print(self.b1, self.b2, self.b3, self.b4, self.b5)
            self.cycles += 1
            self.b1_values.append(self.b1)
            self.b2_values.append(self.b2)
            self.b3_values.append(self.b3)
            self.b4_values.append(self.b4)
            self.b5_values.append(self.b5)

        self.current_capacity = climate * self.niche_capacity
        self.climate_values.append(climate)
        self.mean = climate
        return climate
