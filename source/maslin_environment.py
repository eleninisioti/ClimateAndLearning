""" Implements the climate model described in Maslin."""

from environment import Env
from numpy.random import normal
import numpy as np
from scipy.interpolate import interp1d


class MaslinEnv(Env):

    def __init__(self, mean, ref_capacity, num_niches, factor_time_abrupt=1,
                 factor_time_variable=1, factor_time_steady=1, var_freq=5, var_SD=0.2):
        """ Constructor for Class MaslinEnv.

    Parameters
    ----------
    mean: float
        environmental state at initialization

    ref_capacity: int
        the capacity of a niche (before multiplying with the environmental state) if there was a single niche

    num_niches: int
        number of niches

    factor_time_abrupt: float
        scale duration of transition

    factor_time_variable: float
        scale duration of variable period

    factor_time_steady: float
        scale duration of stable period

    var_freq: float
        frequency of variability during variable period

    var_SD: float
      magnitude of variability during variable period
    """

        super().__init__(mean)
        self.type = "combined"
        self.num_niches = num_niches
        self.ref_capacity = ref_capacity  # capacity of a niche if there was only one
        self.niche_capacity = int(ref_capacity / num_niches)  # capacity of a niche normalized for number of n
        self.low = mean  # mean of environment  during low-resources periods
        self.high = self.low + 2  # mean of environment  during high-resources periods
        self.mean = self.low
        self.current_capacity = self.niche_capacity * self.mean

        # ------ define variability ------
        # breakpoints for determining the different phases
        self.generation_duration = 15  # how many years a generation is
        self.factor_time_abrupt = factor_time_abrupt  # scales abrupt transition in time
        self.factor_time_variable = factor_time_variable  # scales abrupt transition in time
        self.factor_time_steady = factor_time_steady  # scales abrupt transition in time
        self.var_freq = var_freq  # scales abrupt transition in size
        self.SD = var_SD

        """self.b1 = int(8000 / self.generation_duration * self.factor_time_steady)
        self.b2 = int(self.b1 + int(300 / self.generation_duration) * self.factor_time_abrupt)
        self.b3 = int(self.b2 + int(8000 / self.generation_duration) * self.factor_time_steady)
        self.b4 = int(self.b3 + int(2000 / self.generation_duration) * self.factor_time_variable)
        self.b5 = int(self.b4 + int(8000 / self.generation_duration) * self.factor_time_steady)"""
        self.b1 = 200  # low-> high start point
        self.b2 = int(self.b1 + 50 * factor_time_abrupt)  # low-> high end point
        self.b3 = int(self.b2 + 50 * factor_time_steady)  # high -> low start point
        self.b4 = int(self.b3 + 50 * factor_time_abrupt)  # high -> low end point
        self.b5 = self.b4 + 200
        self.rate1 = (self.high - self.low) / (self.b2 - self.b1)
        self.rate2 = (self.high - self.low) / (self.b4 - self.b3)



        # logging
        self.b1_values = [self.b1]
        self.b2_values = [self.b2]
        self.b3_values = [self.b3]
        self.b4_values = [self.b4]
        self.b5_values = [self.b5]
        self.climate_values = []
        self.cycles = 1

    def step(self, gen):
        """ Environment proceeds by one generation.

    Parameters
    ----------

    gen: int
      current generation index
    """

        if gen < self.b1:  # low period
            climate = self.mean

        elif self.b1 <= gen < self.b2:  # abrupt transition
            climate = self.climate_values[-1] + self.rate1

        elif self.b2 <= gen < self.b3:  # high period
            climate = self.high
            self.steps_var = 0

        elif gen == self.b3:  # step at which highly-variable period begins
            # generations at which there is variability using desired frequency
            x_points = [self.b3]
            while x_points[-1] < self.b4:
                x_points.append(int(x_points[-1] + self.var_freq))
            x_points.append(self.b4)

            y_points = []
            for el in x_points:
                offset = normal(0, self.SD)
                y_points.append(self.high - self.rate2 * (el - self.b3) + offset)

            y_points = [max([self.low, min([el, self.high])]) for el in y_points]
            self.interp = interp1d(x_points, y_points)
            climate = self.high

        elif self.b3 < gen < self.b4:  # highly-variable period
            climate = float(self.interp(gen))
            climate = np.min([self.high, climate])
            climate = np.max([self.low, climate])

        elif self.b4 <= gen < self.b5:  # low-resources period
            climate = self.low

        elif gen == self.b5:  # end of current cycle-start new cycle

            climate = self.low
            self.cycles += 1
            """self.b1 = int(gen + int(8000 / self.generation_duration) * self.factor_time_steady)
            self.b2 = int(self.b1 + int(300 / self.generation_duration) * self.factor_time_abrupt)
            self.b3 = int(self.b2 + int(8000 / self.generation_duration) * self.factor_time_steady)
            self.b4 = int(self.b3 + int(2000 / self.generation_duration) * self.factor_time_variable)
            self.b5 = int(self.b4 + int(8000 / self.generation_duration) * self.factor_time_steady)
            """
            self.b1 = gen + 200  # low-> high start point
            self.b2 = int(self.b1 + 50 * self.factor_time_abrupt)  # low-> high end point
            self.b3 = int(self.b2 + 50 * self.factor_time_steady)  # high -> low start point
            self.b4 = int(self.b3 + 50 * self.factor_time_abrupt)  # high -> low end point
            self.b5 = self.b4 + 200
            self.b1_values.append(self.b1)
            self.b2_values.append(self.b2)
            self.b3_values.append(self.b3)
            self.b4_values.append(self.b4)
            self.b5_values.append(self.b5)

        self.current_capacity = climate * self.niche_capacity
        self.climate_values.append(climate)
        self.mean = climate
        return climate
