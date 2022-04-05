""" Describes the abstract interface of an environment
"""


class Env:

    def __init__(self, mean, ref_capacity, num_niches, init):
        """

        Parameters
        ----------
        mean: float
          initial environmental state of reference niche

        ref_capacity: int
          capacity of the simulation if there was a single niche

        num_niches: int
          number of niches

        init: float
          initial value of climate
        """
        self.mean = mean
        self.ref_capacity = ref_capacity
        self.num_niches = num_niches
        self.epsilon = 0.01 # distance between adjacent niches

        self.niche_capacity = int(ref_capacity / num_niches)  # capacity of a niche normalized for number of niches
        self.current_capacity = self.niche_capacity * self.mean
        self.climate = init
        self.climate_values = []
        self.niches = {}
        self.update_niches()

    def update_niches(self):
        x=-int(self.num_niches / 2)
        y= int(self.num_niches / 2 + 0.5)
        for lat in range(x,y):
            lat_climate = self.mean + 0.01 * lat
            niche_capacity = max(int(lat_climate * self.niche_capacity),0)
            self.niches[lat] = {"climate": lat_climate, "capacity": niche_capacity}

    def step(self):
        self.current_capacity = self.mean * self.niche_capacity
        self.climate_values.append(self.mean)
        self.update_niches()

