
class Env:
    """ Abstract class defining the interface of all environments.
    """

    def __init__(self, mean, ref_capacity, num_niches, init, decay_construct):
        """ Class constructor

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
        self.decay_construct = decay_construct

        self.niche_capacity = int(ref_capacity / num_niches)  # capacity of a niche normalized for number of niches
        self.current_capacity = self.niche_capacity * self.mean
        self.climate = init
        self.climate_values = []
        self.niches = {}
        self.max_climate = 10

        southest_lat = -int(self.num_niches / 2)
        northest_lat = int(self.num_niches / 2 + 0.5)
        self.niche_constructions = {}
        for lat in range(southest_lat, northest_lat):
            self.niche_constructions[lat] = 0
        self.niche_constructions[999] = 0

        self.update_niches(self.niche_constructions)

    def update_niches(self, niche_constructions):
        southest_lat = -int(self.num_niches / 2)
        northest_lat = int(self.num_niches / 2 + 0.5)
        for lat in range(southest_lat, northest_lat):
            lat_climate = self.mean + 0.01 * lat
            if lat in niche_constructions.keys():
                lat_climate += niche_constructions[lat]
                lat_climate = min([lat_climate, self.max_climate])
            else:
                niche_constructions[lat] = 0
            niche_constructions[lat] = niche_constructions[lat]*self.decay_construct


            #print("consturciton is ", str(niche_constructions[lat]), " in niche ", str(lat))
            niche_capacity = max(int(lat_climate * self.niche_capacity), 0)
            #niche_capacity = min([niche_capacity, 50])
            #print(" for niche",lat, "niche capacity", niche_capacity," with constructions ",
            #str(niche_constructions[lat]))

            self.niches[lat] = {"climate": lat_climate,
                                "capacity": niche_capacity,
                                "lat": lat,
                                "constructed": niche_constructions[lat]}

    def step(self,  niche_constructions):
        """ Move the environment to the next generation. Updates the climate and capacity of niches based on the reference environmental state.

        Parameters
        ----------
        gen: int
            current generation
        """
        self.current_capacity = self.mean * self.niche_capacity
        self.climate_values.append(self.mean)
        self.update_niches(niche_constructions)


