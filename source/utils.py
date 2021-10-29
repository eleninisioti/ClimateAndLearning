import numpy as np


def compute_selection_strength( log):
    climate = log["Climate"].to_list()
    pop_mean = log["Mean"].to_list()
    strength = [np.abs(el - pop_mean[idx]) for idx, el in climate]
    log["Selection"] = strength
    return log

def compute_survival(log):

    num_latitudes = 20
    window = 5
    all_DI = []
    for lat in range(-int(num_latitudes/2), int(num_latitudes/2) ):
        climate = log["Climate"].to_list()
        num_gens = len(climate)
        survivals = []

        for gen in range(num_gens):
            lat_climate = climate[gen] = lat

            current_mean = log["Mean"]["Generation"==gen]
            current_sigma = log["Sigma"]["Generation" == gen]

            if (current_mean-2*current_sigma) < lat_climate  and lat_climate >  (current_mean-2*current_sigma):
                survived = 1
            else:
                survived = 0
            survivals.append(survived)

        all_DI.append(np.convolve(survivals, np.ones(window, dtype=int), 'valid'))

    dispersal = []
    for el in range(all_DI[0]):
        sum_disp = 0
        for lat_DI in all_DI:
            sum_disp += lat_DI[el]

    log["Dispersal"] = dispersal
    log["Generation_dispersal"] = list(range(0, num_gens, window))
    return log