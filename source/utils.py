import numpy as np


def compute_selection_strength( log):
    climate = log["Climate_avg"].to_list()
    pop_mean = log["Mean"].to_list()
    strength = [np.abs(el - pop_mean[idx]) for idx, el in enumerate(climate)]
    log["Selection"] = strength
    return log



def compute_survival(log, num_niches):

    num_latitudes = num_niches
    window = 11
    all_DI = []
    for lat in range(num_latitudes):
        climate = log["Climate"].to_list()
        num_gens = len(climate)
        survivals = []

        for gen in range(num_gens):
            lat_climate = climate[gen] + 0.01 *lat

            current_mean = log["Mean"].to_list()[gen]
            current_sigma = log["SD"].to_list()[gen]
            if ((current_mean-2*current_sigma) < lat_climate) and (lat_climate < (current_mean+2*current_sigma)):
                survived = 1
            else:
                survived = 0
            survivals.append(survived)

        all_DI.append(list(np.convolve(survivals, np.ones(window, dtype=int), 'valid')))

    dispersal = []
    for el in range(len(all_DI[0])):
        sum_disp = 0
        for lat_DI in all_DI:
            sum_disp += lat_DI[el]
        dispersal.append(sum_disp)

    while len(dispersal) < num_gens:
        dispersal = [0] + dispersal

    if len(dispersal) > num_gens:
        dispersal = dispersal[:num_gens]

    log["Dispersal"] = dispersal
    #log["Generation_dispersal"] = list(range(0, num_gens-window+2))
    return log, np.sum(survivals)