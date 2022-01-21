import numpy as np


def compute_SoS(log, log_niches, num_niches):
    """ Compute Strength of Selection"""
    trials = len(log_niches)
    all_strengths = []
    for trial in range(trials):

        climate_mean = []
        for gen, el in enumerate(list(log["Climate"])):
            climate_values = []
            for lat in range(-int(num_niches/2), int(num_niches/2 +0.5)):
                lat_climate = list(log["Climate"])[gen] + 0.01 *lat
                if lat_climate in log_niches[trial]["inhabited_niches"][gen]:
                    climate_values.append(lat_climate)
            if not len(climate_values):
                climate_values = [0]
            climate_mean.append(np.mean(climate_values))

        pop_mean = log["Mean"]
        if type(pop_mean) is not list:
            pop_mean = pop_mean.to_list()
        strength = [0 if el==0 else np.abs(el - pop_mean[idx]) for idx, el in enumerate(climate_mean)]
        all_strengths.append(strength)
    log["Selection"] = np.mean(np.array(all_strengths), axis=0)
    return log



def compute_dispersal(log, log_niches, num_niches):

    num_latitudes = num_niches
    window = 10
    all_DI = []
    trials = len(log_niches)
    all_dispersals = []
    for trial in range(trials):
        for lat in range(-int(num_latitudes/2), int(num_latitudes/2 +0.5)):
            survivals = []
            climate = log["Climate"].to_list()
            num_gens = len(climate)
            #inhabited_niches = [len(el) for el in log["inhabited_niches"].to_list()]
            inhabited_niches = log_niches[trial]["inhabited_niches"]
            for gen in range(num_gens):
                lat_climate = climate[gen] + 0.01 * lat

                if lat_climate in inhabited_niches[gen]:
                    survivals.append(1)
                else:
                    survivals.append(0)
            DI = list(np.convolve(survivals, np.ones(window, dtype=int), 'valid'))
            DI = [1 if el==window else 0 for el in DI]
            all_DI.append(DI)


        dispersal = []
        for el in range(len(all_DI[0])):
            sum_disp = 0
            for lat_DI in all_DI:
                sum_disp += lat_DI[el]
            dispersal.append(sum_disp)

        while len(dispersal) < num_gens:
            dispersal = [1] + dispersal

        all_dispersals.append(dispersal)


    log["Dispersal"] = np.mean(np.array(all_dispersals), axis=0)
    return log