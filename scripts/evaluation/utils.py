import numpy as np
import pandas as pd


def find_label(config, parameter="selection"):
    label = ""
    if parameter == "selection":
        if config.selection_type == "NF":
            label += "NF-selection"
        elif config.selection_type == "N":
            label += "N-selection"
        elif config.selection_type == "F":
            label += "F-selection"
    elif parameter == "genome":
        label += config.genome_type
    return label

def compute_SoS(log, log_niches, num_niches):
    """ Compute Strength of Selection.

    Parameters
    ---------
    log_niches: Dataframe
        information about niches

    num_niches: int
        number of niches
    """
    trials = len(log_niches)
    for trial in range(1,trials+1):
        all_strengths = []
        log_trial = log.loc[(log['Trial'] == trial)]
        climate_mean = []
        num_gens = min([len(log_niches[trial]["inhabited_niches"]), len(list(log_trial["Climate"]))])
        log_trial = log_trial.loc[log_trial['Generation'] < num_gens]

        for gen in range(num_gens):
            climate_values = []
            for lat in range(-int(num_niches/2), int(num_niches/2 +0.5)):
                lat_climate = list(log_trial["Climate"])[gen] + 0.01 *lat
                if lat_climate in log_niches[trial]["inhabited_niches"][gen]:
                    climate_values.append(lat_climate)
            if not len(climate_values):
                climate_values = [0]
            climate_mean.append(np.mean(climate_values))

        pop_mean = log_trial["Mean"]
        if type(pop_mean) is not list:
            pop_mean = pop_mean.to_list()
        strength = [0 if el==0 else np.abs(el - pop_mean[idx]) for idx, el in enumerate(climate_mean)]
        all_strengths.append(strength)
        log_trial["Selection"] = np.mean(np.array(all_strengths), axis=0)
        if trial==1:
            new_log = log_trial
        else:
            new_log = new_log.append(log_trial)

    return new_log



def compute_dispersal(log, log_niches, num_niches):
    """ Compute dispersal

    Parameters
    ---------
    log_niches: Dataframe
        information about niches

    num_niches: int
        number of niches
    """
    num_latitudes = num_niches
    window = 10
    trials = len(log_niches)
    for trial in range(1,trials+1):
        all_dispersals = []
        all_DI = []
        log_trial = log.loc[(log['Trial'] == trial)]
        climate = log_trial["Climate"].to_list()
        # inhabited_niches = [len(el) for el in log["inhabited_niches"].to_list()]
        inhabited_niches = log_niches[trial-1]["inhabited_niches"]
        num_gens = min([len(inhabited_niches), len(climate)])

        for lat in range(-int(num_latitudes/2), int(num_latitudes/2 +0.5)):
            survivals = []

            for gen in range(num_gens):
                lat_climate = climate[gen] + 0.01 * lat

                if lat_climate in inhabited_niches[gen]:
                    survivals.append(1)
                else:
                    survivals.append(0)
            if not len(survivals):
                survivals = [0]*window
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

        x = np.mean(np.array(dispersal), axis=0)
        log_trial["Dispersal"] = x

        if trial==1:
            new_log = log_trial
        else:
            new_log = new_log.append(log_trial)

    return new_log
