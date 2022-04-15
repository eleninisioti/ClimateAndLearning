import numpy as np


def find_index(trial_dir):
    # find the index of current trial
    trial = trial_dir.find("trial_")
    trial = int(trial_dir[(trial + 6):])
    return trial

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
        if config.genome_type == "evolv":
            label += "$R_{evolve}$"
        elif config.genome_type == "no-evolv":
            label += "$R_{no-evolve}$"
    return label


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
    trials = list(set(log["Trial"]))
    for trial_idx, trial in enumerate(trials):
        all_dispersals = []
        all_DI = []
        log_trial = log.loc[(log['Trial'] == trial)]
        climate = log_trial["Climate"].to_list()
        # inhabited_niches = [len(el) for el in log["inhabited_niches"].to_list()]
        inhabited_niches = log_niches[trial]["inhabited_niches"]
        num_gens = min([len(inhabited_niches), len(climate)])



        for lat in range(-int(num_latitudes/2), int(num_latitudes/2 +0.5)):
            survivals = []

            for gen in range(num_gens):
                lat_climate = climate[gen] + 0.01 * lat
                survival=0
                for el in inhabited_niches[gen]:
                    if (np.abs(el-lat_climate)) < 0.01:
                        survival = 1



                survivals.append(survival)
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

        if not trial_idx:
            new_log = log_trial
        else:
            new_log = new_log.append(log_trial)

    return new_log
