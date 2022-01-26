""" This script can be used to produce plots used in our Gecco paper.
"""
import sys
import os
import yaml
import matplotlib.pyplot as plt
import pickle5 as pickle
import numpy as np
import pandas as pd
import seaborn as sns

params = {'legend.fontsize': 8,
          "figure.autolayout": True}
plt.rcParams.update(params)
ci=95


def sigma_constant(results_dir):
    """ Plot with sigma in the vertical axis, climate value in the horizontal and different lines for number of niches"
    """
    # find all projects
    projects = [os.path.join(results_dir, o) for o in os.listdir(results_dir)]
    projects = [el for el in projects if "plots" not in el ]

    count = 0
    for p in projects:
        config_file = p + "/config.yml"

        with open(config_file, "rb") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        trial_dirs = list(next(os.walk(p+ "/trials"))[1])
        for trial, trial_dir in enumerate(trial_dirs):
            # load outcome of trial
            log = pickle.load(open(p + "/trials/" + trial_dir + '/log.pickle', 'rb'))
            trial_sigma = np.mean(log["SD"][-100:])
            new_row = pd.DataFrame.from_dict({ 'SD': [trial_sigma],
                                                      "Trial": [trial],
                                        "Num_niches": [config.num_niches],
                                        "Climate": [config.climate_mean_init]})
            if not count:
                results = new_row
            else:
                results = results.append(new_row)
            count =1
    niches = list(set(results["Num_niches"].to_list()))
    niches.sort()
    cm = 1 / 2.54
    plt.figure(figsize=(8.48*cm, 6*cm))
    for niche in niches:
        results_niche = results.loc[results['Num_niches'] == niche]
        sns.lineplot(data=results_niche, x="Climate", y="SD", ci=ci, label="$N=$" + str(
            niche))
    plt.yscale('log')
    #plt.ylim([10**(-19), 100])
    plt.xlabel("$e_0$, Reference Environmental State")
    plt.ylabel("$\\bar{\sigma}$, Average Plasticity")
    plt.legend(loc="best")
    plt.savefig(results_dir + "/plots/sigma_constant.png")

    plt.clf()



    # for each project get niches and climate
if __name__ == "__main__":
    results_dir = "../projects/papers/gecco/heatmaps_stable/s2_g2_1"
    sigma_constant(results_dir=results_dir)