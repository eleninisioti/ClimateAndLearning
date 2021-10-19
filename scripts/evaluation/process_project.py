import pickle
import sys
import os

sys.path.insert(0, "../scripts")
sys.path.insert(0, os.getcwd())

from plotter import Plotter
import numpy as np
import pandas as pd
import click


def run(project, trials, climate_noconf):
    top_dir = "../projects/"


    _, env_profile = pickle.load(open(top_dir + project + '/log_total.pickle', 'rb'))
    for trial in range(0, trials + 1):

        if os.path.isfile(top_dir + project + '/trials/trial_' + str(trial)
                          + '/log.pickle'):
            log = pickle.load(open(top_dir + project + '/trials/trial_' + str(trial)
                                   + '/log.pickle', 'rb'))
            if not trial:
                log_df = log
            else:
                log_df = log_df.append(log)

    plotter = Plotter(project, env_profile, climate_noconf=climate_noconf)
    # plotter.plot_with_conf(log_df, [1,0,0,1], 2)
    # plotter.plot_evolution_with_conf(log_df, [1, 0, 1, 0], cycles=1)

    plotter.plot_evolution_with_conf(log_df, [1, 0, 0, 0, 0])
    plotter.plot_evolution_with_conf(log_df, [1, 1, 1, 1, 1])
    #plotter.plot_species_with_conf(log_df, [1, 1, 0, 0])
    #plotter.plot_species_with_conf(log_df, [1, 0, 1, 0])
    #plotter.plot_species_with_conf(log_df, [1, 0, 0, 1])


# def heatmap(x_values, y_values, results):


if __name__ == "__main__":
    if len(sys.argv) == 2:
        p= sys.argv[1]
        run(project="../projects/" + p, trials=1, climate_noconf=0)
    else:
        top_dir = "Maslin/debug"
        projects = [os.path.join(top_dir, o) for o in os.listdir("../projects/" + top_dir)]
        for p in projects[1:]:
            if "plots" not in p:
                run(project=p, trials=1, climate_noconf=0)
    #plotter = Plotter(project=top_dir)
    #plotter.plot_heatmap(top_dir=top_dir, trials=1, x_variables=["var_freq", "factor_time_abrupt"], y_variables=[
    #    "Climate"])

# x_values = []
