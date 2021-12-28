import pickle5 as pickle
#import pickle
import sys
import os
import yaml
sys.path.insert(0, "../scripts")
sys.path.insert(0, os.getcwd())

from plotter import Plotter
import numpy as np
import pandas as pd
import click


def run(project, trials, climate_noconf):
    top_dir = "../projects/"

    """try:
    
        log, env_profile = pickle.load(open(top_dir + project + '/log_total.pickle', 'rb'))
    except:
        print("log file does not exist for project ", top_dir + project)
        return 1"""


    trial_dirs = list(next(os.walk(top_dir+project +"/trials"))[1])
    for trial, trial_dir in enumerate(trial_dirs):
        print(top_dir + project + "/trials/" + trial_dir + '/log.pickle')

        log = pickle.load(open(top_dir + project + "/trials/" + trial_dir + '/log.pickle', 'rb'))
        if trial ==0:
            log_df = log
        else:
            log_df = log_df.append(log)

    plotter = Plotter(project, env_profile=[], climate_noconf=climate_noconf)
    # plotter.plot_with_conf(log_df, [1,0,0,1], 2)
    #plotter.plot_evolution_with_conf(log_df, [1, 0, 1, 0], cycles=1)

    #plotter.plot_evolution_with_conf(log_df, [1, 0, 0, 0, 0, 0, 0])
    skip_lines = 1
    with open(top_dir + project + "/config.yml") as f:
        for i in range(skip_lines):
            _ = f.readline()
        config = yaml.load(f)

    if config["only_climate"]:
        plotter.plot_evolution_with_conf(log_df, [1,0,0,0,0,0,0, 0, 0], config["num_niches"])
    else:
        plotter.plot_evolution_with_conf(log, [1, 1, 1, 1, 1, 0, 0, 1, 1], config["num_niches"])
        plotter.plot_selection_pressure(log_df, config["num_niches"])

    #plotter.plot_evolution_with_conf(log_df, [1,0,0,0,0,0,0, 0, 0], config["num_niches"])
    #plotter.plot_evolution_with_conf(log, [1, 1, 1, 1, 1, 0, 0, 0, 1], config["num_niches"])
    #plotter.plot_species_with_conf(log_df, [1, 1, 1])
    #plotter.plot_evolution_with_conf(log_df, [1, 0, 0, 1, 0, 0, 0])

    #plotter.plot_species_with_conf(log_df, [1, 1, 0, 0])
    #plotter.plot_species_with_conf(log_df, [1, 0, 1, 0])
    #plotter.plot_species_with_conf(log_df, [1, 0, 0, 1])


# def heatmap(x_values, y_values, results):


if __name__ == "__main__":
    if len(sys.argv) == 2:
        p = sys.argv[1]
        run(project= p, trials=1, climate_noconf=0)
    else:
        top_dir = "Maslin/debug/parametric/28_12_2021_debug"
        #top_dir = "report/final/1D_mutate"
        projects = [os.path.join(top_dir, o) for o in os.listdir("../projects/" + top_dir)]
        for idx, p in enumerate(projects):
            if "plots" not in p:
                print(idx, len(projects))
                run(project=p, trials=1, climate_noconf=0)
    #plotter = Plotter(project=top_dir)
    #plotter.plot_heatmap(top_dir=top_dir, trials=1, x_variables=["var_freq", "factor_time_abrupt"], y_variables=[
    #    "Climate"])

# x_values = []
