import pickle
import sys
sys.path.insert(0, "../source")
from plotter import Plotter
import os
import numpy as np
import pandas as pd
import click


def run(project, trials, climate_noconf):
  top_dir = "../projects/"
  log_df = pd.DataFrame(columns=["Generation", "Trial", "Climate",
                                 "Fitness", "Mean", "SD", "Total_Diversity",
                                 "Specialists_Extinct", "Specialists_Number",
                                 "Specialists_Diversity", "Specialists_Diversity_Mean",
                                 "Specialists_Diversity_SD", "Generalists_Extinct",
                                 "Generalists_Number", "Generalists_Diversity",
                                 "Generalists_Diversity_Mean", "Generalists_Diversity_SD"],
                        dtype=np.float)

  _, env_profile = pickle.load(open(top_dir + project + '/log_total.pickle', 'rb'))
  for trial in range(0, trials):

    if os.path.isfile(top_dir + project + '/trials/trial_' + str(trial)
                                          + '/log.pickle'):

      log = pickle.load(open(top_dir + project + '/trials/trial_' + str(trial)
                                          + '/log.pickle', 'rb'))
      log_df = log_df.append(log)

  plotter = Plotter(project, env_profile, climate_noconf=climate_noconf)
  #plotter.plot_with_conf(log_df, [1,0,0,1], 2)
  #plotter.plot_evolution_with_conf(log_df, [1, 0, 1, 0], cycles=1)

  plotter.plot_evolution_with_conf(log_df, [1, 1, 0, 0])
  plotter.plot_evolution_with_conf(log_df, [1, 1, 0, 1])
  plotter.plot_species_with_conf(log_df, [1, 1, 0, 0])
  plotter.plot_species_with_conf(log_df, [1, 0, 1, 0])
  plotter.plot_species_with_conf(log_df, [1, 0, 0, 1])


def heatmap(x_values, y_values, results):


if __name__ == "__main__":
  top_dir = "../projects/Maslin/parametric/"
  projects = [os.path.join(top_dir, o) for o in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, o))]
  for p in projects:
    run(project=p, trials=10, climate_noconf=0)

  x_values = []











