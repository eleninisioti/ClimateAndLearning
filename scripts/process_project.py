import pickle
import sys
sys.path.insert(0, "../source")
from plotter import Plotter
import os
import numpy as np
import pandas as pd

current_project = "Maslin/present_conf"
top_dir = "../projects/"
trials = 50

log_df = pd.DataFrame(columns=["Generation", "Trial", "Climate",
                               "Fitness", "Mean", "SD", "Total_Diversity",
                               "Specialists_Extinct", "Specialists_Number",
                               "Specialists_Diversity", "Specialists_Diversity_Mean",
                               "Specialists_Diversity_SD", "Generalists_Extinct",
                               "Generalists_Number", "Generalists_Diversity",
                               "Generalists_Diversity_Mean", "Generalists_Diversity_SD"],
                      dtype=np.float)
for trial in range(0, trials):
  if trial!=10 and trial!=25:
    log, env_profile = pickle.load(open(top_dir + current_project + '/log_total_part_' + str(trial)
                                        + '.pickle', 'rb'))
    log_df = log_df.append(log)

plotter = Plotter(current_project, env_profile)
#plotter.plot_with_conf(log_df, [1,0,0,1], 2)

# plotter.plot_evolution_with_conf(log_df, [1, 0, 1, 0], 2)
# plotter.plot_species_with_conf(log_df, [1, 1, 0, 0], 2)
plotter.plot_species_with_conf(log_df, [1, 0, 1, 0], 2)
plotter.plot_species_with_conf(log_df, [1, 0, 0, 1], 2)







