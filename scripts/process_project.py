import pickle
import sys
sys.path.insert(0, "../source")
from plotter import Plotter
import os
import numpy as np
import pandas as pd

current_project = "Maslin/debug_conf"
top_dir = "../projects/"
trials = 10

# load data
if os.path.exists(top_dir + current_project + '/log_total.pickle'):
  log, env_profile = pickle.load(open(top_dir + current_project + '/log_total.pickle', 'rb'))
else:
  log_df = pd.DataFrame(columns=["Generation", "Trial", "Climate",
                                 "Fitness", "Mean", "SD", "Total_Diversity",
                                 "Specialists_Extinct", "Specialists_Number",
                                 "Specialists_Diversity", "Specialists_Diversity_Mean",
                                 "Specialists_Diversity_SD", "Generalists_Extinct",
                                 "Generalists_Number", "Generalists_Diversity",
                                 "Generalists_Diversity_Mean","Generalists_Diversity_SD"],
                        dtype=np.float)
  for trial in range(trials):
    trial_log, env_profile = pickle.load(open(top_dir + current_project + '/log_total_' + str(trial) + '.pickle', 'rb'))
    log_df = log_df.append(pd.DataFrame.from_dict(trial_log))

plotter = Plotter(current_project, env_profile)
plotter.plot_with_conf(log)



