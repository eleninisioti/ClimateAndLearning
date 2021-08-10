import pickle
import sys
sys.path.insert(0, "../source")
from plotter import Plotter
import os
import numpy as np
import pandas as pd

current_project = "Maslin/presentation"
top_dir = "../projects/"
trials = 2


# load data

log, env_profile = pickle.load(open(top_dir + current_project + '/log_total.pickle', 'rb'))
env_profile["ncycles"] = 2
env_profile["cycle"] = 2500
plotter = Plotter(current_project, env_profile=env_profile)
for trial in range(0, trials):
  trial_log = pickle.load(open(top_dir + current_project + '/trials/trial_' + str(trial) + '/log.pickle', 'rb'))

  plotter.plot_evolution(trial_log, trial, [1,0,1,0], 2)

  plotter.plot_species(trial_log, trial, [1,1,0,0], 2)

  plotter.plot_species(trial_log, trial, [1,0,0,1], 2)
  plotter.plot_species(trial_log, trial, [1,1,1,1], 2)






#plotter = Plotter(current_project, env_profile)
#plotter.plot_with_conf(log)



