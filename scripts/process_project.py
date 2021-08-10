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
if os.path.exists(top_dir + current_project + '/log_total.pickle'):
  log, env_profile = pickle.load(open(top_dir + current_project + '/log_total.pickle', 'rb'))
  env_profile["ncycles"] = 2
  env_profile["cycle"] = 2500
  plotter = Plotter(current_project, env_profile)
  plotter.plot_with_conf(log, [1,0,0,1], 2)





