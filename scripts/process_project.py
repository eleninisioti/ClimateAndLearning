import pickle
import sys
sys.path.insert(0, "../source")
from plotter import Plotter

current_project = "Maslin/debug_conf"
top_dir = "../projects/"

# load data
log, env_profile = pickle.load(open(top_dir + current_project + '/log_total.pickle', 'rb'))


plotter = Plotter(current_project, env_profile)
plotter.plot_with_conf(log)



