import pickle
import sys
sys.path.insert(0, "../source")
from plotter import Plotter

current_project = "Maslin/report"
top_dir = "../projects/"

# load data
log = pickle.load(open(top_dir + current_project + '/log.pickle', 'rb'))


plotter = Plotter(current_project)
plotter.plot_trial(log, 0)



