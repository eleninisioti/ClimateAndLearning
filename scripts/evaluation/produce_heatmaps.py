import pickle
import sys
import os

sys.path.insert(0, "../scripts")
sys.path.insert(0, os.getcwd())

from plotter import Plotter

if __name__ == "__main__":
    top_dir = sys.argv[1]
    plotter = Plotter(project=top_dir, env_profile={}, climate_noconf=False)
    x_variables = ["factor_time_abrupt", "num_niches"]
    y_variables = ["SD", "R"]
    plotter.plot_heatmap(top_dir, x_variables=x_variables, y_variables=y_variables)
