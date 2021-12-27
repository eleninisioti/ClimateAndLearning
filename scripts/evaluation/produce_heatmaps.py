import pickle
import sys
import os

sys.path.insert(0, "../scripts")
sys.path.insert(0, os.getcwd())

from plotter import Plotter

if __name__ == "__main__":

    top_dir = "Maslin/1D_mutate/parametric_variabl"
    plotter = Plotter(project=top_dir, env_profile={}, climate_noconf=False)
    trials =1
    x_variables = ["factor_time_variable", "factor_time_abrupt"]
    y_variables = ["SD", "R"]
    plotter.plot_heatmap(top_dir, trials=1, x_variables=x_variables, y_variables=y_variables)
