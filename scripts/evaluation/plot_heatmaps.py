""" This script can be used for evaluating a batch of experiments.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from source.plotter import Plotter

if __name__ == "__main__":
    top_dir = sys.argv[1]
    # produce heatmaps for all subdirectories
    projects = [os.path.join(top_dir, o) for o in os.listdir("../projects/" + top_dir)]
    x_variables_sin = ["period", "amplitude"]
    x_variables_stable = ["climate_mean_init"]
    y_variables = ["SD", "R", "diversity", "extinctions"]
    plotter = Plotter(project=top_dir, env_profile={}, climate_noconf=False)

    for idx, p in enumerate(projects):
        plotter.plot_heatmap(top_dir=p,
                             x_variables_sin=x_variables_sin,
                             x_variables_stable=x_variables_stable,
                             y_variables=y_variables)

