import pickle
import sys
sys.path.insert(0, "../source")
from plotter import Plotter
import os
import numpy as np
import pandas as pd
import click

@click.command()
@click.option("--project", type=str, default="") # how many generations will training take?
@click.option("--trials", type=int, default=50)
@click.option("--climate_noconf", type=int, default=0)
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
  for trial in range(0, trials):

    if os.path.isfile(top_dir + project + '/log_total_part_' + str(trial)
                                        + '.pickle'):

      log, env_profile = pickle.load(open(top_dir + project + '/log_total_part_' + str(trial)
                                          + '.pickle', 'rb'))
      log_df = log_df.append(log)

  plotter = Plotter(project, env_profile, climate_noconf=climate_noconf)
  #plotter.plot_with_conf(log_df, [1,0,0,1], 2)

  plotter.plot_evolution_with_conf(log_df, [1, 0, 1, 0])
  plotter.plot_evolution_with_conf(log_df, [1, 1, 0, 1])
  plotter.plot_species_with_conf(log_df, [1, 1, 0, 0])
  plotter.plot_species_with_conf(log_df, [1, 0, 1, 0])
  plotter.plot_species_with_conf(log_df, [1, 0, 0, 1])

if __name__ == "__main__":
  run()








