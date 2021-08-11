""" Main interface to project
"""

import argparse
import os
from life import Life
from plotter import Plotter
import pickle
import yaml
import pandas as pd
import random
import numpy as np
from pathlib import Path

def main(args):
  # create project sub-directories
  if not os.path.exists("../projects/" + args.project + "/plots"):
    os.makedirs("../projects/" + args.project + "/plots")


  # save projecct configuration
  with open("../projects/" + args.project + '/config.yml', 'w') as outfile:
    yaml.dump(args, outfile)

  life_simul = Life(args)
  log_df = pd.DataFrame(columns=["Generation", "Trial", "Climate",
                                 "Fitness", "Mean", "SD", "Total_Diversity",
                                 "Specialists_Extinct", "Specialists_Number",
                                 "Specialists_Diversity", "Specialists_Diversity_Mean",
                                 "Specialists_Diversity_SD", "Generalists_Extinct",
                                 "Generalists_Number", "Generalists_Diversity",
                                 "Generalists_Diversity_Mean","Generalists_Diversity_SD"],
                        dtype=np.float)
  try:
    if args.trial:
      trials_range = [args.trial]
    else:
      trials_range = list(range(args.num_trials))
    for trial in trials_range:
      if not os.path.exists("../projects/" + args.project + "/trials/trial_" + str(trial) + "/plots"):
        Path("../projects/" + args.project + "/trials/trial_" + str(trial) + "/plots").mkdir(parents=True, exist_ok=True)

      print(trial)
      random.seed(trial)
      np.random.seed(trial)
      log = life_simul.run()
      env_profile = log["env_profile"]
      for step in range(args.num_gens+1):
        trial_log = {'Generation': [step], 'Trial': [trial], "Climate": [log["climate_values"][step]],
                     'Fitness': [log["running_fitness"][step]],
                     "Mean": [log["running_mean"][step]],
                     "SD": [log["running_SD"][step]],
                     "Total_Diversity": [log["total_diversity"][step]],
                     "Specialists_Extinct": [log["specialists"]["extinctions"][step]],
                     "Specialists_Number": [log["specialists"]["number"][step]],
                     "Specialists_Diversity": [log["specialists"]["diversity"][step]],
                     "Specialists_Diversity_Mean": [log["specialists"]["diversity_mean"][step]],
                     "Specialists_Diversity_SD": [log["specialists"]["diversity_std"][step]],
                     "Generalists_Extinct": [log["generalists"]["extinctions"][step]],
                     "Generalists_Number": [log["generalists"]["number"][step]],
                     "Generalists_Diversity": [log["generalists"]["diversity"][step]],
                     "Generalists_Diversity_Mean": [log["generalists"]["diversity_mean"][step]],
                     "Generalists_Diversity_SD": [log["generalists"]["diversity_std"][step]]}
        log_df = log_df.append(pd.DataFrame.from_dict(trial_log))

        with open('../projects/' + args.project + '/log_total_part_' + str(trial) + '.pickle', 'wb') as pfile:
          pickle.dump([log_df, env_profile], pfile)

      with open('../projects/' + args.project + '/trials/trial_' + str(trial) + '/log.pickle', 'wb') as pfile:
        pickle.dump(log, pfile)

      plotter = Plotter(args.project, env_profile)
      plotter.plot_trial(log, trial)

    with open('../projects/' + args.project + '/log_total.pickle', 'wb') as pfile:
      pickle.dump([log_df,env_profile], pfile)
    plotter.plot_with_conf(log_df)

  except KeyboardInterrupt:
    print("Running aborted. Saving intermediate results.")
    log = life_simul.log
    with open('../projects/' + args.project + '/log.pickle', 'wb') as pfile:
      pickle.dump(log, pfile)

    plotter = Plotter(args.project)
    plotter.plot_project(log)
    plotter.plot_species(log)
    plotter.plot_diversity(log)





if __name__== "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('--project',
                      help='Name of current project',
                      type=str,
                      default="temp")

  parser.add_argument('--num_agents',
                      help="Number of agents",
                      type=int,
                      default=1000)

  parser.add_argument('--num_gens',
                      help='Name of generations',
                      type=int,
                      default=1500)

  parser.add_argument('--num_trials',
                      help='Name of generations',
                      type=int,
                      default=50)

  parser.add_argument('--trial',
                      help='Current trial',
                      type=int,
                      default=0)

  parser.add_argument('--climate_mean_init',
                      help="Mean of climate",
                      type=float,
                      default=1.0)

  parser.add_argument('--climate_period',
                      help="Mean of climate",
                      type=float,
                      default=10)

  parser.add_argument('--init_SD',
                      help='Initial SD of agents',
                      type=float,
                      default=0.1)

  parser.add_argument('--mutate',
                      help='Mutation rate',
                      type=float,
                      default=0.005)

  parser.add_argument('--env_type',
                      help='Type of environment. Choose between change, sin and combined',
                      type=str,
                      default="change")

  parser.add_argument('--model',
                      help='Model for evolution. Choose between A and B',
                      type=str,
                      default="A")
  parser.add_argument('--capacity',
                      help='Capacity of a niche',
                      type=int,
                      default=1000)

  parser.add_argument('--factor_time_abrupt',
                      help='Scaling factor for time for abrupt transition.',
                      type=int,
                      default=1)

  parser.add_argument('--factor_time_variable',
                      help='Scaling factor for time for variable period.',
                      type=int,
                      default=1)
  parser.add_argument('--var_freq',
                      help='Scaling factor for frequency for abrupt transition.',
                      type=int,
                      default=5)

  args = parser.parse_args()
  main(args)