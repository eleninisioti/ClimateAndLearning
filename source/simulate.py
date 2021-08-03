""" Main interface to project
"""

import argparse
import os
from life import Life
from plotter import Plotter
import pickle
import yaml

def main(args):
  # create project directory
  if not os.path.exists("../projects/" + args.project + "/plots/generations/"):
    os.makedirs("../projects/" + args.project + "/plots/generations/")

  # save projecct configuration
  with open("../projects/" + args.project + '/config.yml', 'w') as outfile:
    yaml.dump(args, outfile)

  life_simul = Life(args)
  try:
    log = life_simul.run()
    with open('../projects/' + args.project + '/log.pickle', 'wb') as pfile:
      pickle.dump(log, pfile)

    plotter = Plotter(args.project)
    plotter.plot_project(log)
    plotter.plot_species(log)
    plotter.plot_diversity(log)

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

  parser.add_argument('--scale_time',
                      help='Scaling factor for time.',
                      type=int,
                      default=1)

  args = parser.parse_args()
  main(args)