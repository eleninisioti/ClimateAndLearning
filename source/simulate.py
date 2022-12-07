""" This script is the main interface to an experiment.
"""

import argparse
import os
from life import Life
import pickle
import yaml
import random
import numpy as np
from pathlib import Path


def simulate():
    """ Run a simulation with desired configuration.

    Parameters
    ----------
    args: dict
        contains desired configuration
    """
    # create project sub-directories
    project = args.project
    if not os.path.exists(project + "/plots"):
        os.makedirs(project + "/plots", exist_ok=True)

    # save project configuration
    with open(project + '/config.yml', 'w') as outfile:
        yaml.dump(args, outfile)

    if args.trial:
        # run a single trial with desired trial index
        trials_range = [args.trial]
    else:
        # or run multiple trials
        trials_range = list(range(args.num_trials))
    for trial in trials_range:
        trial_dir = project + "/trials/trial_" + str(trial)
        if not os.path.exists(trial_dir + "/plots"):
            Path(trial_dir + "/plots").mkdir(parents=True, exist_ok=True)

        # seed simulation with trial index
        random.seed(trial)
        np.random.seed(trial)

        args.trial = trial
        life_simul = Life(args)

        try:
            log, log_niches = life_simul.run()

        except KeyboardInterrupt:
            print("Running aborted. Saving intermediate results.")
            life_simul.logger.final_log()
            log = life_simul.logger.log
            log_niches = life_simul.logger.log_niches

        # save project output
        print("saving at ", trial_dir)
        with open(trial_dir + '/log.pickle', 'wb') as pfile:
            pickle.dump(log, pfile, protocol=pickle.HIGHEST_PROTOCOL)

        print("saving log niches", log_niches.keys())
        with open(trial_dir + '/log_niches.pickle', 'wb') as pfile:
            pickle.dump(log_niches, pfile, protocol=pickle.HIGHEST_PROTOCOL)


def init_parser():
    """ Define input flags for configuring a simulation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--project',
                        help='Name of new project',
                        type=str,
                        default="temp")

    parser.add_argument('--init_num_agents',
                        help="Number of agents at initialization",
                        type=int,
                        default=500)

    parser.add_argument('--num_gens',
                        help='Number of generations.',
                        type=int,
                        default=1500)

    parser.add_argument('--num_trials',
                        help='Number of independent trials',
                        type=int,
                        default=1)

    parser.add_argument('--trial',
                        help='Index of current trial.',
                        type=int,
                        default=0)

    parser.add_argument('--climate_mean_init',
                        help="Mean of climate function at initialization",
                        type=float,
                        default=2.0)

    parser.add_argument('--init_sigma',
                        help='Standard deviation used to initialize genomes',
                        type=float,
                        default=0.5)

    parser.add_argument('--decay_construct',
                        help='Standard deviation used to initialize genomes',
                        type=float,
                        default=0.5)

    parser.add_argument('--init_mutate',
                        help='Initial value of the mutation rate',
                        type=float,
                        default=0.005)

    parser.add_argument('--mutate_mutate_rate',
                        help='Mutation rate of mutation rate (only applicable in genome no-evolv)',
                        type=float,
                        default=0.0005)

    parser.add_argument('--env_type',
                        help='Type of environment. Choose between stable, sin and noisy.',
                        type=str,
                        default="stable")

    parser.add_argument('--capacity',
                        help='Reference capacity (total capacity of all niches)',
                        type=int,
                        default=1000)

    parser.add_argument('--selection_type',
                        help="Type of selection used for reproduction. Choose between N, F and NF",
                        type=str,
                        default="capacity-fitness")

    parser.add_argument('--genome_type',
                        help="Type of genome used. Choose between evolv and no-evolv",
                        type=str,
                        default="evolv")

    parser.add_argument('--only_climate',
                        help='If 1, no population is created (used to quickly simulate climate functions)',
                        type=int,
                        default=0)

    parser.add_argument('--num_niches',
                        help='Number of niches',
                        type=int,
                        default=1)

    parser.add_argument('--period',
                        help='Period of sinusoidal.',
                        type=int,
                        default=0)

    parser.add_argument('--amplitude',
                        help='Amplitude of sinusoidal.',
                        type=float,
                        default=2)

    parser.add_argument('--noise_std',
                        help='Standard deviation of gaussian noise in noisy channel.',
                        type=float,
                        default=0.2)

    parser.add_argument('--time_budget',
                        help='Maximum seconds allowed for a simulation.',
                        type=float,
                        default=60*60*38)

    parser.add_argument('--history_window',
                        help='Window to keep track of ecological history.',
                        type=int,
                        default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = init_parser()
    simulate()
