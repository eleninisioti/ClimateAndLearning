""" This script is the main interface to the simulation.
"""

import argparse
import os
from life import Life
import pickle
import yaml
import random
import numpy as np
from pathlib import Path


def simulate(args):
    # create project sub-directories
    local=False
    if local:
        project = "../projects/" + args.project
    else:
        project =  args.project
    if not os.path.exists(project  + "/plots"):
        os.makedirs(project + "/plots")

    # save project configuration
    with open(project  + '/config.yml', 'w') as outfile:
        yaml.dump(args, outfile)


        if args.trial:
            # run a single trial with desired trial index
            trials_range = [args.trial]
        else:
            # run multiple trials
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
                log, env_profile, log_niches = life_simul.run()

            except KeyboardInterrupt:
                print("Running aborted. Saving intermediate results.")
                life_simul.logger.final_log()
                log = life_simul.logger.log
                env_profile = life_simul.logger.env_profile
                log_niches = life_simul.logger.log_niches

            with open(trial_dir + '/log.pickle', 'wb') as pfile:
                pickle.dump(log, pfile, protocol=pickle.HIGHEST_PROTOCOL)

            with open(trial_dir + '/env_profile.pickle', 'wb') as pfile:
                pickle.dump(env_profile, pfile, protocol=pickle.HIGHEST_PROTOCOL)

            with open(trial_dir + '/log_niches.pickle', 'wb') as pfile:
                pickle.dump(log_niches, pfile, protocol=pickle.HIGHEST_PROTOCOL)



def init_parser():
    """ Define flags for configuring simulation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--project',
                        help='Name of current project',
                        type=str,
                        default="temp")

    parser.add_argument('--init_num_agents',
                        help="Number of agents",
                        type=int,
                        default=500)

    parser.add_argument('--num_gens',
                        help='Number of generations',
                        type=int,
                        default=1500)

    parser.add_argument('--num_trials',
                        help='Number of independent trials',
                        type=int,
                        default=1)

    parser.add_argument('--trial',
                        help='Current trial',
                        type=int,
                        default=0)

    parser.add_argument('--climate_mean_init',
                        help="Mean of climate",
                        type=float,
                        default=2.0)

    parser.add_argument('--climate_period',
                        help="Period of climate (only used in sin_environment)",
                        type=float,
                        default=10)

    parser.add_argument('--init_SD',
                        help='Initial SD of agents',
                        type=float,
                        default=0.1)

    parser.add_argument('--init_mutate',
                        help='Initial value of Mutation rate',
                        type=float,
                        default=0.005)

    parser.add_argument('--mutate_mutate_rate',
                        help='Mutation rate of mutation rate',
                        type=float,
                        default=0.0005)

    parser.add_argument('--env_type',
                        help='Type of environment. Choose between change, sin and Maslin.',
                        type=str,
                        default="combined")

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
                        type=float,
                        default=1)

    parser.add_argument('--factor_time_steady',
                        help='Scaling factor for time for steady periods.',
                        type=float,
                        default=1)

    parser.add_argument('--var_freq',
                        help='Scaling factor for frequency for abrupt transition.',
                        type=int,
                        default=5)

    parser.add_argument('--var_SD',
                        help='SD of variable transition.',
                        type=float,
                        default=0.2)

    parser.add_argument('--irregular',
                        help='If 1, ....',
                        type=float,
                        default=1)

    parser.add_argument('--selection_type',
                        help="Type of fitness used. Choose between 'FP-Grove', 'limited-capacity' and "
                             "'capacity-fitness'.",
                        type=str,
                        default="capacity-fitness")

    parser.add_argument('--genome_type',
                        help="Type of genome used. Choose 'between 1D', '1D_mutate' and '1D_mutate_fixed'",
                        type=str,
                        default="1D_mutate")

    parser.add_argument('--only_climate',
                        help='If 1, no population is created.',
                        type=int,
                        default=0)


    parser.add_argument('--first_gen',
                        help='First timestep to start simulating life.',
                        type=int,
                        default=0)

    parser.add_argument('--num_niches',
                        help='Number of niches (latitudes).',
                        type=int,
                        default=1)

    parser.add_argument('--extinctions',
                        help='If 1, extinct individuals disappear from the population.',
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

    args = parser.parse_args()
    return args


def run():
    args = init_parser()
    simulate(args)


if __name__ == "__main__":
    args = init_parser()
    simulate(args)
