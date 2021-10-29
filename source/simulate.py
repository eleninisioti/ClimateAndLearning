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


def simulate(args):
    # create project sub-directories
    if not os.path.exists("../projects/" + args.project + "/plots"):
        os.makedirs("../projects/" + args.project + "/plots")

    # save projecct configuration
    with open("../projects/" + args.project + '/config.yml', 'w') as outfile:
        yaml.dump(args, outfile)

    life_simul = Life(args)

    try:
        if args.trial:
            trials_range = [args.trial]
        else:
            trials_range = list(range(args.num_trials))
        for trial in trials_range:
            if not os.path.exists("../projects/" + args.project + "/trials/trial_" + str(trial) + "/plots"):
                Path("../projects/" + args.project + "/trials/trial_" + str(trial) + "/plots").mkdir(parents=True,
                                                                                                     exist_ok=True)

            print(trial)
            random.seed(trial)
            np.random.seed(trial)
            log = life_simul.run()
            for step in range(len(log["climate_values"])):
                trial_log = {'Generation': [step], 'Trial': [trial]}
                for key in log.keys():
                    if len(log[key]) and key!= "env_profile":
                        trial_log[key] = log[key][step]

                if step:
                    log_df = log_df.append(pd.DataFrame.from_dict(trial_log))
                else:
                    log_df = pd.DataFrame.from_dict(trial_log)

            # with open('../projects/' + args.project + '/log_total_part_' + str(trial) + '.pickle', 'wb') as pfile:
            #   pickle.dump([log_df, env_profile], pfile)

            with open('../projects/' + args.project + '/trials/trial_' + str(trial) + '/log.pickle', 'wb') as pfile:
                pickle.dump(log_df, pfile)

        with open('../projects/' + args.project + '/log_total.pickle', 'wb') as pfile:
            pickle.dump([log_df, log["env_profile"]], pfile)

    except KeyboardInterrupt:
        print("Running aborted. Saving intermediate results.")
        log = life_simul.log
        with open('../projects/' + args.project + '/log.pickle', 'wb') as pfile:
            pickle.dump(log, pfile)

        plotter = Plotter(args.project)
        plotter.plot_project(log)
        plotter.plot_species(log)
        plotter.plot_diversity(log)


def init_parser():
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
                        default=1)

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

    parser.add_argument('--init_mutate',
                        help='Initial value of Mutation rate',
                        type=float,
                        default=0.005)

    parser.add_argument('--mutate_rate',
                        help='Mutation rate of mutation rate',
                        type=float,
                        default=0.0005)

    parser.add_argument('--env_type',
                        help='Type of environment. Choose between change, sin and combined',
                        type=str,
                        default="combined")

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
                        help='Scaling factor for SD of abrupt transition.',
                        type=float,
                        default=0.2)

    parser.add_argument('--irregular',
                        help='Scaling factor for SD of abrupt transition.',
                        type=float,
                        default=1)

    parser.add_argument('--survival_type',
                        help='Type of fitness used. Choose between "MC" for Minimum Criterion and "FP" for fitness '
                             'proportionate.',
                        type=str,
                        default="FP")

    parser.add_argument('--genome_type',
                        help='Type of genome used. Choose between 1D and 1D_mutate',
                        type=str,
                        default="1D_mutate")

    parser.add_argument('--only_climate',
                        help='If use, no population is created.',
                        action="store_true")

    parser.add_argument('--first_gen',
                        help='First timestep to start simulating life',
                        type=int,
                        default=0)


    args = parser.parse_args()
    return args


def run():
    args = init_parser()
    simulate(args)


if __name__ == "__main__":
    args = init_parser()
    simulate(args)
