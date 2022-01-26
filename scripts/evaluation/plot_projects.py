""" This script can be used to produce plots for multiple projects under a common directory.

For each project it plots:
* the evolution of climate and population dynamics
* the SoS, Strengh of Selection plot
"""



import pickle5 as pickle
import sys
import os
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from source.plotter import Plotter


def run(project, climate_noconf):
    top_dir = "../projects/"

    trial_dirs = list(next(os.walk(top_dir + project + "/trials"))[1])
    for trial, trial_dir in enumerate(trial_dirs):
        # load outcome of trial
        try:
            log = pickle.load(open(top_dir + project + "/trials/" + trial_dir + '/log.pickle', 'rb'))
            log_niches = pickle.load(open(top_dir + project + "/trials/" + trial_dir + '/log_niches.pickle', 'rb'))

        except IOError:
            print("No log file for project")
            return 1

        if trial == 0:
            log_df = log
            log_niches_total = [log_niches]
        else:
            log_df = log_df.append(log)
            log_niches_total.append(log_niches)


    # load configuration
    skip_lines = 1
    with open(top_dir + project + "/config.yml") as f:
        for i in range(skip_lines):
            _ = f.readline()
        config = yaml.load(f)
    print(len(log_niches["inhabited_niches"]), config["num_gens"])
    if len(log_niches["inhabited_niches"]) == config["num_gens"]:
        print("No mass extinction for ", project)
    print(config["env_type"])
    if not os.path.exists(top_dir + project + "/trials/" + trial_dir + '/log_updated.pickle'):
        plotter = Plotter(project=project,
                          env_profile=[],
                          climate_noconf=climate_noconf,
                          log=log_df,
                          log_niches=log_niches_total,
                          num_niches=config["num_niches"])


        if config["only_climate"]:
            # if the simulation had no population just plot the climate dynamics
            log = plotter.plot_evolution(include=["climate"])
        else:
            log = plotter.plot_evolution(include=["climate", "mean", "sigma","mutate", "dispersal",  "fitness",
                                              "num_agents",
                                            "extinct", "diversity"])

        plotter.plot_SoS()

        # break log into trials for saving (with computed dispersal)
        for trial, trial_dir in enumerate(trial_dirs):
            log_trial = log.loc[(log['Trial'] == trial)]
            pickle.dump(log_trial, open(top_dir + project + "/trials/" + trial_dir + '/log_updated.pickle', 'wb'))


if __name__ == "__main__":
    top_dir = sys.argv[1] # choose the top directory containing the projects you want to plot

    sub_dirs = [os.path.join(top_dir, o) for o in os.listdir("../projects/" + top_dir)]
    for idx, sub_dir in enumerate(sub_dirs):
        print(sub_dir)
        projects = [os.path.join(sub_dir,o) for o in os.listdir("../projects/" + sub_dir)]
        for p in projects:
            if "plots" not in p:
                print(p)
                run(project=p, climate_noconf=1)




