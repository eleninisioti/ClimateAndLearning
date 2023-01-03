import os
import shutil


or_project_dir = "/scratch/enisioti/climate_log/projects/niche_construction/27_12_2022/noisy"
projects = [os.path.join(or_project_dir, o) for o in os.listdir(or_project_dir)]
projects = [el for el in projects if "plots" not in el]

project_cut = [o for o in os.listdir(or_project_dir)]
for project_idx, project in enumerate(projects):
    trials = [os.path.join(project + "/trials", o) for o in os.listdir(project + "/trials") if "plots" not in o]
    trials_cut = [o for o in os.listdir(project + "/trials") if "plots" not in o]
    for trial_idx, file in enumerate(trials):
        new_dir = "/scratch/enisioti/climate_log/projects/niche_construction/30_12_2022/noisy/" + project_cut[project_idx] + "/trials_" + str(trials_cut[trial_idx])
        print(file, new_dir)

