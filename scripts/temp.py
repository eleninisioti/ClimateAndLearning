import os
import shutil


or_project_dir = "/scratch/enisioti/climate_log/projects/niche_construction/27_12_2022/noisy"
projects = [os.path.join(or_project_dir, o) for o in os.listdir(or_project_dir)]
projects = [el for el in projects if "plots" not in el]

for project in projects:
    trials = [os.path.join(project, o) for o in os.listdir(project + "/trials") if "plots" not in o]
    for file in trials:
        if "plots" not in file:
            new_dir = "/scratch/enisioti/climate_log/projects/niche_construction/30_12_2022/noisy/trials"
            print(file, new_dir)
            #shutil(file, new_dir)

