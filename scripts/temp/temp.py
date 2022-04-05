import os

top_dir = "../projects/paper_old/stable/sigma_selection/"
projects = [os.path.join(o) for o in os.listdir(top_dir)]
for project in projects:
    if "plots" not in project:
        trials = [os.path.join(top_dir + project + "/trials", o) for o in os.listdir(top_dir+ project + "/trials")]
        for trial in trials:
            if "trial_3" in trial:
                new_project = "../projects/paper/stable/sigma_selection/" + project
                new_path = new_project + "/trials/trial_3"
                os.rename(trial, new_path)
