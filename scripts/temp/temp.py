import os

top_dir = "../projects/paper/stable/sigma_selection"
projects = [os.path.join( top_dir, o) for o in os.listdir(top_dir)]
for project in projects:
    if "plots" not in project:
        trials = [os.path.join(project + "/trials", o) for o in os.listdir(project + "/trials")]
        for trial in trials:
            if "trial_0" in trial:
                new_path = project+ "/trials/trial_3"
                os.rename(trial, new_path )
