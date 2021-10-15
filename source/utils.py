import numpy as np

def consol_trials(project):
  top_dir = "../projects/"
  log_df = pd.DataFrame(columns=["Generation", "Trial", "Climate",
                                 "Fitness", "Mean", "SD", "Total_Diversity",
                                 "Specialists_Extinct", "Specialists_Number",
                                 "Specialists_Diversity", "Specialists_Diversity_Mean",
                                 "Specialists_Diversity_SD", "Generalists_Extinct",
                                 "Generalists_Number", "Generalists_Diversity",
                                 "Generalists_Diversity_Mean", "Generalists_Diversity_SD"],
                        dtype=np.float)

  _, env_profile = pickle.load(open(top_dir + project + '/log_total.pickle', 'rb'))
  for trial in range(0, trials):

    if os.path.isfile(top_dir + project + '/trials/trial_' + str(trial)
                                          + '/log.pickle'):

      log = pickle.load(open(top_dir + project + '/trials/trial_' + str(trial)
                                          + '/log.pickle', 'rb'))
      log_df = log_df.append(log)