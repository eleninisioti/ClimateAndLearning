""" This script contains functions useful for running experiments on the Jean-Zay cluster.
"""

import os
from random import random

# ----- jz user configuration -----
account = "utw61ti"
email = "enisioti@inria.fr"
job_name = "ClimateAndLearning"
prep = "module purge\n module load pytorch-gpu/py3/1.7.1\n"
python_path = "python"
logs_dir = f"/gpfsscratch/rech/imi/{account}/ClimateAndLearning_log/jz_logs"
slurm_dir = f"/gpfsscratch/rech/imi/{account}/ClimateAndLearning_log/slurm"
# ---------------------------------

def run_exp(job_name, script, parameters, gpu=False, time="20:00:00", long_run=False, n_tasks=1):
    """ Submit jz jobs.

    Parameters
    ----------
    job_name: str
        job id name
    script: str
        name of project's interface file (should be simulate.py)

    parameters: str
        configuration of the experiment

    gpu: bool
        if True, use gpus

    time: str
        maximum duration of job

    long_run: bool
        if False, job can be at most 20 hours

    n_tasks: int
        configure cpus
    """
    # ----- prepare submission script in slurmjob file ------
    slurmjob_path = os.path.join(slurm_dir + "/" + job_name + ".sh")
    create_slurmjob_cmd = "touch {}".format(slurmjob_path)
    os.system(create_slurmjob_cmd)
    slurmjob_path = os.path.join(slurm_dir, "{}.sh".format(job_name))
    create_slurmjob_cmd = "touch {}".format(slurmjob_path)
    os.system(create_slurmjob_cmd)
    with open(slurmjob_path, "w") as fh:
        fh.writelines("#!/bin/sh\n")
        if gpu:
            fh.writelines("#SBATCH --account=imi@gpu\n")
            fh.writelines("#SBATCH --gres=gpu:1\n")
            if long_run:
                fh.writelines("#SBATCH --qos=qos_gpu-t4\n")
        else:
            fh.writelines("#SBATCH --account=imi@cpu\n")
            fh.writelines("#SBATCH -N 1\n")
            if long_run:
                fh.writelines("#SBATCH --qos=qos_cpu-t4\n")
        fh.writelines("#SBATCH --job-name={}\n".format(job_name))
        fh.writelines("#SBATCH -o {}/{}_%j.out\n".format(logs_dir, job_name))
        fh.writelines("#SBATCH -e {}/{}_%j.err\n".format(logs_dir, job_name))
        fh.writelines(f"#SBATCH --time={time}\n")
        fh.writelines(f"#SBATCH --ntasks=1\n")
        fh.writelines(f"#SBATCH --cpus-per-task {n_tasks}\n")
        fh.writelines("#SBATCH --hint=nomultithread\n")
        fh.writelines("#SBATCH --mail-type=fail\n")
        fh.writelines("#SBATCH --mail-user={}\n".format(email))
        batch_cmd = prep + "srun {} {} {}".format(python_path, script, parameters)
        fh.writelines(batch_cmd)
    # ----------------------------------------------------------------------
    # submit job
    os.system("sbatch %s" % slurmjob_path)


def run_batch(experiments, param_names, long_run=False, gpu=True, n_tasks=1):
    """
    makes all possible combinations of parameters within the given parameters, and for each combination calls on
    run_exp from looper.py to launch them with SLURM

    parameters
    ----------
    experiments
    param_names : list of strings
        all parameter names, in same order of apparition as in experiments
    n_seeds : int, default = 1
        number of repetitions of the same experiment to take place (another, more ordered, way to do this is to give
        n different log_paths, avoiding file name repetitions.)
    long_run : bool, default = False
        Wether or not to request a qos4 with long runtime on the jean zay cluster, with 100h
    gpu : bool, default = True
        Whether to request a gpu node, or a cpu node (multiprocessing only works on cpu for now)
    cpu_count: int, default = 1
        number of cpus allocated to the task. More cpus gives access to more memory, and more possible parallel
        processes.
    """
    # process flags
    parameters = ""
    for experiment in experiments:
        for i in range(len(experiment)):
            parameters += f" {param_names[i]}={experiment[i]}"

        script = "simulate.py"

        if long_run:
            time = "100:00:00"
        else:
            time = "18:00:00"
        name = "temp_" + str(random())
        run_exp(job_name=name,
                script=script,
                parameters=parameters,
                gpu=gpu,
                time=time,
                long_run=long_run,
                n_tasks=n_tasks)
