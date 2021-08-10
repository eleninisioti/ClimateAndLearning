import os
import os.path as op


email = "enisioti@inria.fr"
job_name = "Maslin_presentation"
logs_dir = "/gpfsscratch/rech/imi/utw61ti/logs"
python_path = "python"
code_dir = "/gpfswork/rech/imi/utw61ti/workspace/ClimateAndLearning/source"
slurm_dir = "/gpfswork/rech/imi/utw61ti/workspace/ClimateAndLearning/jz/slurm"
preparatory_commands = "module purge\n module load pytorch-gpu/py3/1.7.1\n"


def run_exp(
    script,
    parameters,
    gpu=False,
    time="20:00:00",
    long_run=False,
    n_tasks=1,
):
  """
  TODO
  """
  slurmjob_path = op.join(slurm_dir, "{}.sh".format(job_name))
  create_slurmjob_cmd = "touch {}".format(slurmjob_path)
  os.system(create_slurmjob_cmd)

  # write arguments into the slurmjob file
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

    # email alerts
    fh.writelines(f"#SBATCH --ntasks=1\n")
    fh.writelines(f"#SBATCH --cpus-per-task {n_tasks}\n")

    fh.writelines("#SBATCH --hint=nomultithread\n")
    fh.writelines("#SBATCH --mail-type=fail\n")
    fh.writelines("#SBATCH --mail-user={}\n".format(email))
    batch_cmd = prep + "srun {} {} {}".format(python_path, script, parameters)
    fh.writelines(batch_cmd)

  os.system("sbatch %s" % slurmjob_path)