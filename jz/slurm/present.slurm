#!/bin/bash
#SBATCH --account=imi@cpu
#SBATCH --job-name=Maslin_present              # name of the job
#SBATCH -N 1                   # number of nodes
#SBATCH --ntasks=1         # number of MPI tasks per node
#SBATCH --time=00:96:00             # maximum execution time requested (HH:MM:S$
#SBATCH --output=/gpfsscratch/rech/imi/utw61ti/logs/Maslin_present_%j.out          # name of output file
#SBATCH --error=/gpfsscratch/rech/imi/utw61ti/logs/Maslin_present_%j.out           # name of error file (here, in common with $
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=fail
#SBATCH --mail-user=eleni.nisioti@inria.fr

python simulate.py --project Maslin/presentation --env_type combined --model hybrid --num_gens 10000 --num_trials 10




