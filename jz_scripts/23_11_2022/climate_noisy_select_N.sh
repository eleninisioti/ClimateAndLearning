#!/bin/bash
#SBATCH -J fully
#SBATCH -t 30:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/enisioti/climate_log/jz_logs/%j.out
#SBATCH --error=/scratch/enisioti/climate_log/jz_logs/%j.err
module load pytorch-gpu/py3/1.7.1
python simulate.py --project ../projects/niche_construction/23_11_2022/noisy/selection_N_G_niche-construction_N_100_climate_2_noise_0.2 --env_type noisy --num_gens 500 --trial 4 --selection_type N --genome_type niche-construction --num_niches 100 --climate_mean_init 2 