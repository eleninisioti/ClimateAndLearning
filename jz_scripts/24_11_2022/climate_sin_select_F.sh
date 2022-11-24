#!/bin/bash
#SBATCH -J fully
#SBATCH -t 30:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/enisioti/climate_log/jz_logs/%j.out
#SBATCH --error=/scratch/enisioti/climate_log/jz_logs/%j.err
module load pytorch-gpu/py3/1.7.1
python simulate.py --project ../projects//24_11_2022/manim_fig8/S_F_G_evolv_N_100_climate_0.2_T_46_A_0.2 --env_type sin --num_gens 500 --trial 0 --selection_type F --genome_type evolv --num_niches 100 --climate_mean_init 0.2 --amplitude 0.2 --period 46 