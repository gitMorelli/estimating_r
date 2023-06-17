#!/bin/bash
#SBATCH --job-name=amorelli_job
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --ntasks=1
#SBATCH --mem=128000
#SBATCH --partition=longrun 

srun python likelihood_BB_from_map.py

