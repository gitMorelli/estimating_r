#!/bin/bash
#SBATCH --job-name=amorelli_job
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --ntasks=1
#SBATCH --mem=128000
#SBATCH --partition=interactive

srun python sync_gen.py

