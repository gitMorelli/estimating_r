#!/bin/bash
#SBATCH --job-name=amorelli_job
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --ntasks=1
#SBATCH --mem=128000
#SBATCH --partition=shortrun

srun python generate_test.py
#srun python generate_test_many_r.py
