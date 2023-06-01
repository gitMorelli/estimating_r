#!/bin/bash -l
#SBATCH -p skl_usr_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH -t 0:30:00
#SBATCH --job-name=amorelli_job
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH -A INF23_indark
#SBATCH --export=ALL
#SBATCH --mem=182000

cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#module load python/3.6.4
#module load anaconda/2019.10
module load profile/deeplrn
module load gnu/7.3.0
module load openmpi/4.0.1--gnu--7.3.0
conda activate camb

#srun python helloworld.py
#srun python multi_test.py
srun python foreground_noise_generation.py
#srun python synchrotron_test.py

