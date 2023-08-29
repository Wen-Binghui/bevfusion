#!/bin/bash
#SBATCH --job-name="testmpi_cpu"
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user="wwenbinghui@126.com"
#SBATCH --output=/usr/stud/wenb/storage/codes/bevfusion/slurm_log/slurm-%j-output.out
#SBATCH --error=/usr/stud/wenb/storage/codes/bevfusion/slurm_log/slurm-%j-error.out
#SBATCH --partition=NORMAL
#SBATCH --comment="andy"

mpiexec -n 16 /storage/user/wenb/software/anaconda3/envs/bevfusion/bin/python /usr/stud/wenb/storage/codes/bevfusion/tools/testmpi.py