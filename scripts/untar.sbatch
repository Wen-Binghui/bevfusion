#!/bin/bash
#SBATCH --job-name="cpu_untar"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user="wwenbinghui@126.com"
#SBATCH --output=/usr/stud/wenb/storage/codes/bevfusion/slurm_log/slurm-%j-output.out
#SBATCH --error=/usr/stud/wenb/storage/codes/bevfusion/slurm_log/slurm-%j-error.out
#SBATCH --partition=NORMAL
#SBATCH --comment="andy"

tar -C /storage/user/wenb/datasets/nuScenes/nuScenes/ -xzvf /storage/user/wenb/datasets/nuScenes/v1.0-test_blobs.tgz
