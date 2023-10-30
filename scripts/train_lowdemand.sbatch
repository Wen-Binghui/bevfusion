#!/bin/bash
#SBATCH --job-name="btL"
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user="wwenbinghui@126.com"
#SBATCH --output=/usr/stud/wenb/storage/codes/bevfusion/slurm_log/slurm-%j-output.out
#SBATCH --error=/usr/stud/wenb/storage/codes/bevfusion/slurm_log/slurm-%j-error.out
#SBATCH --partition=NORMAL
#SBATCH --comment="andy"

echo start
pwd
nvidia-smi
ml load cuda/11.1
export RDMAV_FORK_SAFE=True
cd /usr/stud/wenb/storage/codes/bevfusion
/usr/stud/wenb/storage/software/anaconda3/envs/bevfusion-torch190/bin/torchpack dist-run -np 2 /usr/stud/wenb/storage/software/anaconda3/envs/bevfusion-torch190/bin/python3 tools/train.py \
/usr/stud/wenb/storage/codes/bevfusion/config_rigorous/configs_rig_v1_bs1.yaml \
 --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth 

# conda activate bevfusion-torch190
