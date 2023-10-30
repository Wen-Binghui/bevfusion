#!/bin/bash
#SBATCH --job-name="bevtfuse"
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4,VRAM:24G
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
/usr/stud/wenb/storage/software/anaconda3/envs/bevfusion-torch190/bin/torchpack dist-run -np 4 python tools/train.py \
/usr/stud/wenb/storage/codes/bevfusion/config_rigorous/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
 --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth 

# --resume_from /usr/stud/wenb/storage/codes/bevfusion/runs/run-3e6a424f-e9be4fd8/latest.pth

# conda activate bevfusion-torch190

# /usr/stud/wenb/storage/software/anaconda3/envs/bevfusion-torch190/bin/torchpack dist-run -np 4 python tools/test.py \
# /usr/stud/wenb/storage/codes/bevfusion/config_rigorous/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
# /usr/stud/wenb/storage/codes/bevfusion/runs/run-3e6a424f-e9be4fd8/latest.pth \
#  --eval bbox \
#  --out ../runs/run-3e6a424f-e9be4fd8/result.pkl
