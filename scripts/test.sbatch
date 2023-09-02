#!/bin/bash
#SBATCH --job-name="nuScenes_ext"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1,VRAM:20G
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user="wwenbinghui@126.com"
#SBATCH --output=/home/wiss/lhao/binghui_DONTDELETE_ME/thesis/bevfusion/slurm_log/slurm-%j-output.out
#SBATCH --error=/home/wiss/lhao/binghui_DONTDELETE_ME/thesis/bevfusion/slurm_log/slurm-%j-error.out
#SBATCH --partition=NORMAL
#SBATCH --comment="andy"

# export GROUP_NAME_ANDY=bev_depth_lss_r50_256x704_128x128_24e_2key_testtest
conda activate bevfusion-torch190
ml load cuda/11.1
cd /usr/stud/wenb/storage/codes/bevfusion
python setup.py develop


# /usr/wiss/lhao/anaconda3/envs/bevfusion-torch190/bin/torchpack dist-run -hostfile scripts/hostfile -np 2 tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox