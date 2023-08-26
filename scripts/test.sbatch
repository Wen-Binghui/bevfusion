#!/bin/bash
#SBATCH --job-name="nuScenes_ext"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2,VRAM:20G
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user="wwenbinghui@126.com"
#SBATCH --output=/home/wiss/lhao/binghui_DONTDELETE_ME/thesis/bevfusion/slurm_log/slurm-%j-output.out
#SBATCH --error=/home/wiss/lhao/binghui_DONTDELETE_ME/thesis/bevfusion/slurm_log/slurm-%j-error.out
#SBATCH --partition=NORMAL
#SBATCH --comment="andy"
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
nvidia-smi
export NCCL_DEBUG=INFO

export GROUP_NAME_ANDY=bev_depth_lss_r50_256x704_128x128_24e_2key_testtest
/usr/wiss/lhao/anaconda3/envs/bevfusion-torch190/bin/torchpack dist-run -hostfile scripts/hostfile -np 2 tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml pretrained/bevfusion-det.pth --eval bbox