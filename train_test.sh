source /mnt/data/anaconda3/etc/profile.d/conda.sh
conda activate torch190
which python
nvidia-smi
echo "start train."
TORCH_DISTRIBUTED_DEBUG=DETAIL torchpack dist-run -np 1 \
python tools/train_configmap.py \
/mnt/data/codes/bevfusion/configs_addmap.yaml \
--model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
--run-dir runs/testrun_addmap


# --load_from pretrained/lidar-only-det.pth \
