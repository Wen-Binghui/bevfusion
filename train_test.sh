source /mnt/data/anaconda3/etc/profile.d/conda.sh
conda activate torch190
which python
nvidia-smi
echo "start train."
TORCH_DISTRIBUTED_DEBUG=DETAIL torchpack dist-run -np 1 \
python tools/train_configpy.py \
config_py/default_rm_aug.py \
--model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
--run-dir runs/testrun_addmap_bs1


# --load_from pretrained/lidar-only-det.pth \
