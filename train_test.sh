TORCH_DISTRIBUTED_DEBUG=DETAIL torchpack dist-run -np 1 \
python tools/train_configmap.py \
/mnt/data/codes/bevfusion/configs_addmap.yaml \
--model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
--load_from pretrained/lidar-only-det.pth \
--run-dir runs/testrun_addmap
