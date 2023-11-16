source /home/andy.wen/miniconda3/etc/profile.d/conda.sh
conda activate torch190
which python
nvidia-smi
echo "start train."
runname="testrun_addmap_bs2_90"

FILE_pth=runs/$runname/latest.pth
if [ -f "$FILE_pth" ]; then
    TORCH_DISTRIBUTED_DEBUG=DETAIL torchpack dist-run -np 2 \
    python tools/train_configpy.py \
    config_py/default_rm_aug.py \
    --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
    --run-dir runs/$runname \
    --resume_from $FILE_pth
else
    TORCH_DISTRIBUTED_DEBUG=DETAIL torchpack dist-run -np 2 \
    python tools/train_configpy.py \
    config_py/default_rm_aug.py \
    --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
    --run-dir runs/$runname
fi