source /mnt/data/anaconda3/etc/profile.d/conda.sh
conda activate torch190
which python
nvidia-smi
echo "start test."
torchpack dist-run -np 1 python tools/visualize.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --mode pred  --checkpoint pretrained/bevfusion-det.pth --eval bbox \
    --split val --out-dir viz --out-dir-data infer_res/test

# torchpack dist-run -np 1 python tools/visualize.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
#     --mode gt  --checkpoint pretrained/bevfusion-det.pth --eval bbox \
#     --split val --out-dir viz_gt


