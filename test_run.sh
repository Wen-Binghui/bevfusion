source /mnt/data/anaconda3/etc/profile.d/conda.sh
conda activate torch190
which python
nvidia-smi
echo "start test."
# torchpack dist-run -np 1 python tools/test_uni.py \
#     /mnt/data/codes/bevfusion/config_py/default_ori.py \
#     pretrained/bevfusion-det.pth --eval bbox --out /mnt/data/codes/bevfusion/result/baseline_od_skip.pkl \
torchpack dist-run -np 1 python tools/test_uni.py \
    /mnt/data/codes/bevfusion/config_py/default_rm_aug_tum_ori.py \
    pretrained/epoch_6_bk.pth --eval bbox --out /mnt/data/codes/bevfusion/result/ori_ep3_od.pkl \

