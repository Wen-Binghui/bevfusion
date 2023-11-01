from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_head

import mmcv
import matplotlib.pyplot as plt
import pdb
import torch
import copy
import numpy as np
import projection_tool.projection_tool as proj_tool
import projection_tool.visualize_tool as vis_tool
from vectormapnet_debug import model as defined_model

def getdata_from_DC(data, name):
    return data[name].data


config_path = '/mnt/data/codes/bevfusion/configs_addmap.yaml'
configs.load(config_path, recursive=True)
cfg = Config(recursive_eval(configs), filename=config_path)
# datasets = build_dataset(cfg.data.train)
# # print(cfg.data.train)
# # exit()
# for idx in range(3):
#     data_ = datasets[idx]
#     # print(data_['polys'])



CONFIG:dict = defined_model['head_cfg']
# CONFIG.pop('test_cfg') 
model = build_head(CONFIG)
print(model)