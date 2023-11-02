from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_head, build_model

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
cfg['model']['heads']['vectormap'] = defined_model['head_cfg']


model = build_model(cfg.model)
print(model)