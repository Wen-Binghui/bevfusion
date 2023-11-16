from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS
from torchvision.transforms import ToPILImage


from .base import Base3DFusionModel

__all__ = ["BEVFusionMap"]
from loguru import logger


def format_det(polys, device):
    batch = {
        "class_label": [],
        "batch_idx": [],
        "bbox": [],
    }

    for batch_idx, poly in enumerate(polys):
        keypoint_label = torch.from_numpy(poly["det_label"]).to(device)
        keypoint = torch.from_numpy(poly["keypoint"]).to(device)

        batch["class_label"].append(keypoint_label)
        batch["bbox"].append(keypoint)

    return batch


def format_gen(polys, device):
    line_cls = []
    polylines, polyline_masks, polyline_weights = [], [], []
    bbox, line_cls, line_bs_idx = [], [], []

    for batch_idx, poly in enumerate(polys):
        # convert to cuda tensor
        for k in poly.keys():
            if isinstance(poly[k], np.ndarray):
                poly[k] = torch.from_numpy(poly[k]).to(device)
            else:  # List of ndarray
                poly[k] = [torch.from_numpy(v).to(device) for v in poly[k]]

        line_cls += poly["gen_label"]  # torch Tensor
        line_bs_idx += [batch_idx] * len(poly["gen_label"])

        # condition
        bbox += poly["qkeypoint"]

        # out
        polylines += poly["polylines"]
        polyline_masks += poly["polyline_masks"]
        polyline_weights += poly["polyline_weights"]

    batch = {}
    batch["lines_bs_idx"] = torch.tensor(line_bs_idx, dtype=torch.long, device=device)
    batch["lines_cls"] = torch.tensor(line_cls, dtype=torch.long, device=device)
    batch["bbox_flat"] = torch.stack(bbox, 0)

    # padding
    batch["polylines"] = pad_sequence(polylines, batch_first=True)
    batch["polyline_masks"] = pad_sequence(polyline_masks, batch_first=True)
    batch["polyline_weights"] = pad_sequence(polyline_weights, batch_first=True)

    return batch


@FUSIONMODELS.register_module()
class BEVFusionMap(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:  # True
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1  # real size, 有几团点云
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)  # f: (M, 10, 5)
            coords.append(
                F.pad(c, (1, 0), mode="constant", value=k)
            )  # M * 3 => M * 4 # pad到了最前面
            if n is not None:
                sizes.append(n)  # N: (M, )
        # ASSUME sum(M) = MS
        feats = torch.cat(feats, dim=0)  # (MS, 10, 5)
        coords = torch.cat(coords, dim=0)  # (MS, 4)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)  # (MS, )
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()  # (MS, 5)
        return feats, coords, sizes

    # @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        polys,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        # print('polys, ', type(polys))
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                polys,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    # @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        polys,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in (
            self.encoders
            if self.training
            else list(self.encoders.keys())[::-1]  # 不是训练反向读
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)  # [1, 80, 180, 180]
        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)  # [1, 256, 180, 180]
        else:
            assert len(features) == 1, features
            x = features[0]
        batch_size = x.shape[0]
        x = self.decoder["backbone"](x)  # model: SECOND (output tuple of len 3)
        x = self.decoder["neck"](
            x
        )  # model: SECONDFPN (output list of len 1) [1, 512, 180, 180] #! AUTOFP16 removed
        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                elif type == "vectormap":
                    logger.warning(f'{x[0].device} ENTER vectormap')
                    map_target = {}
                    valid_idx = [i for i in range(len(polys)) if len(polys[i])]
                    polys = [polys[i] for i in valid_idx]
                    logger.warning(f'{x[0].device} polys CONSTRUCTED')
                    if len(valid_idx) != 0:
                        bev_feats = x[0][
                            valid_idx, :, 90 - 25 : 90 + 25, 90 - 50 : 90 + 50
                        ]
                        
                        
                        map_target["det"] = format_det(polys, x[0].device)
                        logger.warning(f'{x[0].device} format_det DONE')
                        map_target["gen"] = format_gen(polys, x[0].device)
                        logger.warning(f'{x[0].device} format_gen DONE')
                        
                        logger.warning(f'{x[0].device} DGHEAD ENTENCE')
                        _, losses = head(
                            map_target,
                            context={
                                "bev_embeddings": bev_feats,
                                "img_shape": [256, 704],
                            },
                            only_det=False,
                        )
                    else:
                        losses = {}
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
