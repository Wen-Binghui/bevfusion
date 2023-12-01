import pickle
import argparse
import mmcv
import numpy as np
from mmdet3d.core.utils import visualize_camera_nosave, visualize_lidar, visualize_map
import os
from mmdet3d.core import LiDARInstance3DBoxes
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from loguru import logger
import os.path as osp
import cv2


def get_sample_from_scenes(scene, nusc: NuScenes):
    all_token = []
    first_sample_token = scene["first_sample_token"]
    # 获取场景的最后一个样本 token
    last_sample_token = scene["last_sample_token"]
    # 通过 NuScenes API 获取样本信息
    current_sample_token = first_sample_token
    while current_sample_token != "":
        all_token.append(current_sample_token)
        # 获取当前样本的信息
        current_sample = nusc.get("sample", current_sample_token)
        # 处理当前样本的逻辑，例如打印信息
        # 获取下一个样本的 token
        current_sample_token = current_sample["next"]
    return all_token


def vis_data(data):
    metas = data["metas"]
    bboxes = LiDARInstance3DBoxes(np.array(data["bboxes"]), box_dim=9)
    pic = {}
    for k, image_path in enumerate(metas["filename"]):
        image = mmcv.imread(image_path)
        pic[k] = visualize_camera_nosave(
            image,
            bboxes=bboxes,
            labels=data["labels"],
            transform=metas["lidar2image"][k],
            classes=data["classes"],
        )
    return pic


def main() -> None:
    infer_res_dir = "/mnt/data/codes/bevfusion/infer_res/test"
    nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuscenes", verbose=True)
    res = os.listdir(infer_res_dir)
    # pkl_path = "/mnt/data/codes/bevfusion/infer_res/test/1531281440299931-cc18fde20db74d30825b0b60ec511b7b.pkl"
    # data = mmcv.load(pkl_path)
    # vis_data(data)
    scene = nusc.scene[2]
    logger.info(scene["description"])
    all_sample = get_sample_from_scenes(scene, nusc)
    # scene_token = scene["token"]
    res_dict = {
        r.split("-")[-1].split(".")[0]: r
        for r in res
        if r.split("-")[-1].split(".")[0] in all_sample
    }
    if len(res_dict) != len(all_sample):
        logger.warning(
            f"{len(res_dict)} of {len(all_sample)} match, not all sample got inference."
        )
    else:
        logger.warning(f"{len(res_dict)} match, all sample got inference.")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("output_video.avi", fourcc, 3, (2400, 900))
    for sample_token in all_sample:
        if sample_token not in res_dict:
            continue
        data = mmcv.load(osp.join(infer_res_dir, res_dict[sample_token]))
        pic = vis_data(data)
        result = np.vstack(
            [np.hstack([pic[2], pic[0], pic[1]]), np.hstack([pic[4], pic[3], pic[5]])]
        )
        result = cv2.resize(result, (2400, 900))
        out.write(result)

    out.release()
    logger.info("Done.")


if __name__ == "__main__":
    main()
