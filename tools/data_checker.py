from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval
from mmdet3d.datasets import build_dataset
import mmcv
import matplotlib.pyplot as plt
import pdb
import torch
import copy
import numpy as np
import projection_tool.projection_tool as proj_tool
import projection_tool.visualize_tool as vis_tool


def getdata_from_DC(data, name):
    return data[name].data


config_path = '/usr/stud/wenb/storage/codes/bevfusion/configs_modify/nuscenes/default.yaml'
configs.load(config_path, recursive=True)
cfg = Config(recursive_eval(configs), filename=config_path)
# print(cfg)
# ann_file = 'data/nuscenes/nuscenes_infos_train.pkl'
# data = mmcv.load(ann_file)
# mmcv.dump(data["infos"][35], 'data/sample_train_infos.json', indent = 4)
datasets = build_dataset(cfg.data.train)

# dbound = cfg.model.encoders.camera.vtransform.dbound
dbound = [1.0, 60.0, 0.5]
cmap = plt.get_cmap("jet")

for idx in range(3):
    data_ = datasets[idx]
    img = data_["img"].data.permute(0, 2, 3, 1)
    camera2lidar = getdata_from_DC(data_, 'camera2lidar')
    points:torch.Tensor = getdata_from_DC(data_, 'points')[..., :3]
    img_aug_matrix = getdata_from_DC(data_, 'img_aug_matrix')
    lidar_aug_matrix = getdata_from_DC(data_, 'lidar_aug_matrix')
    lidar2image = getdata_from_DC(data_, 'lidar2image')
    N = img.shape[0]
    image_size = (img.shape[1], img.shape[2]) # (256, 704)
    for i in range(N):
        cur_coords = copy.deepcopy(points) # N * 3
        cur_img_aug_matrix = img_aug_matrix[i, ...] # (4, 4)
        cur_lidar_aug_matrix = lidar_aug_matrix # (4, 4)
        cur_lidar2image = lidar2image[i, ...] # (4, 4)

        cur_coords -= cur_lidar_aug_matrix[:3, 3]
        cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
            cur_coords.transpose(1, 0)
        ) 
        # lidar2image => camera coord sys 
        cur_coords = cur_lidar2image[:3, :3].matmul(cur_coords)
        cur_coords += cur_lidar2image[:3, 3].reshape(3, 1)
        
        # get 2d coords
        dist = cur_coords[2, :] # get depth only
        

        cur_coords[2, :] = torch.clamp(cur_coords[2, :], 1e-5, 1e5)
        cam_coors = copy.deepcopy(cur_coords)
        cur_coords[:2, :] /= cur_coords[2:3, :] # x,y / z

        # imgaug
        cur_coords = cur_img_aug_matrix[:3, :3].matmul(cur_coords)
        cur_coords += cur_img_aug_matrix[:3, 3].reshape(3, 1)
        cur_coords = cur_coords[:2, :].transpose(0, 1) # (N ,3)

        # normalize coords for grid sample
        cur_coords = cur_coords[..., [1, 0]]

        on_img = (
            (cur_coords[..., 0] < image_size[0])
            & (cur_coords[..., 0] >= 0)
            & (cur_coords[..., 1] < image_size[1])
            & (cur_coords[..., 1] >= 0)
        )
        # ptc_np:np.ndarray = copy.deepcopy(cam_coors.transpose(0, 1)[on_img, :].numpy())
        # ptc_np = ptc_np.astype(np.float64)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(ptc_np)
        # print(len(pcd.points))
        # np.savetxt(f"vis/fig_{idx}_{i}.txt", cam_coors.transpose(0, 1))
        # o3d.io.write_point_cloud(f"vis/fig_{idx}_{i}.pcd", pcd)
        img_coord = torch.hstack((cur_coords[on_img, :2], dist[on_img].unsqueeze(1)))  # (n, 3)
        img_coord = img_coord[img_coord[:,2]<=60, :]# TODO > 1?, NOT HARD CODING
        print('img_coord', img_coord.shape)
        width, height = image_size[1], image_size[0]
        print(width, height)
        
        depth_collection_mat = proj_tool.create_depth_scatter_matrix(width, 
                                                                     height, 
                                                                     img_coord)
        depth_prob_map = proj_tool.generate_depth_prob_map(height, 
                                                           width,
                                                           dbound, 
                                                           depth_collection_mat)
        print('calcu done')        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 画图比较费时，计算耗时还好
        vis_tool.bar3d_for_depthmap(height, 
                       width, 
                       depth_prob_map, 
                       dbound, 
                       ax, 
                       cmap, 
                       destimg_path = f"vis/fig_{idx}_{i}_3d.png")
        print('img gene done')   
        fig = plt.figure()
        plt.imshow(img[i, ...].numpy())
        plt.scatter(cur_coords[on_img, 1], cur_coords[on_img, 0], c = dist[on_img], s = 0.6, cmap='viridis',alpha = 0.7)
        plt.savefig(f"vis/fig_{idx}_{i}.png", dpi=400)
        plt.close('all')

# pdb.set_trace()
