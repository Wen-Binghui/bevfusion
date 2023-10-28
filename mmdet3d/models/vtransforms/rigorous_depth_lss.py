from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn
import time
from mmdet3d.models.builder import VTRANSFORMS
import projection_tool.projection_tool as proj_tool
from .base import BaseTransform

__all__ = ["RigorousDepthLSSTransform", "RigorousDepthLSSTransform_v1"]


@VTRANSFORMS.register_module()
class RigorousDepthLSSTransform(BaseTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

        self.depth_downsample_ratio = 8.0

    @force_fp32()
    def get_cam_feats(self, x, d, depth_dist):
        '''
        return: size (B, N, self.D, fH, fW, self.C)
        '''
        B, N, C, fH, fW = x.shape
        d = d.view(B * N, *d.shape[2:]) # ori d torch.Size([1, 6, 1, 256, 704])
        x = x.view(B * N, C, fH, fW)
        depth_dist = depth_dist.view(B * N, *depth_dist.shape[2:]).permute(0, 3, 1, 2) # torch.Size([6, 32, 88, 118])
        d = self.dtransform(d) # d dtransformed torch.Size([6, 64, 32, 88]) # 八倍下采样
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x) # x torch.Size([6, 198, 32, 88])
        depth = x[:, : self.D].softmax(dim=1) # depth torch.Size([6, 118, 32, 88])
        #! add with known distribution
        depth_mix =  depth + depth_dist
        depth_mix /= torch.sum(depth_mix, dim = -1).unsqueeze(-1)
        x = depth_mix.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2) # of size (B, N, self.D, fH, fW, self.C)
        return x

    # def forward(self, *args, **kwargs):
    @force_fp32()
    def forward(
        self,
        img,
        points,
        sensor2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        # print(img.shape, self.image_size, self.feature_size)

        batch_size = len(points)
        #! one depth value for one pixel 
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(
            points[0].device
        )
        width, height = self.image_size[1], self.image_size[0]
        # print('width, height: ', width, height) # 704， 256

        n_depth = int((self.dbound[1] - self.dbound[0])/self.dbound[2])
        depth_prob_map = torch.zeros((batch_size,
                                      img.shape[1], 
                                      height//int(self.depth_downsample_ratio),
                                      width//int(self.depth_downsample_ratio),
                                      n_depth
                                      )).to(points[0].device)
        
        for b in range(batch_size): # extract one instance from batch
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug => real coord in physic lidar coord sys 
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            ) 
            # lidar2image => camera coord sys 
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :] # get depth only
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :] # x,y / z

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] > 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] > 0)
            )
            # TODO 将深度每个pixel一个值改为 多个 
            for c in range(on_img.shape[0]): # each camera
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]

                img_coord = torch.hstack((cur_coords[c, on_img[c], :2].squeeze(0), dist[c, on_img[c]].unsqueeze(1)))  # (n, 3)
                img_coord = img_coord[(img_coord[:,2]<=self.dbound[1]) & (img_coord[:,2]>=self.dbound[0]), :]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

                depth_prob_map[b, c, :, :, :] = proj_tool.create_depth_scatter_matrix_v2(width, 
                                                                        height, 
                                                                        img_coord, 
                                                                        self.dbound, 
                                                                        depth.device,
                                                                        downsample_ratio=self.depth_downsample_ratio) # size: (height, width, depth_num) #* torch.Size([32, 88, 118])
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )
        # depth of shape (b, picture_num, 1, height, width)
        x = self.get_cam_feats(img, depth, depth_prob_map)
        x = self.bev_pool(geom, x)

        # inhri
        x = self.downsample(x)
        return x
        

@VTRANSFORMS.register_module()
class RigorousDepthLSSTransform_v1(RigorousDepthLSSTransform):
    """
    Change from RigorousDepthLSSTransform:
        Add a layer to filter depth map (such as a smooth filter) : {depthmap_fuser}

    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
            downsample=downsample
        )
        # n_depth = int((dbound[1] - dbound[0])/dbound[2])
        mid_depthmap_channel = 8
        conv3d_kernel_size = 3
        self.depthmap_fuser = nn.Sequential(
            nn.Conv3d(1, mid_depthmap_channel, kernel_size=conv3d_kernel_size, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv3d(mid_depthmap_channel, 1, kernel_size=conv3d_kernel_size, padding=1, bias=False),
        )
        for name, param in self.depthmap_fuser.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=1, std=0.01) # 正态初始化

    @force_fp32()
    def forward(
        self,
        img,
        points,
        sensor2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        batch_size = len(points)
        #! one depth value for one pixel 
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(
            points[0].device
        )
        width, height = self.image_size[1], self.image_size[0]

        n_depth = int((self.dbound[1] - self.dbound[0])/self.dbound[2])
        depth_prob_map = torch.zeros((batch_size,
                                      img.shape[1], 
                                      height//int(self.depth_downsample_ratio),
                                      width//int(self.depth_downsample_ratio),
                                      n_depth
                                      )).to(points[0].device)
        with torch.autograd.set_detect_anomaly(True):
            for b in range(batch_size): # extract one instance from batch
                cur_coords = points[b][:, :3]
                cur_img_aug_matrix = img_aug_matrix[b]
                cur_lidar_aug_matrix = lidar_aug_matrix[b]
                cur_lidar2image = lidar2image[b]

                # inverse aug => real coord in physic lidar coord sys 
                cur_coords -= cur_lidar_aug_matrix[:3, 3]
                cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                    cur_coords.transpose(1, 0)
                ) 
                # lidar2image => camera coord sys 
                cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
                cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
                # get 2d coords
                dist = cur_coords[:, 2, :] # get depth only
                cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
                cur_coords[:, :2, :] /= cur_coords[:, 2:3, :] # x,y / z

                # imgaug
                cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
                cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
                cur_coords = cur_coords[:, :2, :].transpose(1, 2)

                # normalize coords for grid sample
                cur_coords = cur_coords[..., [1, 0]]

                on_img = (
                    (cur_coords[..., 0] < self.image_size[0])
                    & (cur_coords[..., 0] > 0)
                    & (cur_coords[..., 1] < self.image_size[1])
                    & (cur_coords[..., 1] > 0)
                )
                # TODO 将深度每个pixel一个值改为 多个 
                for c in range(on_img.shape[0]): # each camera
                    masked_coords = cur_coords[c, on_img[c]].long()
                    masked_dist = dist[c, on_img[c]]

                    img_coord = torch.hstack((cur_coords[c, on_img[c], :2].squeeze(0), dist[c, on_img[c]].unsqueeze(1)))  # (n, 3)
                    img_coord = img_coord[(img_coord[:,2]<=self.dbound[1]) & (img_coord[:,2]>=self.dbound[0]), :]
                    depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

                    depth_prob_map[b, c, :, :, :] = proj_tool.create_depth_scatter_matrix_v2(width, 
                                                                            height, 
                                                                            img_coord, 
                                                                            self.dbound, 
                                                                            depth.device,
                                                                            downsample_ratio=self.depth_downsample_ratio) # size: (height, width, depth_num) #* torch.Size([32, 88, 118])
            
            B, C, height_depth_map, width_depth_map, num_depth_partition = depth_prob_map.shape
            depth_prob_map = depth_prob_map.view(B*C, 1, height_depth_map, width_depth_map, num_depth_partition)
            depth_prob_map: torch.Tensor = self.depthmap_fuser(depth_prob_map)
            depth_prob_map = depth_prob_map.view(B, C, height_depth_map, width_depth_map, num_depth_partition)
            depth_prob_map = torch.maximum(torch.tensor(0.0).to(depth_prob_map.device), depth_prob_map)
            extra_rots = lidar_aug_matrix[..., :3, :3]
            extra_trans = lidar_aug_matrix[..., :3, 3]
            geom = self.get_geometry(
                camera2lidar_rots,
                camera2lidar_trans,
                intrins,
                post_rots,
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            )
            # depth of shape (b, picture_num, 1, height, width)
            x = self.get_cam_feats(img, depth, depth_prob_map)
            x = self.bev_pool(geom, x)

            # inhri
            x = self.downsample(x)
        return x
