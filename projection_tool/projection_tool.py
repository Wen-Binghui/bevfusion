import torch
import numpy as np
from scipy.stats import norm


def probability_in_interval(depth_interval, depth, sigma):
    # 返回给定区间内的累计频率
    prob_each_tick = norm.cdf(depth_interval, depth, sigma)
    prob = np.diff(prob_each_tick)
    return prob


def gaussion_prob_from_multidepth(depth_dist, dbound):
    """ Generate depth probability array from given depth list
        Args:
            depth_dist: list of depth  
            dbound: like [1.0, 60.0, 0.5], min, max, step
        Return:
            prob: depth probability array of shape (depth_interval_ticks, )
    """
    # TODO 检查输入depth 会不会在1-60之外，出现zero-divide
    depth_start, depth_end, depth_step = dbound
    # 先假设 range_list 是 list #* [1.0, 60.0, 0.5]
    sigma = 1.0
    n_cluster = len(depth_dist)
    depth_interval_ticks = np.arange(
        depth_start-depth_step/2, depth_end+depth_step/2+1e-3, depth_step)
    prob = np.zeros(depth_interval_ticks.shape[0] - 1)
    for depth in depth_dist:
        prob += probability_in_interval(depth_interval_ticks, depth, sigma)
    if n_cluster > 0:
        prob = prob/np.sum(prob)
    return prob

def generate_depth_prob_map(height, width, dbound, depth_collection_mat) -> np.ndarray:
    """ Generate depth probability map for given possible depth list
        Args:
            depth_collection_mat: np 
        Return:
            depth_prob_map: matrix of (height, width, number of depth centers)
                type: np.ndarray
    """
    n_depth = int((dbound[1] - dbound[0])/dbound[2] + 1)
    depth_prob_map = np.zeros((height, width, n_depth))
    for ih in range(height):
        for jw in range(width):
            depth_prob_map[ih, jw, :] = gaussion_prob_from_multidepth(
                depth_collection_mat[ih][jw], dbound)
    return depth_prob_map

def coord_float2ind(coor: np.ndarray):
    return np.array(np.round(coor[:, :2] - 0.5, ), dtype=int)


def create_depth_scatter_matrix(width, height, img_coord_float: torch.Tensor, clap=[0, 60]):
    """Create a list matrix represent the depth in each pixel.
        Args:
            img_coord_float: (N, 3) <== height, width, depth.
            clap (list[value]): interval of valid depth.
                default: [0, 60].
        Return:
            mat of size [height] [width] [depth num in this pixel]
    """
    mat = [[[] for i in range(width)] for j in range(height)]  # height * width

    idx = torch.argsort(img_coord_float[:, 2])
    img_coord_float = img_coord_float[idx, :]  # sort by depth
    # print(img_coord_float)
    img_coor = coord_float2ind(img_coord_float)
    for i, coord in enumerate(img_coor):
        if img_coord_float[i, 2] > clap[1] or img_coord_float[i, 2] < clap[0]:
            continue
        mat[coord[0]][coord[1]].append(img_coord_float[i, 2])
    return mat


def create_depth_partition_tensor(width, height, depth_partition=20):
    return torch.zeros((width, height, depth_partition))


if __name__ == "__main__":
    pass
