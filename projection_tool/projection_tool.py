import torch
import numpy as np
from scipy.stats import norm
import math
import time

def cdf(ticks, mean, sigma):
    return 0.5 * (1 + torch.erf(ticks - mean) / sigma / math.sqrt(2))

# def probability_in_interval(depth_interval, depth, sigma):
    # # 返回一个depth在给定区间内的累计概率
    # prob_each_tick = norm.cdf(depth_interval, depth, sigma)
    # prob = np.diff(prob_each_tick)
    # return prob

def probability_in_interval(depth_interval, depth, sigma):
    # 返回一个depth在给定区间内的累计概率
    prob_each_tick = cdf(depth_interval, depth, sigma)
    prob = torch.diff(prob_each_tick, dim = -1)
    return prob


def gaussion_prob_from_multidepth_mat(depth, dbound, device):
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

    depth_interval_ticks = torch.arange(depth_start-depth_step/2, depth_end+depth_step/2, depth_step, dtype=torch.float)

    # depth_interval_ticks = torch.tensor(np.arange(
    #     depth_start-depth_step/2, depth_end+depth_step/2+1e-3, depth_step))    
    prob = probability_in_interval(depth_interval_ticks.to(device).unsqueeze(0),
                                    depth.unsqueeze(1),
                                    sigma)
    
    return prob


def gaussion_prob_from_multidepth(depth_dist, dbound, device):
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
    depth_interval_ticks = torch.tensor(np.arange(
        depth_start-depth_step/2, depth_end+depth_step/2+1e-3, depth_step))
    prob = torch.zeros(depth_interval_ticks.shape[0] - 1).to(device)
    
    for depth in depth_dist:
        prob += probability_in_interval(depth_interval_ticks.to(device), depth, sigma)
    if n_cluster > 0:
        prob = prob/torch.sum(prob)
    return prob

def generate_depth_prob_map(height, width, dbound, depth_collection_mat, device) -> np.ndarray:
    """ Generate depth probability map for given possible depth list
        Args:
            depth_collection_mat: np 
        Return:
            depth_prob_map: matrix of (height, width, number of depth centers)
                type: np.ndarray
    """
    n_depth = int((dbound[1] - dbound[0])/dbound[2] + 1)
    depth_prob_map = torch.zeros((height, width, n_depth)).to(device)
    for ih in range(height):
        for jw in range(width):
            if len(depth_collection_mat[ih][jw]) == 0: continue
            depth_prob_map[ih, jw, :] = gaussion_prob_from_multidepth(
                depth_collection_mat[ih][jw], dbound, device)
    return depth_prob_map

# def coord_float2ind(coor: np.ndarray):
#     return np.array(np.round(coor[:, :2] - 0.5, ), dtype=int)

def coord_float2ind(coor: torch.Tensor, downsample_ratio:int = 1):
    return torch.floor(torch.round((coor[:, :2] - 0.5)) / downsample_ratio).long()

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

    img_coord_float = img_coord_float[(img_coord_float[:,2]>clap[0]) &
                                       (img_coord_float[:,2]<clap[1])]
    idx = torch.argsort(img_coord_float[:, 2])
    img_coord_float = img_coord_float[idx, :]  # sort by depth
    img_coor = coord_float2ind(img_coord_float)
    ts = time.time()
    for i, coord in enumerate(img_coor):
        # if img_coord_float[i, 2] > clap[1] or img_coord_float[i, 2] < clap[0]:
        #     continue
        mat[coord[0]][coord[1]].append(img_coord_float[i, 2])
    print(f"for loop cost {time.time() -ts} s")
    return mat


def create_depth_scatter_matrix_varray(width, height, img_coord_float: torch.Tensor, clap=[0, 60]):
    """Create a list matrix represent the depth in each pixel.
        Args:
            img_coord_float: (N, 3) <== height, width, depth.
            clap (list[value]): interval of valid depth.
                default: [0, 60].
        Return:
            mat of size [height] [width] [depth num in this pixel]
    """
    ts = time.time()
    mat = [[[]] * width] * height  # height * width
    print(f"mat cost {time.time() -ts} s")
    ts = time.time()
    img_coord_float = img_coord_float[(img_coord_float[:,2]>clap[0]) &
                                       (img_coord_float[:,2]<clap[1]), :]
    # print(img_coord_float.sh)
    print(f"filter cost {time.time() -ts} s")
    ts = time.time()
    img_coor = coord_float2ind(img_coord_float)
    img_coor = torch.unique(img_coor, dim=0)
    print(f"unique cost {time.time() -ts} s")
    ts = time.time()
    for i, coord in enumerate(img_coor):
        pass
        # if img_coord_float[i, 2] > clap[1] or img_coord_float[i, 2] < clap[0]:
        #     continue
        mat[coord[0]][coord[1]] = 0
        # mat[coord[0]][coord[1]] = img_coord_float[i, 2]
    print(f"for loop cost {time.time() -ts} s")
    return mat

def create_depth_scatter_matrix_v2(width, height, img_coord_float: torch.Tensor,
                                    dbound,
                                    device,
                                    downsample_ratio = 1,
                                    clap=[0, 60]):
    """Create a list matrix represent the depth in each pixel.
        Args:
            img_coord_float: (N, 3) <== height, width, depth.
            clap (list[value]): interval of valid depth.
                default: [0, 60].
        Return:
            mat of size [height] [width] [depth num in this pixel]
    """
    n_depth = int((dbound[1] - dbound[0])/dbound[2])

    img_coord_float = img_coord_float[(img_coord_float[:,2]>clap[0]) &
                                       (img_coord_float[:,2]<clap[1])] # filter depth
    
    img_coord_depth_dist =  gaussion_prob_from_multidepth_mat(img_coord_float[:, 2], dbound, device)
    img_coor = coord_float2ind(img_coord_float, downsample_ratio = downsample_ratio)
    uniques, inverse = torch.unique(img_coor, dim=0, return_inverse=True)

    cumsum_depth_dist = torch.zeros((uniques.shape[0], img_coord_depth_dist.shape[-1])).to(device)

    for i in range(img_coord_depth_dist.shape[-1]):
        cumsum_depth_dist[:, i] = torch.bincount(inverse, weights=img_coord_depth_dist[:,i])
    new_height = int(height/downsample_ratio)
    new_width = int(width/downsample_ratio)
    # print(new_height, new_width, max(uniques[:,0]), max(uniques[:,1]))
    depth_prob_map = torch.zeros((new_height, new_width, n_depth)).to(device)
    depth_prob_map[uniques[:,0], uniques[:,1]] = cumsum_depth_dist
    
    return depth_prob_map



def create_depth_partition_tensor(width, height, depth_partition=20):
    return torch.zeros((width, height, depth_partition))


if __name__ == "__main__":
    pass
