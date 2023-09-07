import torch
import numpy as np

def create_depth_scatter_matrix(width, height, img_coord:torch.Tensor):
    # img_coord: (N, 3)
    mat = np.zeros((width, height), dtype=list)
    idx = torch.argsort(img_coord[:,2])
    img_coord = img_coord[idx, :]
    print(img_coord)
    return mat

def create_depth_partition_tensor(width, height, depth_partition = 20):
    return torch.zeros((width, height, depth_partition))



if __name__ == "__main__":


    pass