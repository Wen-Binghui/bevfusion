from PIL import Image
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import copy

def load_img(filename):
    image = Image.open(filename)
    return image


def dummy_depth(height, width, N=2000):
    # create dummy depth info of (N, 3) <= x, y, depth
    coord = torch.rand((N, 3))
    coord[:, 0] *= height
    coord[:, 1] *= width
    coord[:, 2] *= 40
    return coord


def plot_img_in3D(img, fig, ax):
    print(img.size)
    ax.imshow(np.array(img), cmap=plt.cm.BrBG)  # , extent=[0, 1, 0, 1]

    pass



def bar3d_for_depthmap(height, 
                       width, 
                       depth_prob_map, 
                       dbound, 
                       ax, 
                       cmap, 
                       destimg_path):
    """ create 3D bar plot for one depth map.
        Args:
            height (int): Img Height.
            width (int): Img Width.
            depth_prob_map (list[value]): a matrix of depth.
            dbound (list[value]): depth boundary, e.g. []
            ax (plt.axes): target axes.
            cmap (dict): color map.
            destimg_path (path|str): Path to save figure.
                default: False.
        """
    for ih in range(height):
        for jw in range(width):
            if np.max(depth_prob_map[ih, jw, :]) < 3e-2:
                continue
            depth_dist_ij = copy.deepcopy(depth_prob_map[ih, jw, :])
            depth_dist_ij = np.vstack(
                [np.arange(depth_dist_ij.shape[0]), depth_dist_ij])
            depth_dist_ij = depth_dist_ij[:, depth_dist_ij[1, :] > 3e-2]
            for k in range(depth_dist_ij.shape[1]):
                z = dbound[0] + depth_dist_ij[0, k] * dbound[2]
                color = cmap(depth_dist_ij[1, k])
                # 添加 alpha 值使颜色半透明
                ax.bar3d(ih, jw, -z, 1, 1, dbound[2],
                        color=color, shade=False, alpha=0.3)
    ax.set_title('3D Bar Plot with transparency')
    ax.set_box_aspect([1, 1, 1])
    # plt.show()
    plt.savefig(destimg_path, dpi=400)

if __name__ == '__main__':
    img = load_img(
        'D:\File_Py/bev_vis\data/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151615678113.jpg')
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')
    coord = dummy_depth(img.size[1], img.size[0])
    plot_img_in3D(img, fig, ax1)

    plt.show()
    pass
