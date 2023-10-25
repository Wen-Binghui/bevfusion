import torch
import time
device = 'cuda:0'
N = 10000
C = 16
size = 50
point_cloud = 1.3 * size * (torch.rand((N, 3)).cuda() - 0.5)
ts = time.time()
map_feature = torch.arange(0, size * size).float().cuda()
map_feature = map_feature.reshape(size, size).unsqueeze(-1).repeat(1, 1, C)

indics = torch.floor(point_cloud[:, 0:2] + size/2).long()
indics_valid = (indics[:,0] >= 0) & (indics[:,0] < size) & (indics[:,1] >= 0) & (indics[:,1] < size)
semantic_feature = torch.zeros((N, C)).cuda()
semantic_feature[indics_valid, :] = map_feature[indics[indics_valid, 0], indics[indics_valid, 1], :]
te = time.time()
print(f'computation costs {te - ts} s.')
print(semantic_feature.shape)