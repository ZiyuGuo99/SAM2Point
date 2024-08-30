# Codes are taken from BPNet, CVPR'21
# https://github.com/wbhu/BPNet/blob/main/dataset/voxelizer.py

import collections
import numpy as np
from sam2point.voxelization_utils import sparse_quantize
from scipy.linalg import expm, norm


# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class Voxelizer:

    def __init__(self, voxel_size=1, clip_bound=None):
        '''
        Args:
          voxel_size: side length of a voxel
          clip_bound: boundary of the voxelizer. Points outside the bound will be deleted
            expects either None or an array like ((-100, 100), (-100, 100), (-100, 100)).
          ignore_label: label assigned for ignore (not a training label).
        '''
        self.voxel_size = voxel_size
        self.clip_bound = clip_bound

    def get_transformation_matrix(self):
        voxelization_matrix = np.eye(4)

        # Transform pointcloud coordinate to voxel coordinate.
        scale = 1 / self.voxel_size
        np.fill_diagonal(voxelization_matrix[:3, :3], scale)
        # Get final transformation matrix.
        return voxelization_matrix

    def clip(self, coords, center=None):
        bound_min = np.min(coords, 0).astype(float)
        bound_max = np.max(coords, 0).astype(float)
        bound_size = bound_max - bound_min
        if center is None:
            center = bound_min + bound_size * 0.5
        lim = self.clip_bound
        
        # Clip points outside the limit
        clip_inds = ((coords[:, 0] >= (lim[0][0] + center[0])) &
                     (coords[:, 0] < (lim[0][1] + center[0])) &
                     (coords[:, 1] >= (lim[1][0] + center[1])) &
                     (coords[:, 1] < (lim[1][1] + center[1])) &
                     (coords[:, 2] >= (lim[2][0] + center[2])) &
                     (coords[:, 2] < (lim[2][1] + center[2])))
        return clip_inds

    def voxelize(self, coords, feats, labels, center=None, link=None, return_ind=False):
        assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]
        if self.clip_bound is not None:
            clip_inds = self.clip(coords, center)
            if clip_inds.sum():
                coords, feats = coords[clip_inds], feats[clip_inds]
                if labels is not None:
                    labels = labels[clip_inds]

        # Get rotation and scale
        M_v = self.get_transformation_matrix()
        # Apply transformations
        rigid_transformation = M_v
        homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
        coords_aug = np.floor(homo_coords @ rigid_transformation.T[:, :3])

        # Align all coordinates to the origin.
        min_coords = coords_aug.min(0)
        M_t = np.eye(4)
        M_t[:3, -1] = -min_coords
        rigid_transformation = M_t @ rigid_transformation
        coords_aug = np.floor(coords_aug - min_coords)

        inds, inds_reconstruct = sparse_quantize(coords_aug, return_index=True) #NOTE
        coords_aug, feats, labels = coords_aug[inds], feats[inds], labels[inds] #NOTE

        if return_ind:
            return coords_aug, feats, labels, np.array(inds_reconstruct), inds
        if link is not None:
            return coords_aug, feats, labels, np.array(inds_reconstruct), link[inds]

        return coords_aug, feats, labels, np.array(inds_reconstruct)
