import os
import cv2
import shutil
import numpy as np
import torch


def build_fold(path):
    if os.path.exists(path):
        return True
        # shutil.rmtree(path)
        # return True 
    os.makedirs(path)
    return False
    

def visualize_frame_with_mask(grid0, grid1, grid2, a0_mask, a1_mask, a2_mask, point_coords, resolution, name, args=None):
    os.makedirs(name + 'frames_seg', exist_ok=True)
    a0_dir, a1_dir, a2_dir = name + 'frames_seg/x', name + 'frames_seg/y', name + 'frames_seg/z'
    
    a0_mask, a1_mask, a2_mask = a0_mask.repeat(1, 3, 1, 1), a1_mask.repeat(1, 3, 1, 1), a2_mask.repeat(1, 3, 1, 1)
    a0_mask[:, 1:], a1_mask[:, 1:], a2_mask[:, 1:] = a0_mask[:, 1:] * 0, a1_mask[:, 1:] * 0, a2_mask[:, 1:] * 0
    
    grid0, grid1, grid2 = grid0 * 0.7 + a0_mask * 0.3, grid1 * 0.7 + a1_mask * 0.3, grid2 * 0.7 + a2_mask * 0.3
    
    grid0[point_coords[0], :, point_coords[1], point_coords[2]] = torch.Tensor([0., 1., 0.])
    grid1[point_coords[1], :, point_coords[0], point_coords[2]] = torch.Tensor([0., 1., 0.])
    grid2[point_coords[2], :, point_coords[0], point_coords[1]] = torch.Tensor([0., 1., 0.])
    if not build_fold(a0_dir):
        visualize_per_frame(grid0, a0_dir, resolution, args)
    if not build_fold(a1_dir):
        visualize_per_frame(grid1, a1_dir, resolution, args)
    if not build_fold(a2_dir):
        visualize_per_frame(grid2, a2_dir, resolution, args)
    

def visualize_per_frame(grid, foldpath, resolution, args=None):
    grid = torch.nn.functional.interpolate(grid, size=(resolution, resolution), mode=args.mode)
    
    imgs = grid.cpu().numpy()
    n, _, _, _ = grid.shape
    for ii in range(n):
        r = np.uint8(imgs[ii, 0, :, :]*255)
        g = np.uint8(imgs[ii, 1, :, :]*255)
        b = np.uint8(imgs[ii, 2, :, :]*255)
        img = cv2.merge([b, g, r])
        cv2.imwrite('{}/{}.png'.format(foldpath, ii), img)
    return

def cal(input, points):
    reference_point_3d = np.array(input)  
    distances = np.linalg.norm(points - reference_point_3d, axis=1)
    closest_index = np.argmin(distances)
    closest_point = points[closest_index]
    return [closest_point[0], closest_point[1], closest_point[2]]
