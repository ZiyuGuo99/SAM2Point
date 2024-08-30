import os
import shutil
import numpy as np
import scipy.io as sio
import torch


def load_S3DIS_sample(text_path, sample=False):
    print("Loading point cloud from ", text_path)

    data = np.loadtxt(text_path)
    point, color = data[:, :3], data[:, 3:]

    point = point - point.min(axis=0)
    point = point / point.max(axis=0)
    color = color / 255.

    return point, color

def load_ScanNet_sample(data_path):
    print("Loading point cloud from ", data_path)
    
    all_data = torch.load(data_path)
    
    point = np.array(all_data['coord'])
    color = np.array(all_data['color'])
 
    point = point - point.min(axis=0)
    point = point / point.max(axis=0)
    color = color / 255.

    return point, color

def load_KITTI_sample(data_path, close=False):
    print("Loading point cloud from ", data_path)
    all_data = np.load(data_path)
    
    point = all_data[:, :3]
    color = all_data[:, 3:6]
    
    pmin = point.min(axis=0)
    point = point - pmin
    pmax = point.max(axis=0)
    point = point / pmax

    return point, color

def load_Objaverse_sample(data_path):
    print("Loading point cloud from ", data_path)
    all_data = np.load(data_path)
    
    point = all_data[:, :3]
    color = all_data[:, 3:6]
    
    pmin = point.min(axis=0)
    point = point - pmin
    pmax = point.max(axis=0)
    point = point / pmax
    
    return point, color

def load_Semantic3D_sample(data_path, id, sample=False):
    print("Loading point cloud from ", data_path)
    all_data = np.load(data_path)
    
    point = all_data[:, :3]
    color = all_data[:, 3:6]
    
    pmin = point.min(axis=0)
    point = point - pmin
    pmax = point.max(axis=0)
    point = point / pmax

    if id > 1:  return point, color
    if id == 0:
        filter_mask = (point[:, 0] > 0.4) & (point[:, 1] > 0.4) & (point[:, 2] < 0.4)
    else:
        filter_mask = (point[:, 0] > 0.4) & (point[:, 1] < 0.5)
    point = point[filter_mask]
    color = color[filter_mask]

    pmin = point.min(axis=0)
    point = point - pmin
    pmax = point.max(axis=0)
    point = point / pmax
    
    return point, color
