# -*- coding: utf-8 -*-
import os
import torch
import argparse
import numpy as np
import open3d as o3d

from segment import seg_point, seg_box, seg_mask
import sam2point.dataset as dataset
import sam2point.configs as configs
from sam2point.voxelizer import Voxelizer
from sam2point.utils import cal
from show import render_scene, render_scene_outdoor
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['S3DIS', 'ScanNet', 'Objaverse', 'KITTI', 'Semantic3D'], default='Objaverse', help='dataset selected')
    parser.add_argument('--prompt_type', choices=['point', 'box', 'mask'], default='point', help='prompt type selected')
    parser.add_argument('--sample_idx', type=int, default=0, help='the index of the scene or object')
    parser.add_argument('--prompt_idx', type=int, default=0, help='the index of the prompt')    
    parser.add_argument('--voxel_size', type=float, default=0.02, help='voxel size')    
    parser.add_argument('--theta', type=float, default=0.) 
    parser.add_argument('--mode', type=str, default='bilinear')  
    
    args = parser.parse_args()
    name_list = [args.dataset, "sample" + str(args.sample_idx), args.prompt_type + "-prompt" + str(args.prompt_idx)]
    name = '_'.join(name_list)

    result_name = "cache_results/" + name + '.npy'
    prompt_name = "cache_prompt/" + name + '.npy'

    if os.path.exists("./cache_results/" + name + '.npy') and os.path.exists("./cache_prompt/" + name + '.npy'):
        new_color = np.load("./cache_results/" + name + '.npy') ###
        PROMPT = np.load("./cache_prompt/" + name + '.npy') ###

    if args.dataset == 'S3DIS':
        info = configs.S3DIS_samples[args.sample_idx]
        point, color = dataset.load_S3DIS_sample(info['path'])
    elif args.dataset == 'ScanNet':
        info = configs.ScanNet_samples[args.sample_idx]
        point, color = dataset.load_ScanNet_sample(info['path'])
    elif args.dataset == 'Objaverse':
        info = configs.Objaverse_samples[args.sample_idx]
        point, color = dataset.load_Objaverse_sample(info['path'])
        args.voxel_size = info[configs.VOXEL[args.prompt_type]][args.prompt_idx] 
        args.mode, args.theta = 'nearest', 0.5
    elif args.dataset == 'KITTI':
        info = configs.KITTI_samples[args.sample_idx]
        point, color = dataset.load_KITTI_sample(info['path'])
        args.voxel_size = info[configs.VOXEL[args.prompt_type]][args.prompt_idx] 
        args.mode, args.theta = 'nearest', 0.5
    elif args.dataset == 'Semantic3D':
        info = configs.Semantic3D_samples[args.sample_idx]
        point, color = dataset.load_Semantic3D_sample(info['path'], args.sample_idx)
        args.voxel_size = info[configs.VOXEL[args.prompt_type]][args.prompt_idx] 
        args.mode, args.theta = 'nearest', 0.5
    
    print(args)
    point_color = np.concatenate([point, color], axis=1)
    voxelizer = Voxelizer(voxel_size=args.voxel_size, clip_bound=None)
    
    labels_in = point[:, :1].astype(int)
    locs, feats, labels, inds_reconstruct = voxelizer.voxelize(point, color, labels_in)

    if args.prompt_type == 'point':
        mask = seg_point(locs, feats, info['point_prompts'], args)
        point_prompts = np.array(info['point_prompts'])
        prompt_point = list(point_prompts[args.prompt_idx])
        prompt_box = None
    elif args.prompt_type == 'box':
        mask = seg_box(locs, feats, info['box_prompts'], args)
        box_prompts = np.array(info['box_prompts'])
        prompt_point = None
        prompt_box = list(box_prompts[args.prompt_idx])
    elif args.prompt_type == 'mask':
        if 'mask_prompts' not in info:  info['mask_prompts'] = info['point_prompts']
        mask, prompt_mask = seg_mask(locs, feats, info['mask_prompts'], args)
        prompt_point, prompt_box = None, None
    
    point_locs = locs[inds_reconstruct]
    point_mask = mask[point_locs[:, 0], point_locs[:, 1], point_locs[:, 2]]
    
    point_mask = point_mask.unsqueeze(-1)
    point_mask_not = ~point_mask
    
    point, color = point_color[:, :3], point_color[:, 3:]
    new_color = color * point_mask_not.numpy() + (color * 0 + np.array([[0., 1., 0.]])) * point_mask.numpy()

    os.makedirs('results', exist_ok=True)
    name_list = [args.dataset, "sample" + str(args.sample_idx), args.prompt_type + "-prompt" + str(args.prompt_idx)]
    name = '_'.join(name_list)

    if args.dataset == 'KITTI':    
        render_scene_outdoor(point, new_color, name, prompt_point=prompt_point, prompt_box=prompt_box)  
        render_scene_outdoor(point, new_color, name, prompt_point=prompt_point, prompt_box=prompt_box, close=True)  
    elif args.dataset == 'Semantic3D':
        render_scene_outdoor(point, new_color, name, prompt_point=prompt_point, prompt_box=prompt_box, semantic=True, args=args)  
    else:
        render_scene(point, new_color, name, prompt_point=prompt_point, prompt_box=prompt_box)  

    if args.prompt_type == 'mask':
        point_prompt_mask = prompt_mask[point_locs[:, 0], point_locs[:, 1], point_locs[:, 2]]
        point_prompt_mask = point_prompt_mask.unsqueeze(-1)
        point_prompt_mask_not = ~point_prompt_mask

        color_prompt_mask = color * point_prompt_mask_not.numpy() + (color * 0 + np.array([[1., 0., 0.]])) * point_prompt_mask.numpy()
        name = [args.dataset, "sample" + str(args.sample_idx), args.prompt_type + "-prompt" + str(args.prompt_idx), 'maskprompt']
        name = '_'.join(name)
        render_scene(point, color_prompt_mask, name, prompt_point=prompt_point, prompt_box=prompt_box)  

if __name__=='__main__':
    main()
    

