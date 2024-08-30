import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def render_scene(points, colors, name, prompt_point=None, prompt_box=None):
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    fig = plt.figure(figsize=(16, 8))
    views = [[0, 45], [45, -45], [90, 0]]
    if 'Objaverse' in name:
        alpha = 0.5 
    elif 'ScanNet' in name:
        alpha = 0.04
    elif 'S3DIS' in name:
        alpha = 0.02

    for i in range(1, 4):
        ax = fig.add_subplot(1, 3, i, projection='3d')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker='o', s=1, alpha=alpha)
        if prompt_point != None:
            ax.scatter(prompt_point[0], prompt_point[1], prompt_point[2], color='black', marker='*', s=90) 
        if prompt_box != None:
            xmin, ymin, zmin, xmax, ymax, zmax = prompt_box
            box_edges = [
                [(xmin, ymin, zmin), (xmax, ymin, zmin)],
                [(xmin, ymax, zmin), (xmax, ymax, zmin)],
                [(xmin, ymin, zmax), (xmax, ymin, zmax)],
                [(xmin, ymax, zmax), (xmax, ymax, zmax)],
                [(xmin, ymin, zmin), (xmin, ymax, zmin)],
                [(xmax, ymin, zmin), (xmax, ymax, zmin)],
                [(xmin, ymin, zmax), (xmin, ymax, zmax)],
                [(xmax, ymin, zmax), (xmax, ymax, zmax)],
                [(xmin, ymin, zmin), (xmin, ymin, zmax)],
                [(xmax, ymin, zmin), (xmax, ymin, zmax)],
                [(xmin, ymax, zmin), (xmin, ymax, zmax)],
                [(xmax, ymax, zmin), (xmax, ymax, zmax)],
            ]
            for edge in box_edges:
                x_values, y_values, z_values = zip(*edge)
                ax.plot(x_values, y_values, z_values, color='black', alpha=1)

        ax.view_init(elev=views[i-1][0], azim=views[i-1][1])

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    plt.tight_layout()
    os.makedirs('results/', exist_ok=True)
    plt.savefig('results/' + name + '.png')



def render_scene_outdoor(points, colors, name, prompt_point=None, prompt_box=None, close=False, semantic=False, args=None):
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    views = [[45, -45], [90, 0]]
    alpha = 0.1 

    if close:
        x_min, x_max = 0.48, 0.59
        y_min, y_max = 0.55, 0.85
        z_min, z_max = 0.2, 1.2
        views = [[0, 90], [0, 45]]
        alpha = 0.05
    if semantic:
        views = [[0, 0], [45, -45]] if args.sample_idx == 0 else [[0, -90], [45, -45]]
        alpha = 0.05

    fig = plt.figure(figsize=(24, 8))

    for i in range(1, 3):
        ax = fig.add_subplot(1, 2, i, projection='3d')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker='o', s=1, alpha=alpha)
        if prompt_point != None:
            ax.scatter(prompt_point[0], prompt_point[1], prompt_point[2], color='red', marker='*', s=90) 
        if prompt_box != None:
            xmin, ymin, zmin, xmax, ymax, zmax = prompt_box
            box_edges = [
                [(xmin, ymin, zmin), (xmax, ymin, zmin)],
                [(xmin, ymax, zmin), (xmax, ymax, zmin)],
                [(xmin, ymin, zmax), (xmax, ymin, zmax)],
                [(xmin, ymax, zmax), (xmax, ymax, zmax)],
                [(xmin, ymin, zmin), (xmin, ymax, zmin)],
                [(xmax, ymin, zmin), (xmax, ymax, zmin)],
                [(xmin, ymin, zmax), (xmin, ymax, zmax)],
                [(xmax, ymin, zmax), (xmax, ymax, zmax)],
                [(xmin, ymin, zmin), (xmin, ymin, zmax)],
                [(xmax, ymin, zmin), (xmax, ymin, zmax)],
                [(xmin, ymax, zmin), (xmin, ymax, zmax)],
                [(xmax, ymax, zmin), (xmax, ymax, zmax)],
            ]
            for edge in box_edges:
                x_values, y_values, z_values = zip(*edge)
                ax.plot(x_values, y_values, z_values, color='black', alpha=1)

        ax.view_init(elev=views[i-1][0], azim=views[i-1][1])

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False


    plt.tight_layout()
    os.makedirs('results/', exist_ok=True)
    if not semantic:   name = name + '_closeview' if close else name + '_farview'
    plt.savefig('results/' + name + '.png')


