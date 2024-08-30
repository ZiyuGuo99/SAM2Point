import cv2
import os
import re
import argparse

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group()) if match else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['S3DIS', 'ScanNet', 'Objaverse', 'KITTI', 'Semantic3D'], default='Objaverse', help='dataset selected')
    parser.add_argument('--prompt_type', choices=['point', 'box', 'mask'], default='point', help='prompt type selected')
    parser.add_argument('--sample_idx', type=int, default=0, help='the index of the scene or object')
    parser.add_argument('--prompt_idx', type=int, default=0, help='the index of the prompt')    
    args = parser.parse_args()
    name_list = ["tmp/" + args.dataset, "sample" + str(args.sample_idx), args.prompt_type + "-prompt" + str(args.prompt_idx)]
    name = '_'.join(name_list)

    for axis in ['x', 'y', 'z']:
        folder_path = name + 'frames/' + axis
        output_video = name + 'frames/' + axis + '.mp4'

        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort(key=extract_number)

        if not image_files:
            raise ValueError('No images found in the specified folder.')

        first_image_path = os.path.join(folder_path, image_files[0])
        first_image = cv2.imread(first_image_path)
        height, width, layers = first_image.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_writer = cv2.VideoWriter(output_video, fourcc, 10.0, (width, height))

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            video_writer.write(image)

        video_writer.release()
        print(f'Video saved as {output_video}')

if __name__=='__main__':
    main()