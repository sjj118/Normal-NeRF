#!python
import argparse
import json
import os
import pickle
from glob import glob
from pathlib import Path

import cv2
import numpy as np

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

scene_names = ['angel', 'bell', 'cat', 'horse', 'luyu', 'potion', 'tbell', 'utah_teapot']


def write_to_json(filename: Path, content: dict):
    with open(filename, 'w', encoding='UTF-8') as file:
        json.dump(content, file, indent='\t')


def process(scene_source, scene_target, split):
    src_root = os.path.join(args.source, scene_source)
    img_num = len(glob(os.path.join(src_root, '*.pkl')))

    poses = []
    Ks = []
    for k in range(img_num):
        with open(os.path.join(src_root, f'{k}-camera.pkl'), 'rb') as f:
            pose, K = pickle.load(f)
            poses.append(pose)
            Ks.append(K)
    poses = np.array(poses)
    Ks = np.array(Ks)

    assert (Ks[1:] == Ks[:-1]).all(), 'All K matrix should be same'
    K = Ks[0]

    assert K[0, 0] == K[1, 1], 'fx should == fy'
    focal_length = K[0, 0]

    img_0 = cv2.imread(os.path.join(src_root, '0.png'))
    image_height, image_width = img_0.shape[:2]
    assert image_width / 2.0 == K[0, 2] and image_height / 2.0 == K[1, 2], 'image width, height not match with cx, cy'

    camera_angle_x = 2 * np.arctan(0.5 * image_width / focal_length)
    frames = []
    for k in range(img_num):
        pose = poses[k]
        pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])])
        pose[0, :3] = -pose[0, :3]
        pose = np.linalg.inv(pose)
        pose = pose[[0, 2, 1, 3]]
        pose[0, 3] = -pose[0, 3]
        pose[1:, :3] = -pose[1:, :3]
        frame = dict(
            file_path=f'./{split}/r_{k}',
            transform_matrix=pose.tolist(),
        )
        frames.append(frame)

    meta = dict(
        camera_angle_x=camera_angle_x,
        frames=frames,
    )

    target_root = os.path.join(args.target, scene_target)
    os.makedirs(os.path.join(target_root, split))
    write_to_json(Path(target_root) / f'transforms_{split}.json', meta)
    if split == 'test':
        write_to_json(Path(target_root) / 'transforms_val.json', meta)
    for k in range(img_num):
        if args.verbose:
            print(f'{scene_target}_{k}')
        if split == 'train':
            depth = cv2.imread(os.path.join(src_root, f'{k}-depth.png'), cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float32) / 65535 * 15
        else:
            depth = cv2.imread(os.path.join(src_root, f'{k}-depth0001.exr'), cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float32) * 15
        alpha = depth < 14.5
        image = cv2.imread(os.path.join(src_root, f'{k}.png'))
        image = np.concatenate([image, (255 * alpha)[..., None]], axis=-1)
        cv2.imwrite(os.path.join(target_root, split, f'r_{k}.png'), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset Glossy Synthetic.')
    parser.add_argument('source', type=str, help='root path of dataset.')
    parser.add_argument('target', type=str, help='target output directory.')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        raise OSError(f'{args.source} is not a directory.')
    os.makedirs(args.target)

    for scene in scene_names:
        process('teapot' if scene == 'utah_teapot' else scene, scene, 'train')
        process(scene + '_nvs', scene, 'test')
