#!python
import argparse
import os
import shutil
from glob import glob

import cv2
import numpy as np


def concat_alpha(path):
    shutil.copytree(path, path + '.bak')
    for alpha_path in glob(os.path.join(path, 'r_*_alpha.png')):
        img_path = alpha_path[:-10] + '.png'
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        alpha = cv2.imread(alpha_path, cv2.IMREAD_UNCHANGED)[..., None]
        img = np.concatenate((img, alpha), axis=-1)
        cv2.imwrite(img_path, img)


def copy_test2val(path):
    shutil.copy(os.path.join(path, 'transforms_test.json'), os.path.join(path, 'transforms_val.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset Glossy Synthetic.')
    parser.add_argument('path', type=str, help='root path of dataset.')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    concat_alpha(os.path.join(args.path, 'ball', 'train'))
    concat_alpha(os.path.join(args.path, 'ball', 'test'))

    for scene in ['ball', 'car', 'coffee', 'helmet', 'teapot', 'toaster']:
        copy_test2val(os.path.join(args.path, scene))
