from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import six
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from got10k.datasets import ImageNetVID, GOT10k


class SOT_3D_Loader(object):

    def __init__(self, root_dir, subset='train'):
        super(SOT_3D_Loader, self).__init__()
        assert subset in ['train', 'val', 'test'], 'Unknown subset.'
        self.dataset_name = '3D_SOT'
        self.root_dir = root_dir
        self.subset = subset
        self.meta = {'resolution': '(1920, 1080)'}
        subset_dir = os.path.join(root_dir, subset)
        self.sequence_directories = sorted([os.path.join(subset_dir, sequence)
                                for sequence in os.listdir(subset_dir)])
    pass

    def __getitem__(self, index):
        """
            Args:
                index (integer): Index of a trajectory.

           Returns:
                single_sequence: SingleSeq class, contains left and right images
                    and corresponding annotations
        """
        single_sequence = SingleSeq(self.sequence_directories[index])

        return single_sequence

    def __len__(self):
        return len(self.sequence_directories)

class SingleSeq(object):

    def __init__(self, sequence_dir):
        self.sequence_dir = sequence_dir
        # load camera parameters
        self.calibration_params_file = os.path.join(self.sequence_dir, 'calibration.npz')
        self.__load_calibration_params_npz(self.calibration_params_file)
        # load binocular images
        left_img_dir = os.path.join(self.sequence_dir, 'img/left')
        self.left_img_files = sorted(glob.glob(os.path.join(left_img_dir, '*.jpg')))
        right_img_dir = os.path.join(self.sequence_dir, 'img/right')
        self.right_img_files = sorted(glob.glob(os.path.join(right_img_dir, '*.jpg')))
        # load annotations
        self.annotation_file = os.path.join(sequence_dir, 'groundtruth_rect.csv')
        data_frame = pd.read_csv(self.annotation_file)
        self.left_annotation = [list(map(int, eval(data))) for data in data_frame['left roi']]
        self.right_annotation = [list(map(int, eval(data))) for data in data_frame['right roi']]
    pass

    def __getitem__(self, index):
        left_img_file = self.left_img_files[index]
        right_img_file = self.right_img_files[index]
        left_roi = self.left_annotation[index]
        right_roi = self.right_annotation[index]
        return left_img_file, left_roi, right_img_file, right_roi


    def __len__(self):
        left_img_length = len(self.left_img_files)
        right_img_length = len(self.right_img_files)
        if left_img_length == right_img_length:
            return left_img_length
        else:
            print("left image length: %d" % left_img_length)
            print("right image length: %d" % right_img_length)
            print("the left image length is not equal to the right")
            return 0

    def __load_calibration_params_npz(self, npz_file):
        calibration_params = np.load(npz_file)
        self.left_camera_matrix = calibration_params['left_camera_matrix']
        self.right_camera_matrix = calibration_params['right_camera_matrix']




if __name__ == "__main__":
    sot_dataset = SOT_3D_Loader('E:\\SOT_3D', 'test')
    for s, sequence in enumerate(sot_dataset):
        print("sequence: %d" % s)
        print("sequence length: %d" % sequence.__len__())
        print("sequence path: %s" % sot_dataset.sequence_directories[s])
        for f, (left_img_file, left_roi, right_img_file, right_roi) in enumerate(sequence):
            print("frame: %d" % f)
            left_img = cv2.imread(left_img_file, 1)
            right_img = cv2.imread(right_img_file, 1)
            left_pt1 = [left_roi[0], left_roi[1]]
            left_pt2 = [left_roi[0] + left_roi[2], left_roi[1] + left_roi[3]]
            right_pt1 = [right_roi[0], right_roi[1]]
            right_pt2 = [right_roi[0] + right_roi[2], right_roi[1] + right_roi[3]]
            cv2.rectangle(left_img, pt1=tuple(left_pt1), pt2=tuple(left_pt2),
                          color=(255, 0, 255), thickness=2)
            cv2.rectangle(right_img, pt1=tuple(right_pt1), pt2=tuple(right_pt2),
                          color=(255, 0, 255), thickness=2)
            visualization = np.hstack((left_img, right_img))
            cv2.namedWindow('visualization', cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow('visualization', visualization)
            cv2.waitKey()

