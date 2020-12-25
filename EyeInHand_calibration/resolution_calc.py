import argparse
import struct
import sys
import pyzed.sl as sl
import cv2
import numpy as np
import socket
import math
from ctypes import *
import re
import os
import time
import glob
import EyeInHand_calibration as EIH

if __name__ == '__main__':
    chessboard_size = (6, 9)
    obj_points_3d = EIH.generate_object_points_in_3d(chessboard_size, 25)

    # initial camera
    init_params = sl.InitParameters()
    init_params.svo_real_time_mode = True
    init_params.coordinate_units = sl.UNIT.UNIT_MILLIMETER
    init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
    init_params.camera_fps = 30
    zed = sl.Camera()

    # start camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print('Failed to open ZED!!!')
        zed.close()
        exit(1)

    while (cv2.waitKey(1) != 27):
        left_img, right_img = EIH.get_image_by_zed_camera(zed)

        chessboard_corners = EIH.get_chessboard_corners(left_img, chessboard_size)
        if chessboard_corners is None:
            print('=> Retrying ...')
            continue
        else:
            break

    # calculate camera resolution in x and y axis
    x_resolutions = []
    y_resolutions = []
    # x-axis
    for s, point in enumerate(obj_points_3d):
        next_point = obj_points_3d[(s + 1) % (chessboard_size[0] * chessboard_size[1])]
        # print('point 3d: {}, next point 3d: {}'.format(point, next_point))
        delta_x_3d = next_point[0] - point[0]

        point_2d = chessboard_corners[s].reshape(-1, 1)
        next_point_2d = chessboard_corners[(s + 1) % (chessboard_size[0] * chessboard_size[1])].reshape(-1, 1)
        # print('point 2d: {}, next point 2d: {}'.format(point_2d, next_point_2d))
        delta_x_2d = next_point_2d[0] - point_2d[0]
        resolution = delta_x_3d / delta_x_2d
        x_resolutions.append(resolution)
        print('the resolution in x-axis between point {} and point {}: {}'.format(
            s, (s + 1) % (chessboard_size[0] * chessboard_size[1]), resolution
        ))
    x_resolutions = np.array(x_resolutions).reshape(1, -1)
    x_resolution = np.mean(x_resolutions)
    print('The total resolution in x-axis: {}'.format(x_resolution))
    # y-axis
    for s, point in enumerate(obj_points_3d):
        next_point = obj_points_3d[(s + chessboard_size[0]) % (chessboard_size[0] * chessboard_size[1])]
        # print('point 3d: {}, next point 3d: {}'.format(point, next_point))
        delta_y_3d = abs(next_point[1] - point[1])
        # print('delta y in 3d: {}'.format(delta_y_3d))

        point_2d = chessboard_corners[s].reshape(-1, 1)
        next_point_2d = chessboard_corners[(s + chessboard_size[0]) % (chessboard_size[0] * chessboard_size[1])].reshape(-1, 1)
        # print('point 2d: {}, next point 2d: {}'.format(point_2d, next_point_2d))
        delta_y_2d = abs(next_point_2d[1] - point_2d[1])
        # print('delta y in 2d: {}'.format(delta_y_2d))
        resolution = delta_y_3d / delta_y_2d
        y_resolutions.append(resolution)
        print('the resolution in y-axis between point {} and point {}: {}'.format(
            s + 1, (s + 1 + chessboard_size[0]) % (chessboard_size[0] * chessboard_size[1]), resolution
        ))
    y_resolutions = np.array(y_resolutions).reshape(1, -1)
    y_resolution = np.mean(y_resolutions)
    print('The total resolution in y-axis: {}'.format(y_resolution))

    cv2.waitKey()