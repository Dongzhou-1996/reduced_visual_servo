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




def load_calibration_params(camera=sl.Camera()):
    # Load calibration parameters
    calibration_params = camera.get_camera_information().calibration_parameters
    r_vec = calibration_params.R
    r_mat, jacobi = cv2.Rodrigues(r_vec)
    t = np.array(calibration_params.T).reshape(3, 1)

    left_cam_params = calibration_params.left_cam
    right_cam_params = calibration_params.right_cam
    left_intrinsic_matrix = np.array([[left_cam_params.fx, 0, left_cam_params.cx],
                                      [0, left_cam_params.fy, left_cam_params.cy],
                                      [0, 0, 1]], dtype=float)
    right_intrinsic_matrix = np.array([[right_cam_params.fx, 0, right_cam_params.cx],
                                       [0, right_cam_params.fy, right_cam_params.cy],
                                       [0, 0, 1]], dtype=float)
    print("Rotate matrix: {}".format(r_mat))
    print("Translate vector: {}".format(t))
    print("left intrinsic matrix: {}".format(left_intrinsic_matrix))
    print("right intrisic matrix: {}".format(right_intrinsic_matrix))

    return left_intrinsic_matrix, right_intrinsic_matrix, r_mat, t


def load_left_camera_params(camera=sl.Camera()):
    # Load calibration parameters
    calibration_params = camera.get_camera_information().calibration_parameters

    left_cam_params = calibration_params.left_cam
    left_intrinsic_matrix = np.array([[left_cam_params.fx, 0, left_cam_params.cx],
                                      [0, left_cam_params.fy, left_cam_params.cy],
                                      [0, 0, 1]], dtype=float)
    left_distortion_vec = left_cam_params.disto

    # print("left intrinsic matrix: {}".format(left_intrinsic_matrix))
    # print("left distortion vector: {}".format(left_distortion_vec))

    return left_intrinsic_matrix, left_distortion_vec


def get_image_by_zed_camera(zed):
    left_image = sl.Mat()
    right_image = sl.Mat()
    # Runtime parameters
    runtime_params = sl.RuntimeParameters()
    runtime_params.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD
    while 1:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # print("=> Successfully retrieved image! ")
            zed.retrieve_image(left_image, sl.VIEW.VIEW_LEFT)
            zed.retrieve_image(right_image, sl.VIEW.VIEW_RIGHT)
            break
    left_img = left_image.get_data()
    right_img = right_image.get_data()

    return left_img, right_img


def find_chessboard(img, chessboard_size=(6, 9)):
    # ret, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # cv2.namedWindow('binary image', cv2.WINDOW_GUI_EXPANDED)
    # cv2.imshow('binary image', 255 - binary_img)

    edge_img = cv2.Canny(img, 20, 150)
    # cv2.namedWindow('edge image', cv2.WINDOW_GUI_EXPANDED)
    # cv2.imshow('edge image', edge_img)

    contours, hierachy = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_img = np.zeros((img.shape[0], img.shape[1], 3))

    # cv2.drawContours(contours_img, contours, -1, (255, 0, 255), 3)
    if len(contours) == 0:
        # print('=> Failed to find contours')
        return None
    # contours selection
    asp_ratios = []
    rects = []
    areas = []
    for c, contour in enumerate(contours):
        rect = cv2.boundingRect(contour)
        asp_ratio = rect[2] / rect[3]
        area = rect[2] * rect[3]
        rects.append(rect)
        asp_ratios.append(asp_ratio)
        areas.append(area)

    asp_ratios = np.array(asp_ratios)
    rects = np.array(rects)
    areas = np.array(areas)
    # find the nearest aspect ratio rect with chessboard aspect ratio
    delta_ratios = asp_ratios - (chessboard_size[0] / chessboard_size[1])
    nearest_aspect_ratio_index = np.argmin(delta_ratios)
    # find the maximum area rect
    maximum_area_index = np.argmax(areas)
    # print('nearest aspect_ratio index {}, maximum area index {}'.format(nearest_aspect_ratio_index, maximum_area_index))
    # only satisfied two condition(aspect ratio and area), it would be considered as chessboard
    # if nearest_aspect_ratio_index == maximum_area_index:
    chessboard_rect = rects[maximum_area_index]
    # cv2.rectangle(contours_img, pt1=(chessboard_rect[0], chessboard_rect[1]),
    #               pt2=(chessboard_rect[0] + chessboard_rect[2], chessboard_rect[1] + chessboard_rect[3]),
    #               color=(255, 0, 255), thickness=2)
    # else:
    #     chessboard_rect = None
    #
    # cv2.namedWindow('contours image', cv2.WINDOW_GUI_EXPANDED)
    # cv2.imshow('contours image', contours_img)
    return chessboard_rect


def get_chessboard_corners(img, chessboard_size=(6, 9)):
    # rgb to gray
    w, h, c = img.shape
    if c > 1:
        # print('=> Only support input image with gray, it will be converted.')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        print('=> Starting to find the corners of chessboard.')

    # crop center region of image
    s = 5
    x1 = w // s
    x2 = (s - 1) * w // s
    y1 = h // s
    y2 = (s - 1 )* h // s
    cropped_img = img[x1:x2, y1:y2]
    # cropped_img = img

    cv2.namedWindow('cropped image', cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow('cropped image', cropped_img)
    # print('cropped image shape: {}'.format(cropped_img.shape))

    # # initial orb detector
    # orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=5,
    #                      edgeThreshold=10, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_FAST_SCORE, patchSize=30)

    chessboard_rect = find_chessboard(cropped_img, chessboard_size)

    if chessboard_rect is None:
        # print('=> Failed to find chessboard!')
        return None
    else:
        chessboard_img = cropped_img[chessboard_rect[1]:(chessboard_rect[1] + chessboard_rect[3]),
                         chessboard_rect[0]:(chessboard_rect[0] + chessboard_rect[2])]

        ret, corners = cv2.findChessboardCorners(chessboard_img, chessboard_size, None)

        if ret == True:
            # sub pixel corners
            corners = cv2.cornerSubPix(chessboard_img, corners, (5, 5), (-1, -1),
                                       criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))

            corners = corners.squeeze()[::-1]

            corners = corners + np.array([chessboard_rect[0], chessboard_rect[1]])

            corners = corners + np.array([y1, x1])

            # Draw and display the corners
            # cv2.drawChessboardCorners(cropped_img, chessboard_aspect_ratio, corners, ret)
            for i in range(len(corners)):
                cv2.circle(img, (int(corners[i, 0]), int(corners[i, 1])), 3, color=(255, 0, 255))
            cv2.namedWindow('chessboard image', cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow('chessboard image', img)

            corners = corners.reshape(-1, 1, 2)

            # generate object points
            # object_points = generate_object_points_in_3d(chessboard_aspect_ratio, 0.0025)

            # print('=> calculating camera matrices ...')
            # ret, camera_intrinsic_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera([object_points],
            #                                                                              corners,
            #                                                                              img.shape[::-1],
            #                                                                              None, None)
            # # rotation_matrix = cv2.Rodrigues(rvecs)
            # print('camera intrinsic matrix: {}'.format(camera_intrinsic_matrix))
            # print('camera distortion: {}'.format(distortion))
            # print('rotation : {}'.format(rvecs))
            # print('translation: {}'.format(tvecs))
            return corners
        else:
            print('=> Failed to find chessboard corners!')

        # rgb_chessboard_img = cv2.cvtColor(chessboard_img, cv2.COLOR_GRAY2BGR)
        # chessboard_keypoints = orb.detect(chessboard_img)
        # print("the number of keypoints in chessboard: {}".format(len(chessboard_keypoints)))
        #
        # cv2.drawKeypoints(rgb_chessboard_img, chessboard_keypoints, rgb_chessboard_img)
        # print("chessboard corners shape : {}".format(corners.shape))


def generate_object_points_in_3d(chesssboard_aspect_ratio=(6, 9), unit_size=0.025):
    obj_points = np.zeros((chesssboard_aspect_ratio[0] * chesssboard_aspect_ratio[1], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:chesssboard_aspect_ratio[0], 0:chesssboard_aspect_ratio[1]].T.reshape(-1, 2)
    obj_points = obj_points * unit_size
    return obj_points


def get_object2camera_matrix(zed, chessboard_size=(6, 9)):
    # generate object points
    object_points = generate_object_points_in_3d(chessboard_size, 0.025)
    # load camera matrix
    left_camera_intrinsic_matrix, left_distortion_vec = load_left_camera_params(zed)
    # calculate the transformation from object to camera
    while (cv2.waitKey(1) != 27):
        left_img, right_img = get_image_by_zed_camera(zed)

        chessboard_corners = get_chessboard_corners(left_img, chessboard_size)
        if chessboard_corners is None:
            print('=> Retrying ...')
            continue
        else:
            print('=> Calculating camera extrinsic matrix ...')
            retval, rvec, tvec = cv2.solvePnP(object_points, chessboard_corners,
                                              left_camera_intrinsic_matrix, left_distortion_vec)
            rotation_matrix, jacobi = cv2.Rodrigues(rvec)
            # print('rotation matric: {}'.format(rotation_matrix))
            # print('translation: {}'.format(tvec))
            cMo = np.vstack((np.hstack((rotation_matrix.reshape(3, 3), tvec.reshape(3, 1))), [0, 0, 0, 1]))
            print('object to camera matrix: {}'.format(cMo))
            return cMo


def get_end2base_matrix(server_addr):
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # connect to UR5
    tcp_socket.connect(server_addr)
    # acquire robot data
    print('=> Acquiring UR5 information ...')
    robot_data = tcp_socket.recv(1116)
    # acquire the end effector information
    # end_effector_inform = np.array(struct.unpack('!6d', robot_data[444:492]))
    end_effector_inform = np.array(struct.unpack('!6d', robot_data[588:636]))
    tvec = end_effector_inform[0:3].reshape(3, 1)
    rvec = end_effector_inform[3:6]

    rotation_matirx, jacobi = cv2.Rodrigues(rvec)

    print('End effector translation: {}'.format(tvec))
    print('End effector rotation vec: {}'.format(rvec))
    # print('End effector rotation matrix: {}'.format(rotation_matirx))
    # print('End effector state: {}'.format(end_effector_inform))
    bMe = np.vstack((np.hstack((rotation_matirx, tvec)), [0, 0, 0, 1]))
    print('End effector to base matrix: {}'.format(bMe))
    return bMe


def tcp_x(server_addr, step=0.05, interval=1):
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect(server_addr)
    data = tcp_socket.recv(1116)
    position = struct.unpack('!6d', data[444:492])
    data_pos = np.asarray(position)
    print(data_pos)
    command = "movel(p[%f,%f,%f,%f,%f,%f],a=0.2,v=0.2,t=0,r=0)\n" % (
        data_pos[0] + step, data_pos[1], data_pos[2], data_pos[3], data_pos[4],
        data_pos[5])
    tcp_socket.send(str.encode(command))
    time.sleep(interval)


def tcp_y(server_addr, step=0.05, interval=1):
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect(server_addr)
    data = tcp_socket.recv(1116)
    position = struct.unpack('!6d', data[444:492])
    data_pos = np.asarray(position)
    print(data_pos)
    command = "movel(p[%f,%f,%f,%f,%f,%f],a=0.2,v=0.2,t=0,r=0)\n" % (
        data_pos[0], data_pos[1] + step, data_pos[2], data_pos[3], data_pos[4],
        data_pos[5])
    tcp_socket.send(str.encode(command))
    time.sleep(interval)


def tcp_z(server_addr, step=0.05, interval=1):
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect(server_addr)
    data = tcp_socket.recv(1116)
    position = struct.unpack('!6d', data[444:492])
    data_pos = np.asarray(position)
    print(data_pos)
    command = "movel(p[%f,%f,%f,%f,%f,%f],a=0.2,v=0.2,t=0,r=0)\n" % (
        data_pos[0], data_pos[1] , data_pos[2] + step, data_pos[3], data_pos[4],
        data_pos[5])
    tcp_socket.send(str.encode(command))
    time.sleep(interval)


def tcp_roll(server_addr, speed=0.1, interval=1):
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect(server_addr)
    command = "speedj([0, 0, 0, 0, 0, {}],0.5,{})\n".format(speed, interval)
    tcp_socket.send(str.encode(command))
    time.sleep(1)


def tcp_pitch(server_addr, speed=0.1, interval=1):
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect(server_addr)
    command = "speedj([0, 0, 0, {}, 0, 0],0.5,{})\n".format(speed, interval)
    tcp_socket.send(str.encode(command))
    time.sleep(1)


def tcp_yaw(server_addr, speed=0.1, interval=1):
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect(server_addr)
    command = "speedj([0, 0, 0, 0, {}, 0],0.5,{})\n".format(speed, interval)
    tcp_socket.send(str.encode(command))
    time.sleep(1)


def calibration_planning(zed, server_addr, num=5):
    if num % 4 == 1 and num >= 5:
        print('=> start calibration position planning ...')
    else:
        print('The input number is error, please input number following such format (4n+1, n=1, 2, ...)!')
        exit(1)

    step_num = num // 4

    count = 0
    r_speed = 0.1
    r_interval = 0.2
    t_step = 0.03
    interval = 1

    initialization_file = 'UR5_initial_pos.txt'
    initial_pos = np.loadtxt(initialization_file, delimiter=',')


    for r in range(step_num):
        if r == 0:
            # center point
            print('=> {}th calibration for {} in total'.format(count + 1, num))
            cMo_file = os.path.join(result_dir, 'Object2CameraMatrix_{:02d}.txt'.format(count))
            bMe_file = os.path.join(result_dir, 'End2BaseMatrix_{:02d}.txt'.format(count))
            cMo = get_object2camera_matrix(zed, chessboard_size)
            bMe = get_end2base_matrix(server_addr)
            np.savetxt(cMo_file, cMo, fmt='%.6f', delimiter=',')
            np.savetxt(bMe_file, bMe, fmt='%.6f', delimiter=',')
            count += 1

        # left
        print('=> {}th calibration for {} in total'.format(count + 1, num))
        tcp_x(server_addr, t_step * (r + 1), interval)
        tcp_yaw(server_addr, r_speed * (r + 1), r_interval * (r + 1))
        cMo_file = os.path.join(result_dir, 'Object2CameraMatrix_{:02d}.txt'.format(count))
        bMe_file = os.path.join(result_dir, 'End2BaseMatrix_{:02d}.txt'.format(count))
        cMo = get_object2camera_matrix(zed, chessboard_size)
        bMe = get_end2base_matrix(server_addr)
        np.savetxt(cMo_file, cMo, fmt='%.6f', delimiter=',')
        np.savetxt(bMe_file, bMe, fmt='%.6f', delimiter=',')
        count += 1
        robot_arm_initialization(server_addr, initial_pos)


        # right
        print('=> {}th calibration for {} in total'.format(count + 1, num))
        tcp_x(server_addr, -t_step * (r + 1), interval)
        tcp_yaw(server_addr, -r_speed * (r + 1), r_interval * (r + 1))
        cMo_file = os.path.join(result_dir, 'Object2CameraMatrix_{:02d}.txt'.format(count))
        bMe_file = os.path.join(result_dir, 'End2BaseMatrix_{:02d}.txt'.format(count))
        cMo = get_object2camera_matrix(zed, chessboard_size)
        bMe = get_end2base_matrix(server_addr)
        np.savetxt(cMo_file, cMo, fmt='%.6f', delimiter=',')
        np.savetxt(bMe_file, bMe, fmt='%.6f', delimiter=',')
        count += 1
        robot_arm_initialization(server_addr, initial_pos)


        # upper
        print('=> {}th calibration for {} in total'.format(count + 1, num))
        tcp_z(server_addr, t_step * (r + 1), interval)
        tcp_pitch(server_addr, -r_speed * (r + 1), r_interval * (r + 1))
        cMo_file = os.path.join(result_dir, 'Object2CameraMatrix_{:02d}.txt'.format(count))
        bMe_file = os.path.join(result_dir, 'End2BaseMatrix_{:02d}.txt'.format(count))
        cMo = get_object2camera_matrix(zed, chessboard_size)
        bMe = get_end2base_matrix(server_addr)
        np.savetxt(cMo_file, cMo, fmt='%.6f', delimiter=',')
        np.savetxt(bMe_file, bMe, fmt='%.6f', delimiter=',')
        count += 1
        robot_arm_initialization(server_addr, initial_pos)


        # lower
        print('=> {}th calibration for {} in total'.format(count + 1, num))
        tcp_z(server_addr, -t_step * (r + 1), interval)
        tcp_pitch(server_addr, r_speed * (r + 1), r_interval * (r + 1))
        cMo_file = os.path.join(result_dir, 'Object2CameraMatrix_{:02d}.txt'.format(count))
        bMe_file = os.path.join(result_dir, 'End2BaseMatrix_{:02d}.txt'.format(count))
        cMo = get_object2camera_matrix(zed, chessboard_size)
        bMe = get_end2base_matrix(server_addr)
        np.savetxt(cMo_file, cMo, fmt='%.6f', delimiter=',')
        np.savetxt(bMe_file, bMe, fmt='%.6f', delimiter=',')
        count += 1
        robot_arm_initialization(server_addr, initial_pos)



def matlab_svd(mat):
    m, n = mat.shape[:2]
    U, sdiag, VH = np.linalg.svd(mat)
    #U, sdiag, VH = scipy.linalg.svd(mat, lapack_driver='gesvd')
    S = np.zeros((m, n))
    np.fill_diagonal(S, sdiag)
    V = VH.T.conj()
    return U, S, V


def AXXB_Solver(AA, BB):
    np.set_printoptions(precision=6)
    m, n = AA.shape[:2]
    n //= 4
    A = np.zeros((9*n, 9))
    b = np.zeros((9*n, 1))
    eye = np.eye(3)
    for i in range(n):
        Ra = AA[:3, 4*i:4*i+3]
        Rb = BB[:3, 4*i:4*i+3]
        A[9*i:9*i+9, :] = np.kron(Ra, eye) - np.kron(eye, Rb.T)
    u, s, v = matlab_svd(A)
    x = v[:, -1]
    R = x.reshape(3, 3)
    R = np.sign(np.linalg.det(R))/pow(abs(np.linalg.det(R)), 1/3) * R
    u, s, v = matlab_svd(R)
    R = np.matmul(u, v.T)
    if np.linalg.det(R) < 0:
        R = np.matmul(u, np.diag([1, 1, -1]), v.T)
    C = np.zeros((3*n, 3))
    d = np.zeros((3*n, 1))
    for i in range(n):
        C[3*i:3*i+3] = eye - AA[:3, 4*i:4*i+3]
        d[3*i:3*i+3] = (AA[:3, 4*i+3] - np.matmul(R, BB[:3, 4*i+3])).reshape(3, 1)
    #t = np.linalg.solve(C, d)
    t = np.linalg.lstsq(C, d)[0]
    return np.vstack([np.hstack([R, t]), np.array([0, 0, 0, 1])])


def robot_arm_initialization(server_addr,
        init_pos=[-0.24181421, 0.30435334, 0.81605341, -1.25381702, -0.62217599, 0.7471209]):
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect(server_addr)
    # data_pos = [-0.15747745, 0.44148789, 0.63479836, -1.24728887, -0.66890486, 0.88747505]
    data_pos = init_pos
    command = "movel(p[%f,%f,%f,%f,%f,%f],a=0.2,v=0.2,t=0,r=0)\n" % (
        data_pos[0], data_pos[1], data_pos[2], data_pos[3], data_pos[4],
        data_pos[5])
    tcp_socket.send(str.encode(command))
    time.sleep(2)


def set_robot_arm_initial_location(zed, chessboard_size, server_addr):
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect(server_addr)

    get_object2camera_matrix(zed, chessboard_size)

    data = tcp_socket.recv(1116)
    position = struct.unpack('!6d', data[444:492])
    initial_position = np.asarray(position)
    return initial_position


parser = argparse.ArgumentParser('Eye in hand calibration')
parser.add_argument('--data_num', type=int, default=9, help='the number of acquisition')
parser.add_argument('--result_dir', type=str, default='results', help='the results of calibration files')
args = parser.parse_args()

if __name__ == '__main__':
    # chessboard aspect_ratio
    chessboard_size = (6, 9)

    # initialize the connection with UR5
    server_ip = "192.168.1.3"
    server_port = 30003
    global server_addr
    server_addr = (server_ip, server_port)
    # robot instruction interval, to smooth the movement of UR5
    interval = 1

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
    else:
        print('=> starting to calculate object to camera matrix (cMo)')

    # Initial robot to a fixed position, where chessboard is located in the center of camera
    initialization_file = 'UR5_initial_pos.txt'
    # initial_pos = set_robot_arm_initial_location(zed, chessboard_size, server_addr)
    # np.savetxt(initialization_file, initial_pos, fmt='%0.6f', delimiter=',')
    initial_pos = np.loadtxt(initialization_file, delimiter=',')
    robot_arm_initialization(server_addr, initial_pos)

    acquisition_num = args.data_num
    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        print('=> result directory is not existed, it will be created.')
        os.makedirs(result_dir)

    # step = 0.003
    # for i in range(acquisition_num):
    #     print('=> {}th calibration for {} in total'.format(i + 1, acquisition_num))
    #
    #     if i == (acquisition_num//2):
    #         robot_arm_initialization(server_addr, initial_pos)
    #         step = -step
    #
    #     if i % 3 == 0:
    #         tcp_x(step=step, interval=interval)
    #     elif i % 3 == 1:
    #         tcp_y(step=step, interval=interval)
    #     else:
    #         tcp_z(step=step, interval=interval)
    #
    #     cMo_file = os.path.join(result_dir, 'Object2CameraMatrix_{:02d}.txt'.format(i))
    #     bMe_file = os.path.join(result_dir, 'End2BaseMatrix_{:02d}.txt'.format(i))
    #     cMo = get_object2camera_matrix(zed, chessboard_size)
    #     bMe = get_end2base_matrix(server_addr)
    #
    #     np.savetxt(cMo_file, cMo, fmt='%.6f', delimiter=',')
    #     np.savetxt(bMe_file, bMe, fmt='%.6f', delimiter=',')




    # load result files

    calibration_planning(zed, server_addr, acquisition_num)

    cMo_files = sorted(glob.glob(os.path.join(result_dir, 'Object2CameraMatrix_*.txt')))
    bMe_files = sorted(glob.glob(os.path.join(result_dir, 'End2BaseMatrix_*.txt')))
    eMc_file = os.path.join(os.path.abspath(os.curdir), 'Camera2EndMatrix.txt')

    num = acquisition_num
    AA = []
    BB = []

    for i in range(num):
        # cMo1 and bMe1
        cMo1 = np.loadtxt(cMo_files[i % acquisition_num], delimiter=',')
        bMe1 = np.loadtxt(bMe_files[i % acquisition_num], delimiter=',')

        # cMo2 and bMe2
        cMo2 = np.loadtxt(cMo_files[(i + 1) % acquisition_num], delimiter=',')
        bMe2 = np.loadtxt(bMe_files[(i + 1) % acquisition_num], delimiter=',')

        # A = np.linalg.inv(bMe2) * bMe1
        # B = cMo2 * np.linalg.inv(cMo1)
        A = np.linalg.inv(bMe1) * bMe2
        B = cMo1 * np.linalg.inv(cMo2)

        AA.append(A)
        BB.append(B)

    AA = np.array(AA).reshape((4, -1))
    BB = np.array(BB).reshape((4, -1))

    eMc = AXXB_Solver(AA, BB)
    print('Camera to end-effector matrix: {}'.format(eMc))
    # save camera to end-effector matrix file
    np.savetxt(eMc_file, eMc, fmt='%0.6f', delimiter=',')
    print('=> Successfully saved camera to end-effector matrix in {}'.format(eMc_file))


    # error verification
    total_errs = np.zeros((4, 4), dtype=np.float32)
    for i in range(acquisition_num):
        print('=> calculating the error between position {} and {}'.format(i, (i + 2) % acquisition_num))
        # cMo1 and bMe1
        cMo1 = np.loadtxt(cMo_files[i % acquisition_num], delimiter=',')
        bMe1 = np.loadtxt(bMe_files[i % acquisition_num], delimiter=',')

        # cMo2 and bMe2
        cMo2 = np.loadtxt(cMo_files[(i + 2) % acquisition_num], delimiter=',')
        bMe2 = np.loadtxt(bMe_files[(i + 2) % acquisition_num], delimiter=',')

        err = bMe1 * eMc * cMo1 - bMe2 * eMc * cMo2
        print('the error between position {} and {}: \n{}'.format(i, (i + 2) % acquisition_num, err))
        total_errs = total_errs + err

    average_err = total_errs / acquisition_num
    print("average errors: {}".format(average_err))

    zed.close()
