import os
import pyzed.sl as sl
import numpy as np
import pandas as pd
from struct import pack,unpack
import glob
import cv2
from shutil import copyfile

'''
NEAT pre-processing program:
    This program aims to transfer all of the svo files into NEAT dataset format
NEAT dataset format:
-train
|--sequence id {:04d}
|       |--img
|       |   |--left
|       |   |   |--{:06}.jpg
|       |   |   ...
|       |   |--right
|       |   |   |--{:06}.jpg
|       |   |   ...
|       |--{:04d}.svo
|       |--calibration.npz
|       |--groundtruth_rect.csv
|       |--groundtruth_bbox3d.csv
|       |--init_point_cloud.pcd
-test   
|--sequence id {:04d}
|       |--img
|       |   |--left
|       |   |   |--{:06}.jpg
|       |   |   ...
|       |   |--right
|       |   |   |--{:06}.jpg
|       |   |   ...
|       |--{:04d}.svo
|       |--calibration.npz
|       |--groundtruth_rect.csv
|       |--groundtruth_bbox3d.csv
|       |--init_point_cloud.pcd
'''

class NEAT_Loader(object):

    def __init__(self, root_dir, subset='train'):
        super(NEAT_Loader, self).__init__()
        assert subset in ['train', 'val', 'test'], 'Unknown subset.'
        self.dataset_name = 'NEAT'
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        self.subset = subset
        self.meta = {'resolution': '(1920, 1080)'}
        subset_dir = os.path.join(root_dir, subset)
        if os.path.exists(subset_dir):
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

    def preprocessing(self, input_dir, overwrite=False):
        if not os.path.exists(input_dir):
            print("=> The input directory is not existed.")
            exit(1)
        original_sequences = sorted([os.path.join(input_dir, sequence) for sequence in os.listdir(input_dir)])
        original_sequence_dirs = [sequence_dir for sequence_dir in original_sequences
                                  if os.path.isdir(sequence_dir)]
        for idx, sequence_dir in enumerate(original_sequence_dirs):
            if idx % 2:
                subset = 'train'
            else:
                subset = 'test'
            sequence_idx = (idx // 2) + 1
            output_sequence_dir = os.path.join(self.root_dir, subset, '{:04d}'.format(sequence_idx))
            if not os.path.exists(output_sequence_dir):
                print('=> the sequence {:04d} directory is not in {}, {}'.format(sequence_idx, subset, self.root_dir))
                print('\tit will be created...')
                os.makedirs(output_sequence_dir)

            # load groud-truth file
            groundtruth_file = os.path.join(sequence_dir, 'groundtruth_rect.csv')
            output_groundtruth_file = os.path.join(output_sequence_dir, 'groundtruth_rect.csv')
            df = pd.read_csv(groundtruth_file, index_col=False)
            print("=> The ground-truth file have been loaded successfully!")

            # start write down images and copy svo file to output sequence directory
            svo_file = os.path.join(sequence_dir, '{:04d}.svo'.format(idx+1))
            output_svo_file = os.path.join(output_sequence_dir, '{:04d}.svo'.format(sequence_idx))
            print('=> svo file({}) loading...'.format(svo_file))

            init_params = sl.InitParameters()
            init_params.svo_input_filename = str(svo_file)
            init_params.svo_real_time_mode = False
            init_params.coordinate_units = sl.UNIT.UNIT_MILLIMETER
            init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
            init_params.camera_fps = 15

            zed = sl.Camera()
            err = zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print('Failed to open ZED!!!')
                zed.close()
                exit(1)

            left_image = sl.Mat()
            right_image = sl.Mat()
            Point_cloud = sl.Mat()

            # save camera parameters
            left_intrinsic_matrix, right_intrinsic_matrix, rotation, translation = self._load_calibration_params(zed)
            left_camera_matrix, right_camera_matrix = self._camera_matrix_calc(left_intrinsic_matrix,
                                                                         right_intrinsic_matrix,
                                                                         rotation, translation)
            output_calibration_npz_file = os.path.join(output_sequence_dir, 'calibration.npz')
            np.savez(output_calibration_npz_file,
                     left_camera_matrix=left_camera_matrix,
                     right_camera_matrix=right_camera_matrix)

            # Runtime parameters
            runtime_params = sl.RuntimeParameters()
            runtime_params.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD

            # Sequence length
            sequence_len = zed.get_svo_number_of_frames()

            # output left and right image directory
            output_left_img_dir = os.path.join(output_sequence_dir, 'img', 'left')
            if not os.path.exists(output_left_img_dir):
                print('=> the left image directory is not in sequence {:04d}, subset {}'.format(sequence_idx, subset))
                print('=> it will be created...')
                os.makedirs(output_left_img_dir)
            output_right_img_dir = os.path.join(output_sequence_dir, 'img', 'right')
            if not os.path.exists(output_right_img_dir):
                print('=> the right image directory is not in in sequence {:04d}, subset {}'.format(sequence_idx, subset))
                print('=> it will be created...')
                os.makedirs(output_right_img_dir)

            init_point_cloud_file = os.path.join(output_sequence_dir, 'init_point_cloud.pcd')

            while 1:
                frame = zed.get_svo_position() + 1
                if frame <= sequence_len - 1:
                    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                        print("--Frame: {}/{}".format(frame, sequence_len))
                        left_img_path = os.path.join(output_left_img_dir, '{:06d}.jpg'.format(frame))
                        right_img_path = os.path.join(output_right_img_dir, '{:06d}.jpg'.format(frame))
                        if os.path.exists(left_img_path) and os.path.join(right_img_path) and not overwrite:
                            print('-- skip to write down frame {}'.format(frame))
                            continue
                        zed.retrieve_image(left_image, sl.VIEW.VIEW_LEFT)
                        zed.retrieve_image(right_image, sl.VIEW.VIEW_RIGHT)
                        print("--Successfully retrieved frame {} of {}".format(frame, svo_file.split('/')[-1]))
                        cv2.imwrite(left_img_path, left_image.get_data())
                        cv2.imwrite(right_img_path, right_image.get_data())

                        if frame == 1 and not os.path.exists(init_point_cloud_file):
                            zed.retrieve_measure(Point_cloud, sl.MEASURE.MEASURE_XYZRGBA)
                            point_cloud = Point_cloud.get_data()
                            height, width, channel = point_cloud.shape
                            print('=> Point cloud data shape: {}'.format(point_cloud.shape))
                            left_roi = eval(df.loc[frame, 'left roi'])
                            offset = 20
                            lx = left_roi[0] - offset
                            if lx < 0:
                                lx = 0
                            ly = left_roi[1] - offset
                            if ly < 0:
                                ly = 0
                            rx = left_roi[0] + left_roi[2] + offset
                            if rx > width:
                                rx = width - 1
                            ry = left_roi[1] + left_roi[3] + offset
                            if ry > height:
                                ry = height - 1
                            stride = 2
                            reduced_point_cloud = np.squeeze(point_cloud[ly:ry:stride, lx:rx:stride].reshape(1, -1, 4))
                            print('=> Point cloud data shape after reduction: {}'.format(reduced_point_cloud.shape))
                            delete_index = []
                            for index, point in enumerate(reduced_point_cloud):
                                if np.isnan(point[0]) or np.isinf(point[0]) or np.isnan(point[3]):
                                    delete_index.append(index)
                            reduced_point_cloud = np.delete(reduced_point_cloud, delete_index, 0)
                            print('=> Point cloud data shape excluded nan and inf values: {}'.format(reduced_point_cloud.shape))
                            self._point_cloud_numpy_2_pcd(reduced_point_cloud, init_point_cloud_file)
                            print('=> Initial point cloud have been writen down!')

                else:
                    print('=> sequence {:04d} in {} subset have been writen down!'.format(sequence_idx, subset))
                    print('=> copy {} to {}...'.format(svo_file, output_svo_file))
                    copyfile(svo_file, output_svo_file)
                    print('=> svo copy is done!')
                    print('=> copy {} to {}...'.format(groundtruth_file, output_groundtruth_file))
                    copyfile(groundtruth_file, output_groundtruth_file)
                    print('=> ground-truth file copy is done!\n')
                    break

    def _point_cloud_numpy_2_pcd(self, data, output_path):
        width, channel = data.shape
        if width > 0:
            if channel != 4:
                print('the channel of input data is not 4.')
                exit(1)
        else:
            print('the width of input data is none')

        print('==> begin to write down pcd file ...')
        if os.path.exists(output_path):
            os.remove(output_path)

        Output_Data = open(output_path, 'a')
        Output_Data.write('# .PCD v0.7 - Point Cloud Data file format\n'
                          'VERSION 0.7\n'
                          'FIELDS x y z rgb\n'
                          'SIZE 4 4 4 4\n'
                          'TYPE F F F U\n'
                          'COUNT 1 1 1 1\n')
        string = 'WIDTH ' + str(data.shape[0]) + '\n'
        Output_Data.write(string)
        Output_Data.write('HEIGHT 1\n'
                          'VIEWPOINT 0 0 0 1 0 0 0\n')
        string = 'POINTS ' + str(data.shape[0]) + '\n'
        Output_Data.write(string)
        Output_Data.write('DATA ascii\n')
        for i in range(data.shape[0]):
            rgba = pack('f', data[i][3])
            rgb = self._rgba2rgb(rgba)
            string = pack('4B', rgb[0], rgb[1], rgb[2], 0)
            f_rgb = unpack('f', string)
            string = '{:.6f}'.format(data[i][0] / 100) + ' ' + '{:.6f}'.format(data[i][1] / 100) + ' ' \
                     + '{:.6f}'.format(data[i][2] / 100) + ' ' + str(f_rgb[0]) + '\n'
            # string = '{:.6f}'.format(data[i][0]/1000) + ' ' + '{:.6f}'.format(data[i][1]/1000) + ' ' \
            #          + '{:.6f}'.format(data[i][2]/1000) + ' ' + str(f_rgb[0]) + '\n'
            Output_Data.write(string)
        Output_Data.close()
        return

    def _rgba2rgb(self, rgba):
        norm_rgba = [byte / 255 for byte in rgba]
        bg_color = 1
        r = bg_color * (1 - norm_rgba[3]) + norm_rgba[0] * norm_rgba[3]
        g = bg_color * (1 - norm_rgba[3]) + norm_rgba[1] * norm_rgba[3]
        b = bg_color * (1 - norm_rgba[3]) + norm_rgba[2] * norm_rgba[3]
        rgb = [int(r * 255), int(g * 255), int(b * 255)]
        return rgb

    def _load_calibration_params(self, camera):
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

    def _camera_matrix_calc(self, left_intrinsic_matrix, right_intrinsic_matrix, r_mat, t):
        # Set left camera as the reference
        left_rotation = np.eye(3)
        left_translation = np.zeros((3, 1))
        left_r_t = np.eye(4)

        r_t = np.concatenate((np.concatenate((r_mat, t), axis=1),
                              np.array([[0, 0, 0, 1]])), axis=0)
        # print("r_t: {}".format(r_t))
        # 为了计算right_r_t，因此我们将所有*_r_t矩阵升到4x4维空间
        right_r_t = np.matmul(np.linalg.inv(r_t), left_r_t)
        # 之后为了计算left_camera_matrix & right_camera_matrix，我们又将其降到3x4维空间
        left_r_t = left_r_t[:3, :]
        right_r_t = right_r_t[:3, :]
        # print("left_r_t: {}".format(left_r_t))
        # print("right_r_t: {}".format(right_r_t))
        left_camera_matrix = np.matmul(left_intrinsic_matrix, left_r_t)
        right_camera_matrix = np.matmul(right_intrinsic_matrix, right_r_t)
        print("left_camera_matrix: {}".format(left_camera_matrix))
        print("right_camera_matrix: {}".format(right_camera_matrix))
        return left_camera_matrix, right_camera_matrix

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
        self.left_truncation = [int(data) for data in data_frame['left truncation']]
        self.right_truncation = [int(data) for data in data_frame['right truncation']]
    pass

    def __getitem__(self, index):
        left_img_file = self.left_img_files[index]
        right_img_file = self.right_img_files[index]
        left_roi = self.left_annotation[index]
        right_roi = self.right_annotation[index]
        left_truncation = self.left_truncation[index]
        right_truncation = self.right_truncation[index]
        return left_img_file, left_roi, right_img_file, right_roi, left_truncation, right_truncation


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


if __name__ == '__main__':
    NEAT = NEAT_Loader('E:\\NEAT', 'train')
    # NEAT.preprocessing('E:\\Non-cooperative object tracking', False)
    for s, sequence in enumerate(NEAT):
        print("sequence: %d" % s)
        print("sequence length: %d" % sequence.__len__())
        print("sequence path: %s" % NEAT.sequence_directories[s])
        for f, (left_img_file, left_roi, right_img_file, right_roi, left_truncation, right_truncation) in enumerate(sequence):
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
            key = cv2.waitKey(2)
            if key == ord('n'):
                break
            elif key == 27:
                exit()


