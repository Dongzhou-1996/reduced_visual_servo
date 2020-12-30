import numpy as np
import os
import cv2
from KCF.kcftracker import KCFTracker
import EyeInHand_calibration.EyeInHand_calibration as EIH
import pyzed.sl as sl
import time
import arm_control as ac
import socket
import threading
import KCF.kalman_filter_3d as kf3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import struct


class RobotControl:
    def __init__(self):
        arm_server_addr = ("192.168.1.3", 30003)
        self.sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sk.connect(arm_server_addr)
        self.sk.setblocking(0)

        self.shutdown_lock = threading.Lock()
        self.shutdown_flag = False

        self.info_lock = threading.Lock()
        self.info_q = None
        self.info_qv = None
        self.info_pos = None
        self.info_end = None


        self.cmd_sem = threading.Semaphore(value=0)
        self.cmd_lock = threading.Lock()
        self.cmd_queue = []

    def thread_read(self):
        print("read thread started.")
        nremains = 1116
        bufs = []
        while True:
            try:
                buf = self.sk.recv(nremains)
                if len(buf) != 0:
                    bufs.append(buf)
                    nremains -= len(buf)

            except Exception:
                pass

            if nremains == 0:
                pkt = bufs[0]
                for i_buf in bufs[1:]:
                    pkt += i_buf
                #print(pkt)
                q = np.asarray(struct.unpack('!6d', pkt[252:300]))
                qv = np.asarray(struct.unpack('!6d', pkt[300:348]))
                pos = np.asarray(struct.unpack('!6d', pkt[444:492]))
                end_effector_inform = np.array(struct.unpack('!6d', pkt[588:636]))

                self.info_lock.acquire()
                self.info_q = q
                self.info_qv = qv
                self.info_pos = pos
                self.info_end = end_effector_inform
                self.info_lock.release()

                bufs.clear()
                nremains = 1116

            # time.sleep(0.01)

            self.shutdown_lock.acquire()
            shutdown_flag = self.shutdown_flag
            self.shutdown_lock.release()
            if shutdown_flag:
                break
            pass


        print("read thread closed.")
        pass

    def thread_write(self):
        print("write thread started")
        while True:
            has_cmd = self.cmd_sem.acquire(timeout=0.5)
            if has_cmd:

                self.cmd_lock.acquire()
                cmd = self.cmd_queue.pop(0)
                self.cmd_lock.release()
                #print("cmd:" + cmd)
                self.sk.send(cmd.encode())
                pass
            self.shutdown_lock.acquire()
            shutdown_flag = self.shutdown_flag
            self.shutdown_lock.release()
            if shutdown_flag:
                break
        print("write thread closed.")


    def start(self):
        th1 = threading.Thread(target=self.thread_write)
        th1.start()
        th2 = threading.Thread(target=self.thread_read)
        th2.start()

        th1.join()
        th2.join()
        print('robot control thread closed.')


    def cleanup(self):
        self.sk.close()

    def cmd(self, c: str):
        self.cmd_lock.acquire()
        self.cmd_queue.append(c)
        self.cmd_sem.release()
        self.cmd_lock.release()

    def get_info(self):
        self.info_lock.acquire()
        q = self.info_q
        qv = self.info_qv
        pos = self.info_pos
        self.info_lock.release()
        return q, qv, pos

    def get_arm_info(self):
        self.info_lock.acquire()
        q = self.info_q
        qv = self.info_qv
        pos = self.info_pos
        end = self.info_end
        self.info_lock.release()
        tvec = end[0:3].reshape(3, 1)
        rvec = end[3:6]

        rotation_matirx, jacobi = cv2.Rodrigues(rvec)

        bMe = np.vstack((np.hstack((rotation_matirx, tvec)), [0, 0, 0, 1]))

        return bMe, q, qv, pos


    def shutdown(self):
        self.shutdown_lock.acquire()
        self.shutdown_flag = True
        self.shutdown_lock.release()


class MyControl:
    def __init__(self, xrc: RobotControl):
        self._rc = xrc

        self._pelock = threading.Lock()
        self._pe = None
        self._shutdown_lock = threading.Lock()
        self._shutdown_flag = False

    def _test_ctrl5x(self, pe, l=0.3, dt=0.08):
        pe = pe.reshape(-1, 1).squeeze()
        i = 0
        FLAG = True
        while FLAG:
            # print('i: {}'.format(i))
            i = i + 1

            while True:
                qt, info_qv, pt = self._rc.get_info()
                if qt is None:
                    time.sleep(0.001)
                    continue
                break


            pe1 = np.array([pe[0], pe[1], pe[2]])
            pt1 = np.array([pt[0], pt[1], pt[2]])
            pez = ac.posture(pe1, pt1)
            thetap = ac.position(pe1, qt, pez, l)

            qv = thetap - qt
            if (np.max(qv) > 0.5):
                qv = 0.3 * qv / np.linalg.norm(qv)
            qv = qv.reshape(-1, 1)
            qvp = ac.PZJC(qt)
            qvp = qvp.reshape(-1, 1)

            qve = qv + qvp

            qa = np.linalg.norm(qvp) + 0.5  # qvp越大越接近碰撞，此时关节速度也相应增大
            command = ac.ctrl_speedj(qve, qa, dt)
            self._rc.cmd(command)

            qt, info_qv, pt = self._rc.get_info()

            Tt = ac.ZJ(qt)
            rvt = ac.m2rv(Tt[0:3, 0:3])
            zx = ac.rv2eu(pez) - ac.rv2eu(rvt)
            # print('zx=', zx)
            if np.linalg.norm(zx) < 0.1:
                FLAG = False

    def set_pe(self, pe):
        self._pelock.acquire()
        self._pe = pe
        self._pelock.release()

    def start(self):
        while True:
            self._pelock.acquire()
            pe = self._pe
            self._pelock.release()
            if pe is not None:
                self._test_ctrl5x(pe)
            else:
                time.sleep(0.001)

            self._shutdown_lock.acquire()
            shutdown = self._shutdown_flag
            self._shutdown_lock.release()
            if shutdown:
                break

    def shutdown(self):
        self._shutdown_lock.acquire()
        self._shutdown_flag = True
        self._shutdown_lock.release()


class Multi_Thread(threading.Thread):
    def __init__(self, func, args=[]):
        super(Multi_Thread, self).__init__()
        arm_server_addr = ("192.168.1.3", 30003)
        tcp_socket_ctrl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket_ctrl.connect(arm_server_addr)
        self.tcp_socket_ctrl = tcp_socket_ctrl

        self.func = func

        self.args = args
        self.result = []
        self._running = True

    def input(self, args=[]):
        self.args = args

    def run(self):
        self.result = self.func(self.tcp_socket_ctrl, *self.args)

    def terminate(self):
        self._running = False

    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None


def get_arm_information(tcp_socket):
    # global q, qv, pos, bMe
    # acquire robot data
    print('=> Acquiring UR5 information ...')
    robot_data = tcp_socket.recv(1116)
    # acquire the end effector information
    # end_effector_inform = np.array(struct.unpack('!6d', robot_data[444:492]))
    end_effector_inform = np.array(struct.unpack('!6d', robot_data[588:636]))
    q = np.asarray(struct.unpack('!6d', robot_data[252:300]))
    qv = np.asarray(struct.unpack('!6d', robot_data[300:348]))
    pos = np.asarray(struct.unpack('!6d', robot_data[444:492]))

    tvec = end_effector_inform[0:3].reshape(3, 1)
    rvec = end_effector_inform[3:6]

    rotation_matirx, jacobi = cv2.Rodrigues(rvec)

    # print('End effector translation: {}'.format(tvec))
    # print('End effector rotation vec: {}'.format(rvec))
    # print('End effector rotation matrix: {}'.format(rotation_matirx))
    # print('End effector state: {}'.format(end_effector_inform))
    bMe = np.vstack((np.hstack((rotation_matirx, tvec)), [0, 0, 0, 1]))
    # print('End effector to base matrix: {}'.format(bMe))
    # print('q matrix: {}'.format(q))
    # print('qv matrix: {}'.format(qv))
    # print('pos matrix: {}'.format(pos))
    return bMe, q, qv, pos

def load_eMc_matrix(file_path=''):
    if os.path.exists(file_path):
        print('=> loading camera to end-effector matrix')
        eMc = np.loadtxt(file_path, delimiter=',')
    else:
        print('=> The path of eMc file is not existed, please have a check!')
        exit(1)
    return eMc


def tracklet_display(ax, tracklet, color='r', size=10, alpha=0.5, marker='o', label=''):
    x = tracklet[:, 0]
    y = tracklet[:, 1]
    z = tracklet[:, 2]
    ax.plot(x, y, z, c=color, alpha=alpha, marker=marker, label=label)
    # ax.scatter(x, y, z, s=size, c=color, alpha=alpha, marker=marker)
    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('z', fontdict={'size': size, 'color': 'blue'})
    ax.set_ylabel('y', fontdict={'size': size, 'color': 'blue'})
    ax.set_xlabel('X', fontdict={'size': size, 'color': 'blue'})
    return




if __name__ == '__main__':
    # initial camera
    init_params = sl.InitParameters()
    init_params.svo_real_time_mode = True
    init_params.coordinate_units = sl.UNIT.UNIT_MILLIMETER
    init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
    init_params.camera_fps = 30
    zed = sl.Camera()

    rc = RobotControl()
    rcthread = threading.Thread(target=rc.start)
    rcthread.start()

    armctrl = MyControl(rc)
    acth = threading.Thread(target=armctrl.start)
    acth.start()


    # initialize robot arm
    # EIH.robot_arm_initialization(server_addr=server_addr)

    # load camera to end-effector matrix
    eMc_file = os.path.join('EyeInHand_calibration', 'Camera2EndMatrix1.txt')
    eMc = load_eMc_matrix(eMc_file)

    KF3D = kf3d.KF3D(init_params.camera_fps)

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print('Failed to open ZED!!!')
        zed.close()
        exit(1)
    else:
        print('=> starting to visual servo!')

    # initialize 2D tracker
    kcf = KCFTracker(True, True, True)

    object_name = 'asteroid_2'

    # object point in base coordinate file
    object_points_dir = os.path.join(os.path.abspath(os.curdir), 'tracklets', object_name)
    if not os.path.exists(object_points_dir):
        print('=> object points directory is not existed! It will be created ...')
        os.makedirs(object_points_dir)
    else:
        print('=> object points directory is existed!')
    object_points_base_file = os.path.join(object_points_dir, 'object_points_in_base.txt')
    object_points_img_file = os.path.join(object_points_dir, 'object_points_in_image.txt')
    object_points_end_file = os.path.join(object_points_dir, 'object_points_in_end.txt')

    object_points_base = []
    object_points_2d = []
    object_points_end_kf = []

    # tracking image records
    tracking_results_dir = os.path.join(os.path.abspath(os.curdir), 'tracking_results', object_name)
    if not os.path.exists(tracking_results_dir):
        print('=> tracking result directory is not existed! It will be created ...')
        os.makedirs(tracking_results_dir)
    else:
        print('=> tracking result directory is existed!')


    initialization = 1
    thread_init = True

    global obj_point_base
    global obj_point_end_kf

    frame = 0
    while(cv2.waitKey(3) != 27):
        frame += 1
        print('=> Frame: {:06d}'.format(frame))
        left_image = sl.Mat()
        Point_cloud = sl.Mat()
        # Runtime parameters
        runtime_params = sl.RuntimeParameters()
        runtime_params.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD
        while 1:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # print("=> Successfully retrieved image! ")
                zed.retrieve_image(left_image, sl.VIEW.VIEW_LEFT)
                zed.retrieve_measure(Point_cloud, sl.MEASURE.MEASURE_XYZ)
                break
        left_img = left_image.get_data()
        point_cloud = Point_cloud.get_data()
        # print('Point cloud shape: {}'.format(point_cloud.shape))
        # print('img shape: {}'.format(left_img.shape))
        height, width, channel = left_img.shape

        # initialize tracker
        if initialization == 1:
            cv2.namedWindow('visual servo initialization', cv2.WINDOW_GUI_EXPANDED)
            init_roi = cv2.selectROI('visual servo initialization', left_img,
                                     showCrosshair=True, fromCenter=False)
            print('initial roi: {}'.format(init_roi))
            if init_roi == (0, 0, 0, 0):
                print('Failed to initialize tracker, please try again.')
                initialization = 1
            else:
                initialization = 0
                init_frame = left_img.copy()
                cv2.rectangle(init_frame, (init_roi[0], init_roi[1]),
                              (init_roi[0] + init_roi[2], init_roi[1] + init_roi[3]),
                              color=(0, 255, 0), thickness=2)
                # acquire object points
                left_roi = init_roi
                center_point = (left_roi[0] + left_roi[2] // 2,
                                left_roi[1] + left_roi[3] // 2)
                bottom_left = (init_roi[0], init_roi[1] + init_roi[3])
                top_left = (init_roi[0], init_roi[1])
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

                stride = 5
                # print('point cloud value at (500, 500): {}'.format(point_cloud[500][500]))
                reduced_point_cloud = np.squeeze(point_cloud[ly:ry:stride, lx:rx:stride].reshape(1, -1, 4))
                object_point = np.nanmean(reduced_point_cloud, axis=0)

                object_point = object_point[:3].reshape(3, 1)
                object_point = np.vstack((object_point, [1]))
                # print('object point shape: {}'.format(object_point.shape))
                # print('object point in camera coordinate: {}'.format(object_point))

                obj_point_end = np.matmul(eMc, object_point)

                obj_point_end = obj_point_end / 1000
                KF3D.initialize(obj_point_end[0], obj_point_end[1], obj_point_end[2])
                obj_point_end = obj_point_end[:3].reshape(1, 3).squeeze()
                obj_point_end_kf = np.array([obj_point_end[0], obj_point_end[1], obj_point_end[2], 1]).reshape(4, 1)
                print('Initial object point in end effector: {}'.format(obj_point_end_kf))

                object_points_end_kf.append(obj_point_end_kf)

                # transfer object point from camera coordinate system to base coordinate system
                bMe, _, _, _ = rc.get_arm_info()
                # obj_point_base = obj_point_end
                obj_point_base = np.matmul(bMe, obj_point_end_kf).reshape(1, -1)
                print('Initial object point in base: {}'.format(obj_point_base))
                object_points_base.append(obj_point_base)
                object_points_2d.append(center_point)

                cv2.putText(init_frame, '[{:0.2f}, {:0.2f}, {:0.2f}]'.format(obj_point_base[0, 0],
                                                                           obj_point_base[0, 1],
                                                                           obj_point_base[0, 2]),
                            top_left, cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 0), 2)
                cv2.circle(left_img, center_point, 1, (0, 0, 255), 2)
                cv2.imshow('visual servo initialization', init_frame)

                kcf.init(list(init_roi), left_img)
                cv2.destroyWindow('visual servo initialization')

        else:
            # start tracking
            start_time = time.time()
            KF3D.predict()
            tracking_result = kcf.update(left_img)
            boundingbox = list(map(int, tracking_result))
            cv2.rectangle(left_img, (boundingbox[0], boundingbox[1]),
                          (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]),
                          color=(0, 255, 0), thickness=2)

            center_point = (boundingbox[0] + boundingbox[2]//2,
                            boundingbox[1] + boundingbox[3]//2)
            bottom_left = (boundingbox[0], boundingbox[1] + boundingbox[3])
            top_left = (boundingbox[0], boundingbox[1])

            # acquire object points
            left_roi = boundingbox
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

            stride = 5
            # print('point cloud value at (500, 500): {}'.format(point_cloud[500][500]))
            reduced_point_cloud = np.squeeze(point_cloud[ly:ry:stride, lx:rx:stride].reshape(1, -1, 4))
            object_point = np.nanmean(reduced_point_cloud, axis=0)

            object_point = object_point[:3].reshape(3, 1)
            object_point = np.vstack((object_point, [1]))
            # print('object point shape: {}'.format(object_point.shape))
            # print('object point in camera coordinate: {}'.format(object_point))

            obj_point_end = np.matmul(eMc, object_point)
            obj_point_end = obj_point_end / 1000
            obj_point_end = obj_point_end[:3].reshape(1, 3).squeeze()

            obj_point_end_kf = KF3D.update(True, obj_point_end[0], obj_point_end[1], obj_point_end[2])
            obj_point_end_kf = np.array([float(obj_point_end_kf[0]), float(obj_point_end_kf[1]), float(obj_point_end_kf[2]), 1]).reshape(4, 1)
            object_points_end_kf.append(obj_point_end_kf)

            # transfer object point from camera coordinate system to base coordinate system
            bMe, q, qv, pos = rc.get_arm_info()
            # obj_point_base = obj_point_end
            obj_point_base = np.matmul(bMe, obj_point_end_kf).reshape(1, -1)

            print('object point in base: \n {}'.format(obj_point_base))
            object_points_base.append(obj_point_base)
            object_points_2d.append(center_point)

            armctrl.set_pe(obj_point_base)

            end_time = time.time()
            fps = 1 / (end_time - start_time)

            cv2.putText(left_img,
                        '[{:0.2f}, {:0.2f}, {:0.2f}]'.format(obj_point_base[0, 0],
                                                             obj_point_base[0, 1],
                                                             obj_point_base[0, 2]),
                        bottom_left, cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 2)
            # cv2.putText(left_img, '[{:0.2f}, {:0.2f}, {:0.2f}]'.format(object_point[0][0],
            #                                                            object_point[1][0],
            #                                                            object_point[2][0]),
            #             top_left, cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 0), 2)
            # cv2.putText(left_img, 'FPS: {:0.2f}'.format(fps),
            #             (0, height), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 0), 2)
            cv2.circle(left_img, center_point, 1, (0, 0, 255), 2)
            cv2.namedWindow('tracking result displaying', cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow('tracking result displaying', left_img)

            if frame % 5 == 0:
                img_path = os.path.join(tracking_results_dir, '{:06d}.jpg'.format(frame))
                cv2.imwrite(img_path, left_img)



    object_points_base = np.array(object_points_base).reshape(-1, 4).squeeze()
    object_points_end_kf = np.array(object_points_end_kf).reshape(-1, 4).squeeze()
    object_points_2d = np.array(object_points_2d).reshape(-1, 2).squeeze()

    # object_points_end_kf = np.array(object_points_end_kf).reshape(-1, 3)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.ion()
    # tracklet_display(ax, object_points_end, color='r', size=10, alpha=0.5, marker='o', label='tracklet')
    # tracklet_display(ax, object_points_end_kf, color='b', size=10, alpha=0.5, marker='*', label='tracklet_kf')
    # plt.legend()
    # plt.ioff()/
    # plt.show()
    np.savetxt(object_points_end_file, object_points_end_kf, fmt='%0.6f', delimiter=',')
    np.savetxt(object_points_base_file, object_points_base, fmt='%0.6f', delimiter=',')
    np.savetxt(object_points_img_file, object_points_2d, fmt='%0.6f', delimiter=',')
    print('=> Successfully write down object points in end-effector!!!')


    armctrl.shutdown()
    print("wait for armctrl thread...")
    # acth.join()
    acth.join()

    rc.shutdown()
    print("wait for rc thread...")
    rcthread.join()
    rc.cleanup()
    print("rc thread term.")

    exit(1)



