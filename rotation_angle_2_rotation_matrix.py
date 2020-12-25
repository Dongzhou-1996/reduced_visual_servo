import numpy as np
import math
import EyeInHand_calibration.EyeInHand_calibration as EIH

def rotation_angle_2_rotation_matrix(alpha, beta, gamma):
    R = np.array([[math.cos(alpha) * math.cos(gamma) - math.sin(alpha) * math.sin(gamma) * math.cos(beta),
                   math.sin(alpha) * math.cos(beta) + math.cos(alpha) * math.sin(gamma) * math.cos(beta),
                   math.sin(gamma) * math.sin(beta)],
                  [-math.cos(alpha) * math.sin(gamma) - math.sin(alpha) * math.cos(gamma) * math.cos(beta),
                   -math.sin(alpha) * math.sin(gamma) + math.cos(alpha) * math.cos(beta),
                   math.cos(gamma) * math.sin(beta)],
                  [math.sin(alpha) * math.sin(beta), -math.cos(alpha) * math.sin(beta), math.cos(beta)]])
    return R

def rotation_matrix_2_angle(matrix):
    beta = math.acos(matrix[2, 2]) * 57.3
    alpha = math.asin(matrix[2, 0] / math.sin(beta)) * 57.3
    gamma = math.asin(matrix[0, 2] / math.sin(beta)) * 57.3
    return alpha, beta, gamma

if __name__ == '__main__':
    alpha = 25.56 * math.pi / 180
    beta = 26.67 * math.pi / 180
    gamma = -26.17 * math.pi / 180
    rotation_matrix = rotation_angle_2_rotation_matrix(alpha, beta, gamma)
    print('rotation matrix: \n{}'.format(rotation_matrix))

    # server_ip = "192.168.1.3"
    # server_port = 30003
    # global server_addr
    # server_addr = (server_ip, server_port)
    # EIH.get_end2base_matrix(server_addr)

    eMc = np.loadtxt('EyeInHand_calibration/Camera2EndMatrix3.txt', delimiter=',')
    R = eMc[:3, :3]
    print("Rotation matrix: {}".format(R))
    alpha, beta, gamma = rotation_matrix_2_angle(R)
    print('alpha: {}, beta: {}, gamma: {}'.format(alpha, beta, gamma))