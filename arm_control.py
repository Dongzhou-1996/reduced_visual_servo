import numpy as np
import socket
import time
import struct
import math
HOST = "192.168.1.3"
PORT = 30003
# tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# tcp_socket.connect((HOST, PORT))


def getpos():
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect((HOST, PORT))
    data = tcp_socket.recv(1116)
    pos = struct.unpack('!6d', data[444:492])
    #tcp_socket.close()
    return np.asarray(pos)

def getpos1(tcp_socket):
    data = tcp_socket.recv(1116)
    pos = struct.unpack('!6d', data[444:492])
    #tcp_socket.close()
    return np.asarray(pos)

def getq():
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect((HOST, PORT))
    data = tcp_socket.recv(1116)
    q = struct.unpack('!6d', data[252:300])
    return np.asarray(q)

def getq1(tcp_socket):
    data = tcp_socket.recv(1116)
    q = struct.unpack('!6d', data[252:300])
    return np.asarray(q)

def getqv():
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect((HOST, PORT))
    data = tcp_socket.recv(1116)
    qv = struct.unpack('!6d', data[300:348])
    #tcp_socket.close()
    return np.asarray(qv)

def getqv1(tcp_socket):
    data = tcp_socket.recv(1116)
    qv = struct.unpack('!6d', data[300:348])
    #tcp_socket.close()
    return np.asarray(qv)

def ctrl_movej(q,ja,jv,t,r):
#关节空间内线性
    command = 'movej(['+ str(q[0]) + ','+ str(q[1]) + ','+ str(q[2]) + ','+ str(q[3]) + ','+ str(q[4]) + ','+\
               str(q[5]) + '],a=' + str(ja) + ',v=' + str(jv) + ',t=' + str(t) + ',r=' + str(r) + ')\n'
    return command

def ctrl_speedj(qv,a,t):
#关节空间内线性
    command = 'speedj([' + str(qv[0,0]) + ',' + str(qv[1,0]) + ',' + str(qv[2,0]) + ',' + str(qv[3,0]) + ',' +\
              str(qv[4,0]) + ',' +str(qv[5,0]) + '],a=' + str(a) + ',t=' + str(t) + ')\n'
    return command

def speedj(qv,a,t):
    tcp_command = "speedj([%f,%f,%f,%f,%f,%f],a=%f,t=%f)\n" % (
        qv[0], qv[1], qv[2], qv[3], qv[4], qv[5], a, t)
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect((HOST, PORT))
    tcp_socket.send(str.encode(tcp_command))  #利用字符串的encode方法编码成bytes，默认为utf-8类型
    tcp_socket.close()

def speedj1(tcp_socket,qv,a,t):
    tcp_command = "speedj([%f,%f,%f,%f,%f,%f],a=%f,t=%f)\n" % (
        qv[0], qv[1], qv[2], qv[3], qv[4], qv[5], a, t)
    tcp_socket.send(str.encode(tcp_command))  #利用字符串的encode方法编码成bytes，默认为utf-8类型


def askew(v):  # 三维实向量的反对称阵
    v = v.reshape(-1, 1)
    v0 = v[0, 0]
    v1 = v[1, 0]
    v2 = v[2, 0]
    m = np.array([[0, -v2, v1], [v2, 0, -v0], [-v1, v0, 0]])
    return m

def ZJ(theta):
    theta = theta.reshape(-1)
    a1 = theta[0]
    a2 = theta[1]
    a3 = theta[2]
    a4 = theta[3]
    a5 = theta[4]
    a6 = theta[5]
    T1 = np.array([[np.cos(a1), 0, np.sin(a1), 0],
        [np.sin(a1), 0, -np.cos(a1), 0],
        [0, 1, 0, 0.08945],
        [0, 0, 0, 1]])
    T2 = np.array([[np.cos(a2),-np.sin(a2),0,-0.425*np.cos(a2)],
        [np.sin(a2), np.cos(a2), 0, -0.425*np.sin(a2)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    T3 = np.array([[np.cos(a3), -np.sin(a3), 0, -0.392*np.cos(a3)],
      [np.sin(a3), np.cos(a3), 0, -0.392*np.sin(a3)],
      [0, 0, 1, 0],
      [0, 0, 0, 1]])
    T4 = np.array([[np.cos(a4), 0, np.sin(a4), 0],
      [np.sin(a4), 0, -np.cos(a4), 0],
      [0, 1, 0, 0.109],
      [0, 0, 0, 1]])
    T5 = np.array([[np.cos(a5), 0, -np.sin(a5), 0],
      [np.sin(a5), 0, np.cos(a5), 0],
      [0, -1, 0, 0.0947],
      [0, 0, 0, 1]])
    T6 = np.array([[np.cos(a6), -np.sin(a6), 0, 0],
       [np.sin(a6), np.cos(a6), 0, 0],
       [0, 0, 1, 0.0823],
       [0, 0, 0, 1]])
    T = T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)
    return T

def ZJo3(theta):
    theta = theta.reshape(-1)
    a1 = theta[0]
    a2 = theta[1]
    a3 = theta[2]
    T1 = np.array([[np.cos(a1), 0, np.sin(a1), 0],
        [np.sin(a1), 0, -np.cos(a1), 0],
        [0, 1, 0, 0.08945],
        [0, 0, 0, 1]])
    T2 = np.array([[np.cos(a2),-np.sin(a2),0,-0.425*np.cos(a2)],
        [np.sin(a2), np.cos(a2), 0, -0.425*np.sin(a2)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    T3 = np.array([[np.cos(a3), -np.sin(a3), 0, -0.392*np.cos(a3)],
      [np.sin(a3), np.cos(a3), 0, -0.392*np.sin(a3)],
      [0, 0, 1, 0],
      [0, 0, 0, 1]])
    To3 = T1.dot(T2).dot(T3)
    return To3

def Jaco(theta):
    a = np.array([0, -0.42500, -0.39225, 0, 0, 0])
    d = np.array([0.089159, 0, 0, 0.10915, 0.09465, 0.08230])
    a1 = theta[0]
    a2 = theta[1]
    a3 = theta[2]
    a4 = theta[3]
    a5 = theta[4]
    a6 = theta[5]
    T1 = np.array([[math.cos(a1),0,math.sin(a1),0],
                  [math.sin(a1),0,-math.cos(a1),0],
                  [0,1,0,0.08945],[0,0,0,1]])
    T2 = np.array([[math.cos(a2),-math.sin(a2),0,-0.425*math.cos(a2)],
                  [math.sin(a2),math.cos(a2),0,-0.425*math.sin(a2)],
                  [0,0,1,0],[0,0,0,1]])
    T3 = np.array([[math.cos(a3),-math.sin(a3),0,-0.392*math.cos(a3)],
                  [math.sin(a3),math.cos(a3),0,-0.392*math.sin(a3)],
                  [0,0,1,0],[0,0,0,1]])
    T4 = np.array([[math.cos(a4),0,math.sin(a4),0],
                  [math.sin(a4),0,-math.cos(a4),0],
                  [0,1,0,0.109], [0, 0, 0, 1]])
    T5 = np.array([[math.cos(a5),0,-math.sin(a5),0],
                  [math.sin(a5),0,math.cos(a5),0],
                  [0,-1,0,0.0947], [0, 0, 0, 1]])
    T6 = np.array([[math.cos(a6),-math.sin(a6),0,0],
                  [math.sin(a6),math.cos(a6),0,0],
                  [0,0,1,0.0823], [0, 0, 0, 1]])
    T10 = T1
    T20 = np.dot(T1,T2)
    T30 = T1.dot(T2).dot(T3)
    T40 = T1.dot(T2).dot(T3).dot(T4)
    T50 = T1.dot(T2).dot(T3).dot(T4).dot(T5)
    T60 = T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)
    z0 = np.array([[0,0,1]]).T
    z1 = T10[0:3, 2]
    z1 = z1.reshape(-1, 1)
    z2 = T20[0:3, 2]
    z2 = z2.reshape(-1, 1)
    z3 = T30[0:3, 2]
    z3 = z3.reshape(-1, 1)
    z4 = T40[0:3, 2]
    z4 = z4.reshape(-1, 1)
    z5 = T50[0:3, 2]
    z5 = z5.reshape(-1, 1)
    o0 = np.array([[0,0,0]]).T
    o1 = T10[0:3, 3]
    o1 = o1.reshape(-1, 1)
    o2 = T20[0:3, 3]
    o2 = o2.reshape(-1, 1)
    o3 = T30[0:3, 3]
    o3 = o3.reshape(-1, 1)
    o4 = T40[0:3, 3]
    o4 = o4.reshape(-1, 1)
    o5 = T50[0:3, 3]
    o5 = o5.reshape(-1, 1)
    o6 = T60[0:3, 3]
    o6 = o6.reshape(-1, 1)
    Jw = np.concatenate((z0, z1, z2, z3, z4, z5), axis = 1)
    Jv1 = askew(z0).dot((o6 - o0))
    Jv2 = askew(z1).dot((o6 - o1))
    Jv3 = askew(z2).dot((o6 - o2))
    Jv4 = askew(z3).dot((o6 - o3))
    Jv5 = askew(z4).dot((o6 - o4))
    Jv6 = askew(z5).dot((o6 - o5))
    Jv = np.concatenate((Jv1, Jv2, Jv3, Jv4, Jv5, Jv6), axis = 1)
    Ja = np.concatenate((Jv,Jw),axis=0) # 几何雅可比
    return Ja

def qccosm(pos):
    pos = pos.reshape(-1, 1)
    ang = pos[3:6,0]
    ang = ang.reshape(-1, 1)
    R = np.zeros([4,4])
    R[0,0] = math.cos(ang[0,0])*math.cos(ang[1,0])
    R[0,1] = math.cos(ang[0,0])*math.sin(ang[1,0])*math.sin(ang[2,0])-math.sin(ang[0,0])*math.cos(ang[2,0])
    R[0,2] = math.cos(ang[0,0])*math.sin(ang[1,0])*math.cos(ang[2,0])+math.sin(ang[0,0])*math.sin(ang[2,0])
    R[0,3] = pos[0,0]
    R[1,0] = math.sin(ang[0,0])*math.cos(ang[1,0])
    R[1,1] = math.sin(ang[0,0])*math.sin(ang[1,0])*math.sin(ang[2,0])+math.cos(ang[0,0])*math.cos(ang[2,0])
    R[1,2] = math.sin(ang[0,0])*math.sin(ang[1,0])*math.cos(ang[2,0])-math.cos(ang[0,0])*math.sin(ang[2,0])
    R[1,3] = pos[1,0]
    R[2,0] = -math.sin(ang[1,0])
    R[2,1] = math.cos(ang[1])*math.sin(ang[2,0])
    R[2,2] = math.cos(ang[1,0])*math.cos(ang[2,0])
    R[2,3] = pos[2,0]
    R[3,:] = [0, 0, 0, 1]
    return R

def PZJC(theta):
    theta = theta.reshape(-1, 1)
    a1 = theta[0,0]
    a2 = theta[1,0]
    a3 = theta[2,0]
    a4 = theta[3,0]
    a5 = theta[4,0]
    a6 = theta[5,0]
    T1 = np.array([[math.cos(a1), 0, math.sin(a1), 0],
                  [math.sin(a1), 0, -math.cos(a1), 0],
                  [0, 1, 0, 0.08945], [0, 0, 0, 1]])
    T2 = np.array([[math.cos(a2), -math.sin(a2), 0, -0.425 * math.cos(a2)],
                  [math.sin(a2), math.cos(a2), 0, -0.425 * math.sin(a2)],
                  [0, 0, 1, 0], [0, 0, 0, 1]])
    T3 = np.array([[math.cos(a3), -math.sin(a3), 0, -0.392 * math.cos(a3)],
                  [math.sin(a3), math.cos(a3), 0, -0.392 * math.sin(a3)],
                  [0, 0, 1, 0], [0, 0, 0, 1]])
    T4 = np.array([[math.cos(a4), 0, math.sin(a4), 0],
                  [math.sin(a4), 0, -math.cos(a4), 0],
                  [0, 1, 0, 0.109], [0, 0, 0, 1]])
    T5 = np.array([[math.cos(a5), 0, -math.sin(a5), 0],
                  [math.sin(a5), 0, math.cos(a5), 0],
                  [0, -1, 0, 0.0947], [0, 0, 0, 1]])
    T6 = np.array([[math.cos(a6), -math.sin(a6), 0, 0],
                  [math.sin(a6), math.cos(a6), 0, 0],
                  [0, 0, 1, 0.0823], [0, 0, 0, 1]])
    T6b = np.array([[math.cos(a6), -math.sin(a6), 0, 0],
                  [math.sin(a6), math.cos(a6), 0, 0],
                  [0, 0, 1, -0.06], [0, 0, 0, 1]])
    T10 = T1
    T20 = np.dot(T1, T2)
    T30 = T1.dot(T2).dot(T3)
    T40 = T1.dot(T2).dot(T3).dot(T4)
    T50 = T1.dot(T2).dot(T3).dot(T4).dot(T5)
    T60 = T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)
    T6b0 = T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6b)
    o1 = T10[0:3, 3]
    o2 = T20[0:3, 3]
    o3 = T30[0:3, 3]
    o4 = T40[0:3, 3]
    o5 = T50[0:3, 3]
    o6 = T60[0:3, 3]
    o6b = T6b0[0:3, 3]
#############################################################
##########计算末端与L3的距离#################################
    L3 = o3 - o2
    o6L3 = (o6 - o2) - np.dot((o6 - o2), L3) / np.dot(L3, L3) * L3 #L3上o6投影点指o6的向量
    d_o6L3 = np.linalg.norm(o6L3)
    #判断末端与L3相对位置
    A = (np.dot((o6 - o2),L3)) * (np.dot((o6-o3),L3))
    k = 0.5 * (abs(A) - A) / abs(A) * (0.05 / (abs(d_o6L3) - 0.086))
    if (abs(d_o6L3) - 0.086) > 0.025:
        k = 0
####以连杆L3末端坐标系o3x3y3z3为基坐标系计算末端避让L3时需要的关节速度
    T63 = T4.dot(T5).dot(T6)
    T53 = T4.dot(T5)
    T43 = T4
    z33 = np.array([[0, 0, 1]]).T
    z43 = T43[0:3, 2]
    z43 = z43.reshape(-1, 1)
    z53 = T53[0:3, 2]
    z53 = z53.reshape(-1, 1)
    o33 = np.array([[0, 0, 0]]).T
    o43 = T43[0:3, 3]
    o43 = o43.reshape(-1, 1)
    o53 = T53[0:3, 3]
    o53 = o53.reshape(-1, 1)
    o63 = T63[0:3, 3]
    o63 = o63.reshape(-1, 1)
    Jw3 = np.concatenate((z33, z43, z53), axis=1)
    Jv43 = askew(z33).dot(o63 - o33)
    Jv53 = askew(z43).dot(o63 - o43)
    Jv63 = askew(z53).dot(o63 - o53)
    Jv3 = np.concatenate((Jv43, Jv53, Jv63),axis = 1)
    Ja3 = np.concatenate((Jv3, Jw3), axis = 0)
    tcpb63 = o6L3
    T31 = T1.dot(T2).dot(T3)
    tcpb3 = T31[0:3, 0:3].T.dot(tcpb63)/(np.linalg.norm(tcpb63))#在关节3坐标系下的末端避让速度方向
    tcpb3 = tcpb3.reshape(-1, 1)
    qvb3 = 5 * k * np.linalg.pinv(Ja3).dot(np.concatenate((tcpb3, np.zeros([3,1])),axis = 0))
###################################################################
###############计算末端蓝色部分与L3距离 ###########################
    L3 = o3 - o2
    o6bL3 = (o6b - o2) - np.dot((o6b - o2), L3) / np.dot(L3, L3) * L3 #L3上o6b投影点指o6b的向量
    d_o6bL3 = np.linalg.norm(o6bL3)
###############判断蓝色部分与L3相对位置
    A = (np.dot((o6b - o2), L3)) * (np.dot((o6b-o3), L3))
    k = 0.5 * (abs(A) - A) / abs(A) * (0.05 / (abs(d_o6bL3)))
    if (abs(d_o6bL3)) > 0.085:
        k = 0
#########以连杆L3末端坐标系o3x3y3z3为基坐标系计算末端蓝色避让L3时需要的关节速度
    T6b3 = T4.dot(T5).dot(T6b)
    T53 = T4.dot(T5)
    T43 = T4
    z33 = np.array([[0, 0, 1]]).T
    z43 = T43[0:3, 2]
    z43 = z43.reshape(-1, 1)
    z53 = T53[0:3, 2]
    z53 = z53.reshape(-1, 1)
    o33 = np.array([[0, 0, 0]]).T
    o43 = T43[0:3, 3]
    o43 = o43.reshape(-1, 1)
    o53 = T53[0:3, 3]
    o53 = o53.reshape(-1, 1)
    o6b3 = T6b3[0:3, 3]
    o6b3 = o6b3.reshape(-1, 1)
    Jw3 = np.concatenate((z33,z43,z53),axis = 1)
    Jv43 = askew(z33).dot(o6b3 - o33)
    Jv53 = askew(z43).dot(o6b3 - o43)
    Jv6b3 = askew(z53).dot(o6b3 - o53)
    Jv3 = np.concatenate((Jv43, Jv53, Jv6b3), axis = 1)
    Ja3 = np.concatenate((Jv3, Jw3), axis = 0)
    tcpb6b3 = o6bL3
    T31 = T1.dot(T2).dot(T3)
    tcpb3 = T31[0:3,0:3].T.dot(tcpb6b3)/(np.linalg.norm(tcpb6b3)) #在关节3坐标系下的末端蓝色避让速度方向
    tcpb3 = tcpb3.reshape(-1, 1)
    qvb3b = 5 * k * np.linalg.pinv(Ja3).dot(np.concatenate((tcpb3, np.zeros([3, 1])), axis = 0))
###############################################################################
######计算o3与L2的距离#########################################################
    L2 = o2 - o1
    o3L2 = (o3 - o1) - np.dot((o3 - o1), L2) / np.dot(L2, L2)*L2 #L2上o3投影点指o4的向量
    d_o3L2 = np.linalg.norm(o3L2)
##判断o3与L2相对位置
    A = np.dot((o3 - o1),L2) * np.dot((o3-o2),L2)
    k = 0.5 * (abs(A) - A)/abs(A) * (0.05/(abs(d_o3L2) - 0.1))
    if (abs(d_o6L3) - 0.1) > 0.35:
        k = 0
 ##以连杆L2末端坐标系o2x2y2z2为基坐标系计算o3避让L2时需要的关节速度
    T32 = T3
    z22 = np.array([[0,0,1]]).T
    o22 = np.array([[0,0,0]]).T
    o32 = T32[0:3,3]
    o32 = o32.reshape(-1, 1)
    Jw2 = z22
    Jv2 = askew(z22).dot(o32 - o22)
    Ja2 = np.concatenate((Jv2, Jw2),axis = 0)
    tcpb32 = o3L2
    T21 = T1.dot(T2)
##在关节2坐标系下的o3避让速度
    tcpb2 = T21[0:3, 0:3].T.dot(tcpb32)/np.linalg.norm(tcpb32)
    tcpb2 = tcpb2.reshape(-1, 1)
    qvb2 = 5 * k * np.linalg.pinv(Ja2).dot(np.concatenate((tcpb2, np.zeros([3, 1])), axis = 0))
    q1 = np.array([[qvb3[0,0]],[qvb3[1,0]]])
    q2 = np.array([[qvb3b[0,0]],[qvb3b[1,0]]])
    qvb = np.concatenate((np.zeros([3, 1]),q1,np.zeros([1,1])), axis = 0) + \
          np.concatenate((np.zeros([3, 1]),q2, np.zeros([1,1])), axis = 0) + \
          np.concatenate((np.zeros([2, 1]), qvb2, np.zeros([3,1])), axis = 0)
    return qvb

def m2eular(m):
    psi = math.atan2(m[2,1], m[2,2])
    theta = math.atan2(-m[2,0], math.sqrt(pow(m[0, 0], 2) + pow(m[1,0], 2)))
    phi = math.atan2(m[1, 0], m[0, 0])
    eu = np.array([[phi, theta, psi]]).T
    return eu

def rv2m(rv):
    rv = rv.reshape(-1, 1)
    nm2 = rv.T.dot(rv)
    if nm2 < 1e-8:   # 如果模方很小，则可用泰勒展开前几项求三角函数
        a = 1-nm2*(1/6-nm2/120)
        b = 0.5-nm2*(1/24-nm2/720)  # a->1, b->0.5
    else:
        nm = math.sqrt(nm2)
        a = math.sin(nm)/nm
        b = (1-math.cos(nm))/nm2
    VX = askew(rv)
    m = np.eye(3) + a*VX + b*VX.dot(VX)
    return m

def m2rv(m):
    m = m.reshape(-1, 3)
    theta = math.acos(0.5*(np.trace(m)-1))
    askew_rv = 0.5*(m-m.T)/math.sin(theta)
    rvx = 0.5*(askew_rv[2,1]-askew_rv[1,2])
    rvy = 0.5*(askew_rv[0,2]-askew_rv[2,0])
    rvz = 0.5*(askew_rv[1,0]-askew_rv[0,1])
    rv = theta*np.array([rvx, rvy, rvz])
    return rv

def rv2eu(rv):
    m = rv2m(rv)
    eu = m2eular(m)
    return eu

def prv2peu(prv):
    rv = np.array([prv[3],prv[4],prv[5]])
    m = rv2m(rv)
    eu = m2eular(m)
    eu = eu.reshape(-1)
    peu = np.array([prv[0], prv[1], prv[2], eu[0], eu[1], eu[2]])
    return peu

def njxz(qt,theta):
    qt = np.tile(qt,(theta.shape[0],1))
    qm = qt - theta
    qmmax_index = np.argmax(abs(qm), axis = 1)
    qmmax = qm[range(qm.shape[0]), qmmax_index]
    ind = np.where(abs(qmmax) == np.min(abs(qmmax)))
    index = ind[0]
    qe = theta[index[0], :]
    return qe

def posture(pe,pt):
    pe = pe.reshape(-1,1)
    pt = pt.reshape(-1,1)
    pet = pe - pt
    posz = pet/np.linalg.norm(pet)
    if  pet[2,0]!=0:
        px = pet[2,0] / abs(pet[2,0]) * pet[0,0]
        py = pet[2,0] / abs(pet[2,0]) * pet[1,0]
        pz = -pet[2,0] / abs(pet[2,0])*(pow(px,2)+pow(py,2))/pet[2,0]
        k = math.sqrt(1 / (pow(px,2)+pow(py,2)+pow(pz,2)))
        #k = -pet[2,0] / abs(pet[2,0]) * abs(k)
        posy = k *np.array([px, py, pz])#由相机安装情况确定
    else:
        posy =np.array([0, 0, -1])
    posy = posy.reshape(-1,1)
    posz =posz.reshape(-1,1)
    posx = askew(posy).dot(posz)
    posx = posx / np.linalg.norm(posx)
    posx = posx.reshape(-1,1)
    m = np.concatenate((posx, posy, posz),axis = 1)
    rv = m2rv(m)
    rv = rv.reshape(-1)
    return rv.T

def nijie45(end_z):
    end_z = end_z/np.linalg.norm(end_z)
    zx = end_z[0]
    zy = end_z[1]
    zz = end_z[2]
    theta4 = np.zeros((4, 1))
    theta5 = np.zeros((4, 1))
    theta4[0:2, 0] = np.arctan2(zy, zx)
    theta4[2:4, 0] = np.arctan2(-zy, -zx)
    theta5[0, 0] = np.arccos(zz)
    theta5[1, 0] = -np.arccos(zz)
    theta5[2, 0] = np.arccos(zz)
    theta5[3, 0] = -np.arccos(zz)
    theta45 = np.concatenate((theta4,theta5), axis =1)
    # print('theta4', theta4)
    # print('theta5', theta5)
    zcx = -np.cos(theta4)*np.sin(theta5)
    zcy = -np.sin(theta4)*np.sin(theta5)
    zcz = np.cos(theta5)
    zc = np.array([zcx, zcy, zcz]).squeeze()
    zc = zc.reshape(3, 4)
    # print(zc)
    jc = abs(zc-end_z.reshape(3, 1))
    # print(jc)
    jc1 = np.zeros((4, 1))
    for i in range(4):
        jc1[i, :] = np.linalg.norm(jc[:, i])
    theta45 = theta45[np.all(jc1<0.05, axis = 1)]
    # print(theta45)
    # nrow1 = theta45[:,0] <= -90 / 180 * np.pi
    # nrow2 = -220 / 180 * math.pi <= theta45[:,0]
    # nrowz = np.bitwise_and(nrow1[0], nrow2[0])
    # print(nrow1,nrow2)
    # theta45 = theta45[nrowz]
    return theta45

def nijieo3(o3):
    a = [0, -0.42500, -0.39225, 0, 0, 0]
    d = [0.089159, 0, 0, 0.10915, 0.09465, 0.08230]
    x = o3[0]
    y = o3[1]
    z = o3[2]
    a2 = 0
    a3 = 0
    theta1 = np.arctan2(y, x)
    a1 = theta1
    T1 = np.array([[np.cos(a1), 0, np.sin(a1), 0],
                  [np.sin(a1), 0, -np.cos(a1), 0],
                  [0, 1, 0, 0.08945],
                  [0, 0, 0, 1]])
    T2 = np.array([[np.cos(a2), -np.sin(a2), 0, -0.425*np.cos(a2)],
                  [np.sin(a2), np.cos(a2), 0, -0.425*np.sin(a2)],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    T3 = np.array([[np.cos(a3), -np.sin(a3), 0, -0.392*np.cos(a3)],
                  [np.sin(a3), np.cos(a3), 0, -0.392*np.sin(a3)],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    T = T1.dot(T2).dot(T3)
    if T[0,3] / T[1,3] - x / y > 0.1:
        theta1 = np.arctan2(y, -x)
        if T[0,3] / T[1,3] - x / y > 0.1:
            theta1 = np.arctan2(-y, x)
    c3 = (pow((y / np.sin(theta1)),2) + pow((z - d[0]),2) - pow(a[1],2) - pow(a[2],2)) / (2 * a[1] * a[2])
    theta3 = np.arccos(c3)
    A = a[2] * np.cos(theta3) + a[1]
    B = a[2] * np.sin(theta3)
    fm = np.sqrt(np.dot(A,A) + np.dot(B,B))
    theta2 = np.zeros((4, 1))
    thetaz = np.zeros((8, 3))
    theta2[0, 0] = np.arcsin((z - d[0])/ fm) - np.arccos(A / fm)
    theta2[1, 0] = -np.pi - np.arcsin((z - d[0])/ fm) - np.arccos(A / fm)
    theta2[2, 0] = np.arcsin((z - d[0]) / fm) + np.arccos(A/ fm)
    theta2[3, 0] = -np.pi - np.arcsin((z - d[0]) / fm) + np.arccos(A / fm)
    thetaz[0: 8, 0] = theta1
    thetaz[0: 4, 2] = theta3.reshape(-1)
    thetaz[4: 8, 2] = -theta3.reshape(-1)
    thetaz[0: 4, 1] = theta2.reshape(-1)
    thetaz[4: 8, 1] = theta2.reshape(-1)
    cz = np.zeros((8,1))
    for i in range(8):
        a1 = thetaz[i, 0]
        a2 = thetaz[i, 1]
        a3 = thetaz[i, 2]
        T1 = np.array([[np.cos(a1), 0, np.sin(a1), 0],
                      [np.sin(a1), 0, -np.cos(a1), 0],
                      [0, 1, 0, 0.08945],
                      [0, 0, 0, 1]])
        T2 = np.array([[np.cos(a2), -np.sin(a2), 0, -0.425 * np.cos(a2)],
                      [np.sin(a2), np.cos(a2), 0, -0.425 * np.sin(a2)],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        T3 = np.array([[np.cos(a3), -np.sin(a3), 0, -0.392 * np.cos(a3)],
                      [np.sin(a3), np.cos(a3), 0, -0.392 * np.sin(a3)],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        T = T1.dot(T2).dot(T3)
        cz[i, 0] = np.linalg.norm(T[0:3, 3]-o3.reshape(-1))
    thetaz = thetaz[~np.all(cz > 0.05, axis=1)]

    theta1 = np.arctan2(-y, -x)
    a1 = theta1
    T1 = np.array([[np.cos(a1), 0, np.sin(a1), 0],
                  [np.sin(a1), 0, -np.cos(a1), 0],
                  [0, 1, 0, 0.08945],
                  [0, 0, 0, 1]])
    T2 = np.array([[np.cos(a2), -np.sin(a2), 0, -0.425 * np.cos(a2)],
                  [np.sin(a2), np.cos(a2), 0, -0.425 * np.sin(a2)],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    T3 = np.array([[np.cos(a3), -np.sin(a3), 0, -0.392 * np.cos(a3)],
                  [np.sin(a3), np.cos(a3), 0, -0.392 * np.sin(a3)],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    T = T1.dot(T2).dot(T3)
    if T[0, 3] / T[1, 3] - x / y > 0.1:
        theta1 = np.arctan2(y, -x)
        if T[0, 3] / T[1, 3] - x / y > 0.1:
            theta1 = np.arctan2(-y, x)
    c3 = (pow((y / np.sin(theta1)), 2) + pow((z - d[0]), 2) - pow(a[1], 2) - pow(a[2], 2)) / (2 * a[1] * a[2])
    theta3 = np.arccos(c3)
    A = a[2] * np.cos(theta3) + a[1]
    B = a[2] * np.sin(theta3)
    fm = np.sqrt(np.dot(A, A) + np.dot(B, B))
    theta2 = np.zeros((4, 1))
    thetaf = np.zeros((8, 3))
    theta2[0, 0] = np.arcsin((z - d[0]) / fm) - np.arccos(A / fm)
    theta2[1, 0] = -np.pi - np.arcsin((z - d[0]) / fm) - np.arccos(A / fm)
    theta2[2, 0] = np.arcsin((z - d[0]) / fm) + np.arccos(A / fm)
    theta2[3, 0] = -np.pi - np.arcsin((z - d[0]) / fm) + np.arccos(A / fm)
    thetaf[0: 8, 0] = theta1
    thetaf[0: 4, 2] = theta3.reshape(-1)
    thetaf[4: 8, 2] = -theta3.reshape(-1)
    thetaf[0: 4, 1] = theta2.reshape(-1)
    thetaf[4: 8, 1] = theta2.reshape(-1)
    cf = np.zeros((8, 1))
    for i in range(7):
        a1 = thetaf[i, 0]
        a2 = thetaf[i, 1]
        a3 = thetaf[i, 2]
        T1 = np.array([[np.cos(a1), 0, np.sin(a1), 0],
                      [np.sin(a1), 0, -np.cos(a1), 0],
                      [0, 1, 0, 0.08945],
                      [0, 0, 0, 1]])
        T2 = np.array([[np.cos(a2), -np.sin(a2), 0, -0.425 * np.cos(a2)],
                      [np.sin(a2), np.cos(a2), 0, -0.425 * np.sin(a2)],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        T3 = np.array([[np.cos(a3), -np.sin(a3), 0, -0.392 * np.cos(a3)],
                      [np.sin(a3), np.cos(a3), 0, -0.392 * np.sin(a3)],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        T = T1.dot(T2).dot(T3)
        cf[i, 0] = np.linalg.norm(T[0:3, 3] - o3.reshape(-1))
    thetaf = thetaf[~np.all(cz > 0.05, axis=1)]
    theta = np.concatenate((thetaz,thetaf),axis = 0)
    return theta

def position(pe,qt,rv,l):
    a1 = qt[0]
    a2 = qt[1]
    a3 = qt[2]
    a4 = qt[3]
    a5 = qt[4]
    a6 = qt[5]
    p3 = pe - np.array([0, 0, 0.5])
    o3e = l * p3 / np.linalg.norm(p3) + np.array([0, 0, 0.5])
    print('o3e: {}'.format(o3e))
    theta123 = nijieo3(o3e)
    q123 = np.array([a1, a2, a3])
    q123_next = njxz(q123, theta123)
    T1 = np.array([[np.cos(a1), 0, np.sin(a1), 0],
                   [np.sin(a1), 0, -np.cos(a1), 0],
                   [0, 1, 0, 0.08945],
                   [0, 0, 0, 1]])
    T2 = np.array([[np.cos(a2), -np.sin(a2), 0, -0.425 * np.cos(a2)],
                   [np.sin(a2), np.cos(a2), 0, -0.425 * np.sin(a2)],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    T3 = np.array([[np.cos(a3), -np.sin(a3), 0, -0.392 * np.cos(a3)],
                   [np.sin(a3), np.cos(a3), 0, -0.392 * np.sin(a3)],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    T30 = T1.dot(T2).dot(T3)
    m = rv2m(rv)
    ez = m[0:3,2]
    ez3 = np.linalg.inv(T30[0:3, 0:3]).dot(ez)
    theta45 = nijie45(ez3)
    qt45 = np.array([a5, a6])
    # theta45 = njxz(qt45, theta45)
    print('theta45',theta45)
    theta4 = theta45[1,0]
    theta5 = theta45[1,1]
    T4 = np.array([[np.cos(theta4), 0, np.sin(theta4), 0],
                   [np.sin(theta4), 0, -np.cos(theta4), 0],
                   [0, 1, 0, 0.109],
                   [0, 0, 0, 1]])
    T5 = np.array([[np.cos(theta5), 0, -np.sin(theta5), 0],
                   [np.sin(theta5), 0, np.cos(theta5), 0],
                   [0, -1, 0, 0.0947],
                   [0, 0, 0, 1]])
    T6 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0.0823],
                   [0, 0, 0, 1]])
    T63 = T4.dot(T5).dot(T6)
    p63 = T63[0:3, 3]
    T30e = ZJo3(q123_next)
    p6 = T30e[0:3,0:3].dot(p63)
    pee = o3e+p6
    # print(pee)
    pee = pee.reshape(-1)
    thetap = np.array([q123_next[0],q123_next[1],q123_next[1],theta4,theta5,0])
    # print('thetap',thetap)
    return thetap

def nijie5x(T):
    a = np.array([0, -0.42500, -0.39225, 0, 0, 0])
    d = np.array([0.089159, 0, 0, 0.10915, 0.09465, 0.08230])
    nx = T[0, 0]
    ny = T[1, 0]
    nz = T[2, 0]
    ox = T[0, 1]
    oy = T[1, 1]
    oz = T[2, 1]
    ax = T[0, 2]
    ay = T[1, 2]
    az = T[2, 2]
    px = T[0, 3]
    py = T[1, 3]
    pz = T[2, 3]
    # 求解关节角1
    theta1 = np.zeros((8, 1))
    m = d[5] * ay - py
    n = ax * d[5] - px
    yx = pow(m, 2) + pow(n, 2) - pow(d[3], 2)
    if yx < 0:
        theta = np.array([])
    else:
        theta1[0:4, 0] = np.arctan2(m, n) - np.arctan2(d[3], math.sqrt(pow(m, 2) + pow(n, 2) - pow(d[3], 2)))
        theta1[4:8, 0] = math.atan2(m, n) - math.atan2(d[3], -math.sqrt(pow(m, 2) + pow(n, 2) - pow(d[3], 2)))
        # 求解关节角5
        theta5 = np.zeros((8, 1))
        theta5[0:2, 0] = np.arccos(ax * np.sin(theta1[0, 0]) - ay * np.cos(theta1[0, 0]))
        theta5[2:4, 0] = -np.arccos(ax * np.sin(theta1[2, 0]) - ay * np.cos(theta1[2, 0]))
        theta5[4:6, 0] = np.arccos(ax * np.sin(theta1[4, 0]) - ay * np.cos(theta1[4, 0]))
        theta5[6:8, 0] = -np.arccos(ax * np.sin(theta1[6, 0]) - ay * np.cos(theta1[6, 0]))
        # 求解关节角3
        theta3 = np.zeros((8,1))
        c234 = (ax * np.cos(theta1) + ay * np.sin(theta1)) / -np.sin(theta5)
        s234 = az / -np.sin(theta5)
        mmm = -d[5] * (ax * np.cos(theta1) + ay * np.sin(theta1)) + px * np.cos(theta1) + \
                py * np.sin(theta1) - d[4] * s234
        nnn = pz - d[0] - az * d[5] + d[4] * c234
        hs = (mmm * mmm + nnn * nnn - pow(a[1], 2) - pow(a[2], 2)) / (2 * a[1] * a[2])
        nrow = [abs(hs) <= 1]
        hs[abs(hs) > 1] = 1
        theta3[0, 0] = np.arccos(hs[0, 0])
        theta3[1, 0] = -np.arccos(hs[1, 0])
        theta3[2, 0] = np.arccos(hs[2, 0])
        theta3[3, 0] = -np.arccos(hs[3, 0])
        theta3[4, 0] = np.arccos(hs[4, 0])
        theta3[5, 0] = -np.arccos(hs[5, 0])
        theta3[6, 0] = np.arccos(hs[6, 0])
        theta3[7, 0] = -np.arccos(hs[7, 0])
        # 求解关节角2
        s2 = ((a[2]*np.cos(theta3) + a[1])* nnn - a[2] * np.sin(theta3) * mmm)/ \
            (pow(a[1], 2) + pow(a[2], 2) + 2 * a[1] * a[2] * np.cos(theta3))
        c2 = (mmm + a[2] * np.sin(theta3) * s2)/(a[2] * np.cos(theta3) + a[1])
        theta2 = np.arctan2(s2, c2)
        # 整理关节角1,5,3,2
        theta = np.zeros((8, 5))
        theta[:, 0] = theta1.reshape(-1)
        theta[:, 1] = theta2.reshape(-1)
        theta[:, 2] = theta3.reshape(-1)
        theta[:, 4] = theta5.reshape(-1)
        # 求解关节角4
        theta[:, 3] = np.arctan2(s234, c234).reshape(-1) - theta[:, 1]-theta[:, 2]
        theta = np.concatenate((theta, np.zeros((theta.shape[0],1))), axis=1)
        nrow1 = [theta[:, 3].reshape(-1, 1) <= -90/180 * math.pi]
        nrow2 = [-220/180*math.pi <= theta[:, 3].reshape(-1, 1)]
        nrowz = np.bitwise_and(nrow1[0],nrow2[0])
        nrowz = np.bitwise_and(nrowz,nrow)
        # 除去数学计算中不满足的解
        theta = theta * np.tile(np.array(nrowz), (1, 6)).reshape(8, 6)
        theta = theta[~np.all(theta == 0, axis=1)]
    return theta

def position_ctrl5x(tcp_socket_w, tcp_socket_r, pe, l=0.4, dt=0.1):
    # server_ip = "192.168.1.3"
    # server_port = 30003
    # server_addr = (server_ip, server_port)
    # tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # tcp_socket.connect(server_addr)

    pe = pe.reshape(-1, 1).squeeze()
    i = 0
    FLAG = True
    while FLAG:
        print('i: {}'.format(i))
        i = i + 1
        qt = getq1(tcp_socket_r)
        print('qt: {}'.format(qt))
        print('pe=', pe)
        pt = getpos1(tcp_socket_r)
        # qvt = getqv1(tcp_socket)
        pe1 = np.array([pe[0], pe[1], pe[2]])
        pt1 = np.array([pt[0], pt[1], pt[2]])
        pez = posture(pe1, pt1)
        thetap = position(pe1,qt,pez,l)
        # J = Jaco(qt)
        # c = np.sqrt(np.linalg.det(J.dot(J.T)))
        # lamda = 16 / math.tan(c + 1.491)

        qv = thetap-qt
        # print('qv =', qv)
        if (np.max(qv) > 0.66):
            qv = 0.6 * qv / np.linalg.norm(qv)
        qv = qv.reshape(-1,1)
        qvp = PZJC(qt)
        qvp = qvp.reshape(-1,1)
        #qve = qv +qvp + 0.14*np.linalg.inv(J).dot(wc)
        qve = qv + qvp
        # print(qv)
        # print('qve=',qve)
        qa = np.linalg.norm(qvp)+0.5 #qvp越大越接近碰撞，此时关节速度也相应增大
        command = ctrl_speedj(qve, qa, dt)
        tcp_socket_w.send(str.encode(command))
        qt = getq1(tcp_socket_r)
        Tt = ZJ(qt)
        rvt = m2rv(Tt[0:3,0:3])
        zx = rv2eu(pez)- rv2eu(rvt)
        print('zx=', zx)
        if np.linalg.norm(zx) < 0.1:
            FLAG = False

def position_ctrl5xw(tcp_socket, pe, qt, pt,  l, dt):
    # qt = getq1(tcp_socket)
    pe = pe.reshape(4, 1).squeeze()
    # print('pe=', pe)
    # pt = getpos1(tcp_socket)

    pe1 = np.array([pe[0], pe[1], pe[2]])
    pt1 = np.array([pt[0], pt[1], pt[2]])
    pez = posture(pe1, pt1)
    thetap = position(pe1,qt,pez,l)

    qv = thetap - qt
    # print('qv =', qv)
    if (np.max(qv) > 0.66):
        qv = 0.6 * qv / np.linalg.norm(qv)
    qv = qv.reshape(-1, 1)
    qvp = PZJC(qt)
    qvp = qvp.reshape(-1, 1)
    #qve = qv +qvp + 0.14*np.linalg.inv(J).dot(wc)
    qve = qv + qvp
    # print(qv)
    # print('qve=', qve)
    qa = np.linalg.norm(qvp)+0.5 #qvp越大越接近碰撞，此时关节速度也相应增大
    command = ctrl_speedj(qve, qa, dt)
    tcp_socket.send(str.encode(command))
