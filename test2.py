#!/usr/bin/env python3

import arm_control as ec

#("192.168.1.3", 30003)
import struct
import numpy as np
import threading
import socket
import time
import signal
import sys
import os


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

                self.info_lock.acquire()
                self.info_q = q
                self.info_qv = qv
                self.info_pos = pos
                self.info_lock.release()

                bufs.clear()
                nremains = 1116

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
        #tcp_socket_ctrl.close()
        #    time.sleep(1)

    def start(self):
        th1 = threading.Thread(target=self.thread_write)
        th1.start()
        th2 = threading.Thread(target=self.thread_read)
        th2.start()

        th1.join()
        th2.join()


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

    def shutdown(self):
        self.shutdown_lock.acquire()
        self.shutdown_flag = True
        self.shutdown_lock.release()



#sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#sk.connect(("192.168.1.3", 30003))

#
# def fuck():
#     FREQ = 10
#     while True:
#         cmd = ec.ctrl_speedj(np.asarray([[1],[0],[0],[0],[0],[0]]),np.asarray([0.1]),np.asarray([0.5]))
#         rc.cmd(cmd)
#         time.sleep(1.0 / FREQ)



rc = RobotControl()

def sigint_handler(sig, frame):
    print("caught sigint.")
    rc.shutdown()
signal.signal(signal.SIGINT, sigint_handler)

rc.start()
rc.cleanup()

print("shutdown.")
sys.exit(0)

#while True:
#    print(sk.recv(1116))
