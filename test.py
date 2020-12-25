import numpy as np
import socket
import threading
import time

class Multi_Thread(threading.Thread):
    def __init__(self, func, args=()):
        super(Multi_Thread, self).__init__()
        self.func = func
        self.args = args
        self.result = []
        self.a = 11
        self._running = True

    def input(self, args=[]):
        self.args = args

    def run(self):
        self.result = self.func(self.a, *self.args)

    def terminate(self):
        self._running = False

    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None

def add(a, b):
    time.sleep(1)
    return a + b

if __name__ == '__main__':
    thread = Multi_Thread(func=add)
    a = 1
    b = 2
    thread.input(args=[b])
    thread.start()
    print('thread status: {}'.format(thread.is_alive()))
    print(thread.get_result())
    print('thread status: {}'.format(thread.is_alive()))
