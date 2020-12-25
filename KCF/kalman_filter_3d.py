import numpy as np
import os

class KF3D:
    def __init__(self, Hz):
        self.Hz = Hz
        self.delta_t = 1 / self.Hz
        # the coefficient of process noise covariance matrix
        self.coefficient_q = 0.01
        # the coefficient of measurement noise covariance matrix
        self.coefficient_r = 1
        # state: {x, x', y, y', z, z', l, l', h, h', w, w'}
        self.state = np.zeros((12, 1)).astype(np.float32)
        # predicted state by KF3D
        self.predict_state = np.zeros((12, 1))
        # transition matrix
        self.F = np.zeros((12, 12))
        # state covariance matrix
        self.P = np.zeros((12, 12))
        # predicted state covariance matrix by KF3D
        self.predict_P = np.zeros((12, 12))
        # process noise covariance matrix
        self.Q = np.zeros((12, 12))
        # measurement: {x, y, z, l, h, w}
        self.z = np.zeros((6, 1))
        # measurement matrix
        self.H = np.zeros((6, 12))
        # measurement noise covariance matrix
        self.R = np.zeros((6, 6))

    def initialize(self, x=0, y=0, z=0, l=0, h=0, w=0):
        # state initial
        self.state = np.array([x, 0, y, 0, z, 0, l, 0, h, 0, w, 0]).reshape(12, 1).astype(np.float32)
        # Transition matrix
        self.F = np.array([[1, self.delta_t, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, self.delta_t, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, self.delta_t, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, self.delta_t, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, self.delta_t, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, self.delta_t],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], np.float32)
        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]], np.float32)
        # State covariance matrix
        self.P = np.zeros(12).astype(np.float32)
        # Process covariance matrix Q
        self.Q = np.array([[10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.float32) * self.coefficient_q
        # Measurement covariance matrix R
        self.R = np.eye(6).astype(np.float32) * self.coefficient_r

    def predict(self):
        """Predict state vector self.state and covariance of uncertainty self.P
                    where,
                    self.state: previous state vector
                    self.P: previous covariance matrix
                    self.F: transition matrix
                    self.Q: process noise covariance matrix
                Equations:
                    hat{state}_{k|k-1} = F*hat{state}_{k-1|k-1}
                    hat{P}_{k|k-1} = F*hat{P}_{k-1|k-1}*F.T + Q
                    where,
                        F.T is F transpose
                Args:
                    None
                Return:
                    vector of predicted state and state covariance matrix
                """
        self.predict_state = np.matmul(self.F, self.state)
        self.predict_P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.Q
        return self.predict_state, self.predict_P

    def update(self, flag=True, x=0, y=0, z=0, l=0, h=0, w=0):
        """Correct or update state vector self.state and covariance of uncertainty self.P
                where,
                self.state: previous state vector
                self.predict_state: predicted state vector
                self.P: state covariance matrix
                self.predict_P: predicted state covariance matrix
                self.H: measurement matrix
                self.z: vector of observations
                R: measurement noise covariance matrix
                Equations:
                    err_{k} = z_{k} - H * hat{state}_{k|k-1}
                    s = H * hat{P}_{k|k-1} * H.T + R
                    k = hat{P}_{k|k-1} * H.T * (s.Inv)
                    state_{k|k} = hat{state}_{k|k-1} + k*err_{k}
                    P_{k|k} = (I - k * H) * P_{k|k-1}
                    where,
                        H.T is H transpose
                        s.Inv is s inverse
                Args:
                    z: vector of observations
                    flag: if "true" prediction result will be updated else detection
                Return:
                    predicted state vector u
                """
        # Observe value
        if flag:
            self.z = np.array([x, y, z, l, h, w]).reshape(6, 1)
        err = self.z - np.matmul(self.H, self.predict_state)
        s = np.matmul(np.matmul(self.H, self.predict_P), self.H.T) + self.R
        k = np.matmul(np.matmul(self.predict_P, self.H.T), np.linalg.inv(s))
        self.state = self.predict_state + np.matmul(k, err)
        self.P = np.matmul((np.eye(12)-np.matmul(k, self.H)), self.predict_P)
        self.z = np.matmul(self.H, self.state)
        return self.z





