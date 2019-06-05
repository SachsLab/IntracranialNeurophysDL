import numpy as np


class ManualKalmanFilter(object):
    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.m = np.size(self.x, 0)
        self.n = np.size(self.z, 0)
        self.F = np.zeros((self.m, self.m))
        self.H = np.zeros((self.n, self.m))
        self.Q = np.zeros((self.m, self.m))
        self.R = np.zeros((self.n, self.n))
        self.xk_bar = np.zeros(self.m)
        self.pk_bar = np.zeros((self.m, self.m))
        self.P = np.eye(self.m)
        self.model_initialize()

    def model_initialize(self):
        # F = X2.X1^T.(X1.X1^T)^-1
        x1 = self.x[:, :-1]
        x2 = self.x[:, 1:]
        temp1 = np.dot(x2, x1.T)
        temp2 = np.linalg.inv(np.dot(x1, x1.T))
        self.F = np.dot(temp1, temp2)
        # Q = ((X2 - F.X1).(X2 - FX1)^T) / (M-1)
        temp = x2 - np.dot(self.F, x1)
        self.Q = np.dot(temp, temp.T) / (self.m - 1)
        # H = Z.X^T.(X.X^T)^-1
        temp1 = np.dot(self.z, self.x.T)
        temp2 = np.linalg.inv(np.dot(self.x, self.x.T))
        self.H = np.dot(temp1, temp2)
        # R = ((Z - H.X).(Z - H.X)^T) / M
        Z_HX = self.z - np.dot(self.H, self.x)
        temp = np.dot(Z_HX, Z_HX.T)
        self.R = np.divide(temp, self.m)

    def predict(self):
        # I. Priori Step
        x_k_minus_one = self.x[:, -1]  # Initial State
        p_k_minus_one = self.P  # Initial Error Covariance
        self.xk_bar = np.dot(self.F, x_k_minus_one)
        temp = np.dot(self.F, p_k_minus_one)
        self.pk_bar = np.dot(temp, np.transpose(self.F)) + self.Q

    def update(self, z_test):
        # Kk = Pk^-.H^T.(H.Pk^-.H^T + R)^-1
        temp = np.dot(self.pk_bar, np.transpose(self.H))
        temp1 = np.dot(self.H, temp) + self.R
        temp2 = np.linalg.inv(temp1)
        kk = np.dot(temp, temp2)
        # Next State Estimation
        zk = z_test
        temp1 = zk - np.dot(self.H, self.xk_bar)  # z - dot(H, x)
        temp2 = np.dot(kk, temp1)
        xk = self.xk_bar + temp2  # x = x + dot(K, y)
        # Next Error Covariance Estimation
        temp = np.eye(self.m) - np.dot(kk, self.H)  # I_KH = self._I - dot(self.K, H)
        pk = np.dot(temp, self.pk_bar)  # self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)
        # Initializing Next Loop
        xk = np.reshape(xk, (self.m, 1))
        self.x = np.hstack((self.x, xk))
        self.P = pk
        # Updating the model
        zk = np.reshape(zk, (self.n, 1))
        self.z = np.hstack((self.z, zk))
        return xk
