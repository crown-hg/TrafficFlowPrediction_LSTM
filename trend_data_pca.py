import numpy as np


class PCA:
    def __init__(self, X, retain):
        # X=n*m n为样本个数，m为样本的维度
        # X必须均值为0,只有在均值为0时，协方差sigma=xT*x
        self.X = X
        dim = X.shape[1]
        sigma = np.matmul(self.X.T, self.X) / dim  # sigma是协方差矩阵，T是转置
        self.u, self.s, _ = np.linalg.svd(sigma)  # u的每一列都是特征向量，s是特征值与u一一对应

        # here iteration is over rows but the columns are the eigenvectors of sigma
        self.X_rot = np.matmul(self.X, self.u)
        u_sum = np.cumsum(self.s)
        self.retain_num = dim
        for i in range(dim):
            if u_sum[i] / u_sum[-1] >= retain:
                self.retain_num = i
                break
        self.main = self.X_rot[:, 0:self.retain_num + 1]
        self.rest = self.X_rot[:, self.retain_num + 1:]

    def reconstruct(self, rest_x):
        self.X_rot[:, self.retain_num + 1:] = rest_x
        return np.matmul(self.X_rot, self.u.T)
