import scipy.io as sio
import numpy as np


def train_fso_test_f(step=1, train_num=71 * 96, test_num=18 * 96):
    load_data = sio.loadmat('data/new147k1.mat')
    pems = np.load('data/pems_speed_occupancy_15min.npz')
    flow = abnormal_data_process(load_data['data'])
    speed = abnormal_data_process(pems['speed'])
    occupancy = abnormal_data_process(pems['occupancy'])
    speed = normalize(speed)
    occupancy = normalize(occupancy)
    data = np.hstack((flow, speed, occupancy))

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for i in range(train_num):
        train_x.append(data[i:i + step, :])
        train_y.append(flow[i + step, :])

    for i in range(test_num):
        test_x.append(data[train_num + i:train_num + i + step, :])
        test_y.append(flow[train_num + i + step, :])

    train_x_np = np.array(train_x)
    train_y_np = np.array(train_y)
    test_x_np = np.array(test_x)
    test_y_np = np.array(test_y)

    train_x_np[train_x_np <= 0] = 3 / 1956
    train_y_np[train_y_np <= 0] = 3 / 1956
    test_x_np[test_x_np <= 0] = 3 / 1956
    test_y_np[test_y_np <= 0] = 3 / 1956

    return train_x_np, train_y_np, test_x_np, test_y_np


def abnormal_data_process(data):
    # 从0开始的24,25,26,27需要去掉
    data = np.delete(data, [24, 25, 26, 27], axis=1)
    return data


def normalize(data):
    dmin, dmax = data.min(), data.max()  # 求最大最小值
    data = (data - dmin) / (dmax - dmin)
    return data
