import scipy.io as sio
import numpy as np


def get_pems_data(step=1, train_num=71 * 96, test_num=18 * 96):
    load_data = sio.loadmat('data/new147k1.mat')
    data = load_data['data']

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for i in range(train_num):
        train_x.append(data[i:i + step, :])
        train_y.append(data[i + step, :])

    for i in range(test_num):
        test_x.append(data[train_num + i:train_num + i + step, :])
        test_y.append(data[train_num + i + step, :])

    train_x_np = np.array(train_x)
    train_y_np = np.array(train_y)
    test_x_np = np.array(test_x)
    test_y_np = np.array(test_y)

    train_x_np[train_x_np <= 0] = 3 / 1956
    train_y_np[train_y_np <= 0] = 3 / 1956
    test_x_np[test_x_np <= 0] = 3 / 1956
    test_y_np[test_y_np <= 0] = 3 / 1956

    return train_x_np, train_y_np, test_x_np, test_y_np
