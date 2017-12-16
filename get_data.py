import scipy.io as sio
import numpy as np


def get_pems_data(step=1):
    load_fn = 'data/new147k1.mat'
    load_data = sio.loadmat(load_fn)
    data = load_data['data']

    train_num = 71 * 96
    test_num = 18 * 96

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

    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)
