import numpy as np
import scipy.io as sio


class Standard:
    def __init__(self, data):
        self.data_mean = data.mean(axis=0)
        self.data_std = data.std(axis=0)
        self.data = (data - self.data_mean) / self.data_std

    def reconstruct(self, data_standard):
        inverse_data = data_standard * self.data_std + self.data_mean
        return inverse_data


class Normal:
    def __init__(self, data):
        self.data_min, self.data_max = data.min(), data.max()
        self.data = (data - self.data_min) / (self.data_max - self.data_min)

    def reconstruct(self, data_normalize):
        inverse_data = data_normalize * (self.data_max - self.data_min) + self.data_min
        return inverse_data


def abnormal_data_process_for_occupancy(data):
    # 从0开始的24,25,26,27需要去掉
    data = np.delete(data, [24, 25, 26, 27], axis=1)
    return data


def load_flow():
    load_data = sio.loadmat('data/new147k1.mat')
    flow = abnormal_data_process_for_occupancy(load_data['data']) * 1956
    flow[flow <= 0] = 3
    return flow


def load_speed():
    pems_data = np.load('data/pems_speed_occupancy_15min.npz')
    speed = abnormal_data_process_for_occupancy(pems_data['speed'])
    return speed


def load_occupancy():
    pems_data = np.load('data/pems_speed_occupancy_15min.npz')
    occupancy = abnormal_data_process_for_occupancy(pems_data['occupancy'])
    occupancy[occupancy <= 0] = 0.0002
    return occupancy


def create_train_test_f_f_normal(time_step, train_num, test_num):
    flow = load_flow()
    flow_normal = Normal(flow)
    train_x, train_y, test_x, test_y = create_train_test(flow_normal.data, flow_normal.data,
                                                         time_step, train_num, test_num)
    data = {'data_process': flow_normal, 'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}
    return data


def create_train_test(x, y, time_step, train_num, test_num):
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for i in range(train_num):
        train_x.append(x[i:i + time_step, :])
        train_y.append(y[i + time_step, :])

    for i in range(test_num):
        test_x.append(x[train_num + i:train_num + i + time_step, :])
        test_y.append(y[train_num + i + time_step, :])

    train_x_np = np.array(train_x)
    train_y_np = np.array(train_y)
    test_x_np = np.array(test_x)
    test_y_np = np.array(test_y)

    return train_x_np, train_y_np, test_x_np, test_y_np
