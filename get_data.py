import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def abnormal_data_process(data):
    # 从0开始的24,25,26,27需要去掉
    data = np.delete(data, [24, 25, 26, 27, 51, 95, 105], axis=1)
    return data


def normalize(data):
    data_min, data_max = data.min(), data.max()
    data_normalize = (data - data_min) / (data_max - data_min)
    return data_normalize, data_min, data_max


def inverse_normalize(data_normalize, data_min, data_max):
    inverse_data = data_normalize * (data_max - data_min) + data_min
    return inverse_data


def standardize(data):
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data_standard = (data - data_mean) / data_std
    return data_standard, data_mean, data_std


def inverse_standardize(data_standard, data_mean, data_std):
    inverse_data = data_standard * data_std + data_mean
    return inverse_data


def create_train_test(x, y, step, train_num, test_num):
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for i in range(train_num):
        train_x.append(x[i:i + step, :])
        train_y.append(y[i + step, :])

    for i in range(test_num):
        test_x.append(x[train_num + i:train_num + i + step, :])
        test_y.append(y[train_num + i + step, :])

    train_x_np = np.array(train_x)
    train_y_np = np.array(train_y)
    test_x_np = np.array(test_x)
    test_y_np = np.array(test_y)

    return train_x_np, train_y_np, test_x_np, test_y_np


_load_data = sio.loadmat('data/new147k1.mat')
_flow = abnormal_data_process(_load_data['data']) * 1956
_flow[_flow <= 0] = 3

_pems = np.load('data/pems_speed_occupancy_15min.npz')
_speed = abnormal_data_process(_pems['speed'])
_occupancy = abnormal_data_process(_pems['occupancy'])
_occupancy[_occupancy <= 0] = 0.0002

flow_normalized, flow_min, flow_max = normalize(_flow)
speed_normalized, speed_min, speed_max = normalize(_speed)
occupancy_normalized, occupancy_min, occupancy_max = normalize(_occupancy)

flow_standardized, flow_mean, flow_std = standardize(_flow)
speed_standardized, speed_mean, speed_std = standardize(_speed)
occupancy_standardized, occupancy_mean, occupancy_std = standardize(_occupancy)

# def monday_aver_flow(train_num):
flow_diff = _flow.copy()[0:89 * 96, :]

# x = range(6 * 96, 7 * 96)
# mflow = _flow[x, :][np.newaxis, :]
# # plt.plot(_flow[6 * 96:7 * 96, station])
# # plt.show()
# for i in [1, 3, 4, 5, 7, 8, 9]:
#     x = range((6 + i * 7) * 96, (7 + i * 7) * 96)
#     # plt.plot(_flow[x, station])
#     # plt.show()
#     f = _flow[x, :][np.newaxis, :]
#     mflow = np.concatenate((mflow, f), axis=0)
# x = range(68 * 96, 69 * 96)
# f = _flow[x, :][np.newaxis, :]
# mflow = np.concatenate((mflow, f), axis=0)
# mflow_mean = mflow.mean(axis=0)
#
# x = range(6 * 96, 7 * 96)
# flow_diff[x, :] = flow_diff[x, :] - mflow_mean
# for i in range(1, 10):
#     x = range((6 + i * 7) * 96, (7 + i * 7) * 96)
#     flow_diff[x, :] = flow_diff[x, :] - mflow_mean
# for i in range(0, 3):
#     x = range((68 + i * 7) * 96, (69 + i * 7) * 96)
#     flow_diff[x, :] = flow_diff[x, :] - mflow_mean


def data_diff(first_day, abnormal_day, weeks):
    x = range((first_day - 1) * 96, first_day * 96)
    mflow = _flow[x, :][np.newaxis, :]
    for i in weeks:
        x = range(((first_day - 1) + i * 7) * 96, (first_day + i * 7) * 96)
        f = _flow[x, :][np.newaxis, :]
        tflow = np.concatenate((mflow, f), axis=0)
    x = range((abnormal_day - 1) * 96, abnormal_day * 96)
    f = _flow[x, :][np.newaxis, :]
    mflow = np.concatenate((mflow, f), axis=0)
    mflow_mean = mflow.mean(axis=0)

    x = range((first_day - 1) * 96, first_day * 96)
    flow_diff[x, :] = flow_diff[x, :] - mflow_mean
    for i in weeks:
        x = range((6 + i * 7) * 96, (7 + i * 7) * 96)
        flow_diff[x, :] = flow_diff[x, :] - mflow_mean
    for i in range(0, 3):
        x = range((abnormal_day + i * 7 - 1) * 96, (abnormal_day + i * 7) * 96)
        flow_diff[x, :] = flow_diff[x, :] - mflow_mean
    return mflow_mean


station = 0
first_day = 7
abnormal_day = 69
weeks = [1, 3, 4, 5, 7, 8, 9]

ax = data_diff(first_day, abnormal_day, weeks)

# plt.plot(flow_diff[x, :])
# plt.show()

# plt.plot(mflow_mean[:, station], color='k', linewidth=2)
# # plt.show()
# plt.plot(_flow[75 * 96:76 * 96, station], color='r', linewidth=2)
# plt.show()
# a = flow_diff[6 * 96:7 * 96, :] > 400
# b = np.argwhere(a == True)
# plt.plot(flow_diff[20 * 96:21 * 96, :]);
# plt.show()
