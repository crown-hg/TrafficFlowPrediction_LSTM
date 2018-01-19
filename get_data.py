import scipy.io as sio
import numpy as np


def abnormal_data_process_for_occupancy(data):
    # 从0开始的24,25,26,27需要去掉
    data = np.delete(data, [24, 25, 26, 27], axis=1)
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


def abnormal_data_process_for_week(data):
    data = np.delete(data, [51, 53, 89, 90, 93, 94, 95, 96, 97, 98, 103, 105], axis=1)
    return data


# def diff_to_original(flow_diff, week_aver):


def get_aver_diff():
    ld = sio.loadmat('data/new147k1.mat')
    flow_data = abnormal_data_process_for_week(ld['data']) * 1956
    flow_data[flow_data <= 0] = 3
    station_num = flow_data.shape[1]
    flow_diff = np.zeros(shape=[90 * 96, station_num])
    week_aver = np.zeros(shape=flow_diff.shape)

    def data_diff(first_day, after_abnormal_day, weeks):
        x = range((first_day - 1) * 96, first_day * 96)
        mflow = flow_data[x, :][np.newaxis, :]
        for i in weeks:
            x = range(((first_day - 1) + i * 7) * 96, (first_day + i * 7) * 96)
            f = flow_data[x, :][np.newaxis, :]
            mflow = np.concatenate((mflow, f), axis=0)
        x = range((after_abnormal_day - 1) * 96, after_abnormal_day * 96)
        f = flow_data[x, :][np.newaxis, :]
        mflow = np.concatenate((mflow, f), axis=0)
        mflow_mean = mflow.mean(axis=0)

        x = range((first_day - 1) * 96, first_day * 96)
        flow_diff[x, :] = flow_data[x, :] - mflow_mean
        week_aver[x, :] = mflow_mean
        for i in weeks:
            x = range((first_day - 1 + i * 7) * 96, (first_day + i * 7) * 96)
            flow_diff[x, :] = flow_data[x, :] - mflow_mean
            week_aver[x, :] = mflow_mean
        for i in range(0, 3):
            x = range((after_abnormal_day + i * 7 - 1) * 96, (after_abnormal_day + i * 7) * 96)
            flow_diff[x, :] = flow_data[x, :] - mflow_mean
            week_aver[x, :] = mflow_mean
        return mflow_mean

    week_aver = list()
    week_aver.append(data_diff(7, 69, [1, 3, 4, 5, 7, 8, 9]))
    week_aver.append(data_diff(8, 70, range(1, 10)))
    week_aver.append(data_diff(2, 71, range(1, 11)))
    week_aver.append(data_diff(3, 72, range(1, 11)))
    week_aver.append(data_diff(4, 73, range(1, 11)))
    week_aver.append(data_diff(5, 74, range(1, 11)))
    week_aver.append(data_diff(6, 75, range(1, 10)))
    x = range(89 * 96, 90 * 96)
    flow_diff[x, :] = flow_data[x, :] - week_aver[0]  # 为了产生train和test数据方便，最后加了一天
    return flow_diff, week_aver


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
_flow = abnormal_data_process_for_occupancy(_load_data['data']) * 1956
_flow[_flow <= 0] = 3

_pems = np.load('data/pems_speed_occupancy_15min.npz')
_speed = abnormal_data_process_for_occupancy(_pems['speed'])
_occupancy = abnormal_data_process_for_occupancy(_pems['occupancy'])
_occupancy[_occupancy <= 0] = 0.0002

flow_normalized, flow_min, flow_max = normalize(_flow)
speed_normalized, speed_min, speed_max = normalize(_speed)
occupancy_normalized, occupancy_min, occupancy_max = normalize(_occupancy)

# flow_standardized, flow_mean, flow_std = standardize(_flow)
# speed_standardized, speed_mean, speed_std = standardize(_speed)
# occupancy_standardized, occupancy_mean, occupancy_std = standardize(_occupancy)


# a = np.argwhere(np.abs(flow_diff) > 300)
# dict(zip(*np.unique(a[:, 1], return_counts=True)))
# print(np.unique(a[:, 1], return_counts=True))
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
