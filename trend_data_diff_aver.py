import numpy as np
import scipy.io as sio
from get_data import normalize


def abnormal_data_process_for_week(data):
    data = np.delete(data, [51, 53, 89, 90, 93, 94, 95, 96, 97, 98, 103, 105], axis=1)
    return data


def create_train_test_week(step, train_num, test_num):
    ld = sio.loadmat('data/new147k1.mat')
    flow_data = abnormal_data_process_for_week(ld['data']) * 1956
    flow_data[flow_data <= 3] = 3
    flow_data = flow_data[0:90 * 96, :]
    station_num = flow_data.shape[1]
    flow_diff = np.zeros(shape=[90 * 96, station_num])
    week_aver = np.zeros(shape=flow_diff.shape)

    def data_diff(first_day, after_abnormal_day, weeks, aver_weeks=None):
        if aver_weeks is None:
            aver_weeks = weeks
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
        for i in aver_weeks:
            x = range((first_day - 1 + i * 7) * 96, (first_day + i * 7) * 96)
            flow_diff[x, :] = flow_data[x, :] - mflow_mean
            week_aver[x, :] = mflow_mean
        for i in range(0, 3):
            x = range((after_abnormal_day + i * 7 - 1) * 96, (after_abnormal_day + i * 7) * 96)
            flow_diff[x, :] = flow_data[x, :] - mflow_mean
            week_aver[x, :] = mflow_mean
        return mflow_mean

    monday_aver = data_diff(7, 69, [1, 3, 4, 5, 7, 8, 9], range(1, 10))
    data_diff(8, 70, range(1, 10), range(-1, 10))
    data_diff(2, 71, range(1, 11))
    data_diff(3, 72, range(1, 11))
    data_diff(4, 73, range(1, 11))
    data_diff(5, 74, range(1, 11))
    data_diff(6, 75, range(1, 10))
    xx = range(89 * 96, 90 * 96)
    flow_diff[xx, :] = flow_data[xx, :] - monday_aver  # 为了产生train和test数据方便，最后加了一天
    week_aver[xx, :] = monday_aver


    flow_diff_normalize, flow_diff_min, flow_diff_max = normalize(flow_diff)

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    train_y_aver = []
    test_y_aver = []

    for i in range(train_num):
        train_x.append(flow_diff_normalize[i:i + step, :])
        train_y.append(flow_diff_normalize[i + step, :])
        train_y_aver.append(week_aver[i + step, :])

    for i in range(test_num):
        test_x.append(flow_diff_normalize[train_num + i:train_num + i + step, :])
        test_y.append(flow_diff_normalize[train_num + i + step, :])
        test_y_aver.append(week_aver[train_num + i + step, :])

    train_x_np = np.array(train_x)
    train_y_np = np.array(train_y)
    test_x_np = np.array(test_x)
    test_y_np = np.array(test_y)
    train_y_aver_np = np.array(train_y_aver)
    test_y_aver_np = np.array(test_y_aver)

    return train_x_np, train_y_np, train_y_aver_np, test_x_np, test_y_np, test_y_aver_np, flow_diff_min, flow_diff_max
