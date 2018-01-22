import numpy as np
import scipy.io as sio
from get_data import Normal, Standard


class TrendDiff:
    def __init__(self, data_process, train_num, train_y_aver_np, test_num, test_y_aver_np):
        self.data_process = data_process
        self.train_num = train_num
        self.train_y_aver_np = train_y_aver_np
        self.test_num = test_num
        self.test_y_aver_np = test_y_aver_np

    def reconstruct(self, data):
        data_diff = self.data_process.reconstruct(data)
        num = data_diff.shape[0]
        if num == self.train_num:
            return data_diff + self.train_y_aver_np
        else:
            return data_diff + self.test_y_aver_np


class PCA:
    def __init__(self, x, retain):
        # X=m*n m为样本个数，n为样本的维度
        # X必须均值为0,只有在均值为0时，协方差sigma=xT*x=[n,m]*[m,n]=[n,n]
        self.x = x
        m, n = x.shape
        sigma = np.matmul(self.x.T, self.x) / m  # sigma是协方差矩阵，T是转置,sigma=xT*x=[n,m]*[m,n]=[n,n]
        self.u, self.s, _ = np.linalg.svd(sigma)  # u的每一列都是特征向量，s是特征值与u一一对应
        # here iteration is over rows but the columns are the eigenvectors of sigma
        u_sum = np.cumsum(self.s)
        self.retain_num = n
        for i in range(n):
            if u_sum[i] / u_sum[-1] >= retain:
                self.retain_num = i
                break
        self.main_vector = self.u[:, 0:self.retain_num + 1]
        self.rest_vector = self.u[:, self.retain_num + 1:]
        main_x_rot = np.matmul(self.x, self.main_vector)
        rest_x_rot = np.matmul(self.x, self.rest_vector)
        self.main_x = np.matmul(main_x_rot, self.main_vector.T)
        self.rest_x = np.matmul(rest_x_rot, self.rest_vector.T)

    def reduce(self, data):
        main_data_rot = np.matmul(data, self.main_vector)
        rest_data_rot = np.matmul(data, self.rest_vector)
        data_main = np.matmul(main_data_rot, self.main_vector.T)
        data_rest = np.matmul(rest_data_rot, self.rest_vector.T)
        return data_main, data_rest

    def reconstruct(self, rest_x):
        return self.main_x + rest_x


def abnormal_data_process_for_week(data):
    data = np.delete(data, [51, 53, 89, 90, 93, 94, 95, 96, 97, 98, 103, 105], axis=1)
    return data


def create_train_test_diff_pca(step, train_num, test_num, retain, data_type='normal'):
    ld = sio.loadmat('data/new147k1.mat')
    flow_data = abnormal_data_process_for_week(ld['data']) * 1956
    flow_data[flow_data <= 3] = 3
    flow_data = flow_data[0:90 * 96, :]
    station_num = flow_data.shape[1]
    flow_diff = np.zeros(shape=[90 * 96, station_num])
    flow_week_pca_mean = np.zeros(shape=flow_diff.shape)
    flow_data_mean = flow_data.mean(axis=0)
    flow_remove_mean = flow_data - flow_data_mean
    flow_pca = PCA(flow_remove_mean, retain)

    def data_diff(first_day, after_abnormal_day, weeks, next_weeks=None):
        if next_weeks is None:
            next_weeks = weeks
        x = range((first_day - 1) * 96, first_day * 96)
        week_flow = flow_data[x, :][np.newaxis, :]
        for i in weeks:
            x = range(((first_day - 1) + i * 7) * 96, (first_day + i * 7) * 96)
            f = flow_data[x, :][np.newaxis, :]
            week_flow = np.concatenate((week_flow, f), axis=0)
        x = range((after_abnormal_day - 1) * 96, after_abnormal_day * 96)
        f = flow_data[x, :][np.newaxis, :]
        week_flow = np.concatenate((week_flow, f), axis=0)

        days, interval, stations = week_flow.shape
        week_flow_pca_main = np.zeros(shape=week_flow.shape)
        for i in range(days):
            week_flow_pca_main[i], _ = flow_pca.reduce(week_flow[i])
        week_flow_pca_mean = week_flow_pca_main.mean(axis=0)

        x = range((first_day - 1) * 96, first_day * 96)
        flow_diff[x, :] = flow_data[x, :] - week_flow_pca_mean
        flow_week_pca_mean[x, :] = week_flow_pca_mean
        for i in next_weeks:
            x = range((first_day - 1 + i * 7) * 96, (first_day + i * 7) * 96)
            flow_diff[x, :] = flow_data[x, :] - week_flow_pca_mean
            flow_week_pca_mean[x, :] = week_flow_pca_mean
        for i in range(0, 3):
            x = range((after_abnormal_day + i * 7 - 1) * 96, (after_abnormal_day + i * 7) * 96)
            flow_diff[x, :] = flow_data[x, :] - week_flow_pca_mean
            flow_week_pca_mean[x, :] = week_flow_pca_mean
        return week_flow_pca_mean

    monday_aver = data_diff(7, 69, [1, 3, 4, 5, 7, 8, 9], range(1, 10))
    data_diff(8, 70, range(1, 10), range(-1, 10))
    data_diff(2, 71, range(1, 11))
    data_diff(3, 72, range(1, 11))
    data_diff(4, 73, range(1, 11))
    data_diff(5, 74, range(1, 11))
    data_diff(6, 75, range(1, 10))
    xx = range(89 * 96, 90 * 96)
    flow_diff[xx, :] = flow_data[xx, :] - monday_aver  # 为了产生train和test数据方便，最后加了一天
    flow_week_pca_mean[xx, :] = monday_aver

    if data_type == 'normal':
        flow_diff_ns = Normal(flow_diff)
    else:
        flow_diff_ns = Standard(flow_diff)

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    train_y_aver = []
    test_y_aver = []

    for i in range(train_num):
        train_x.append(flow_diff_ns.data[i:i + step, :])
        train_y.append(flow_diff_ns.data[i + step, :])
        train_y_aver.append(flow_week_pca_mean[i + step, :])

    for i in range(test_num):
        test_x.append(flow_diff_ns.data[train_num + i:train_num + i + step, :])
        test_y.append(flow_diff_ns.data[train_num + i + step, :])
        test_y_aver.append(flow_week_pca_mean[train_num + i + step, :])

    train_x_np = np.array(train_x)
    train_y_np = np.array(train_y)
    test_x_np = np.array(test_x)
    test_y_np = np.array(test_y)
    train_y_aver_np = np.array(train_y_aver)
    test_y_aver_np = np.array(test_y_aver)

    flow_diff_trend = TrendDiff(flow_diff_ns, train_num, train_y_aver_np, test_num, test_y_aver_np)

    data = {'data_process': flow_diff_trend, 'train_x': train_x_np, 'train_y': train_y_np,
            'test_x': test_x_np, 'test_y': test_y_np}
    return data

# def create_train_test_pca(step, train_num, test_num, retain, data_type='standard'):
#     ld = sio.loadmat('data/new147k1.mat')
#     flow_data = abnormal_data_process_for_week(ld['data']) * 1956
#     flow_data[flow_data <= 3] = 3
#     flow_data = flow_data[0:90 * 96, :]
#     station_num = flow_data.shape[1]
#     flow_main = np.zeros(shape=[90 * 96, station_num])
#     # flow_main是flow_pca_main+flow_mean
#     # 运算关系如下：
#     # flow_data = flow_mean + flow_diff
#     # flow_diff = flow_pca_main + flow_pca_rest
#     # flow_data = flow_mean + flow_pca_main + flow_pca_rest = flow_main + flow_pca_rest
#     # flow_pca_rest就是用来预测的数据
#     flow_pca_rest = np.zeros(shape=[90 * 96, station_num])
#
#     def data_diff(first_day, after_abnormal_day, weeks, next_weeks=None):
#         if next_weeks is None:
#             next_weeks = weeks
#         days = range((first_day - 1) * 96, first_day * 96)
#         # week_day_flow是3d的，三维分别是9*96*135，days, interval, stations
#         week_day_flow = flow_data[days, :][np.newaxis, :]
#         for i in weeks:
#             days = range(((first_day - 1) + i * 7) * 96, (first_day + i * 7) * 96)
#             f = flow_data[days, :][np.newaxis, :]
#             week_day_flow = np.concatenate((week_day_flow, f), axis=0)
#         days = range((after_abnormal_day - 1) * 96, after_abnormal_day * 96)
#         f = flow_data[days, :][np.newaxis, :]
#         week_day_flow = np.concatenate((week_day_flow, f), axis=0)
#
#         days, interval, stations = week_day_flow.shape
#         week_flow_2d = np.reshape(week_day_flow, newshape=[days * interval, stations])
#         week_flow_mean = week_flow_2d.mean(axis=0)
#         week_flow_diff = week_flow_2d - week_flow_mean
#         week_flow_diff_pca = PCA(week_flow_diff, retain)
#
#         def data_main_rest(days):
#             f = flow_data[days, :]
#             f_diff = f - week_flow_mean
#             f_pca_main, f_pca_rest = week_flow_diff_pca.reduce(f_diff)
#             flow_main[days] = f_pca_main + week_flow_mean
#             flow_pca_rest[days] = f_pca_rest
#
#         days = range((first_day - 1) * 96, first_day * 96)
#         data_main_rest(days)
#         for i in next_weeks:
#             days = range((first_day - 1 + i * 7) * 96, (first_day + i * 7) * 96)
#             data_main_rest(days)
#         for i in range(0, 3):
#             days = range((after_abnormal_day + i * 7 - 1) * 96, (after_abnormal_day + i * 7) * 96)
#             data_main_rest(days)
#
#     data_diff(7, 69, range(1, 10))
#     data_diff(1, 70, range(1, 11))
#     data_diff(2, 71, range(1, 11))
#     data_diff(3, 72, range(1, 11))
#     data_diff(4, 73, range(1, 11))
#     data_diff(5, 74, range(1, 11))
#     data_diff(6, 75, range(1, 10))
#     day_90 = range(89 * 96, 90 * 96)
#     day_7 = range(6 * 96, 7 * 96)
#     flow_main[day_90, :] = flow_main[day_7, :]
#     flow_pca_rest[day_90, :] = flow_pca_rest[day_7, :]
#
#     train_x = []
#     train_y = []
#     test_x = []
#     test_y = []
#     train_y_main = []
#     test_y_main = []
#
#     if data_type == 'normal':
#         flow_pca_rest_ns = Normal(flow_pca_rest)
#     else:
#         flow_pca_rest_ns = Standard(flow_pca_rest)
#
#     for i in range(train_num):
#         train_x.append(flow_pca_rest_ns.data[i:i + step, :])
#         train_y.append(flow_pca_rest_ns.data[i + step, :])
#         train_y_main.append(flow_main[i + step, :])
#
#     for i in range(test_num):
#         test_x.append(flow_pca_rest_ns.data[train_num + i:train_num + i + step, :])
#         test_y.append(flow_pca_rest_ns.data[train_num + i + step, :])
#         test_y_main.append(flow_main[train_num + i + step, :])
#
#     train_x_np = np.array(train_x)
#     train_y_np = np.array(train_y)
#     test_x_np = np.array(test_x)
#     test_y_np = np.array(test_y)
#
#     flow_pca_trend = TrendDiff(flow_pca_rest_ns, train_num, train_y_main, test_num, test_y_main)
#
#     data = {'data_process': flow_pca_trend, 'train_x': train_x_np, 'train_y': train_y_np,
#             'test_x': test_x_np, 'test_y': test_y_np}
#     return data


# def create_train_test_pca_mean(step, train_num, test_num, retain, data_type='normal'):
#     ld = sio.loadmat('data/new147k1.mat')
#     flow_data = abnormal_data_process_for_week(ld['data']) * 1956
#     flow_data[flow_data <= 3] = 3
#     flow_data = flow_data[0:90 * 96, :]
#     station_num = flow_data.shape[1]
#     flow_main = np.zeros(shape=[90 * 96, station_num])
#     flow_test_main_mean = np.zeros(shape=[90 * 96, station_num])
#     # flow_main是flow_pca_main+flow_mean
#     # 运算关系如下：
#     # flow_data = flow_mean + flow_diff
#     # flow_diff = flow_pca_main + flow_pca_rest
#     # flow_data = flow_mean + flow_pca_main + flow_pca_rest = flow_main + flow_pca_rest
#     # flow_pca_rest就是用来预测的数据
#     flow_pca_rest = np.zeros(shape=[90 * 96, station_num])
#
#     def data_diff(first_day, after_abnormal_day, weeks, next_weeks=None):
#         if next_weeks is None:
#             next_weeks = weeks
#         days = range((first_day - 1) * 96, first_day * 96)
#         # week_day_flow是3d的，三维分别是9*96*135，days, interval, stations
#         week_day_flow = flow_data[days, :][np.newaxis, :]
#         for i in weeks:
#             days = range(((first_day - 1) + i * 7) * 96, (first_day + i * 7) * 96)
#             f = flow_data[days, :][np.newaxis, :]
#             week_day_flow = np.concatenate((week_day_flow, f), axis=0)
#         days = range((after_abnormal_day - 1) * 96, after_abnormal_day * 96)
#         f = flow_data[days, :][np.newaxis, :]
#         week_day_flow = np.concatenate((week_day_flow, f), axis=0)
#         days, interval, stations = week_day_flow.shape
#         week_flow_2d = np.reshape(week_day_flow, newshape=[days * interval, stations])
#         week_flow_mean = week_flow_2d.mean(axis=0)
#         week_flow_diff = week_flow_2d - week_flow_mean
#         week_flow_diff_pca = PCA(week_flow_diff, retain)
#         week_day_flow_pca_main = np.zeros(shape=week_day_flow.shape)
#         for i in range(days):
#             week_day_flow_pca_main[i], _ = week_flow_diff_pca.reduce(week_day_flow[i])
#         week_day_flow_pca_main_mean = week_day_flow_pca_main.mean(axis=0)
#
#         def data_main_rest(days):
#             f = flow_data[days, :]
#             f_diff = f - week_flow_mean
#             f_pca_main, f_pca_rest = week_flow_diff_pca.reduce(f_diff)
#             flow_main[days] = f_pca_main + week_flow_mean
#             flow_test_main_mean[days, :] = week_day_flow_pca_main_mean
#             flow_pca_rest[days] = f_pca_rest
#
#         days = range((first_day - 1) * 96, first_day * 96)
#         data_main_rest(days)
#         for i in next_weeks:
#             days = range((first_day - 1 + i * 7) * 96, (first_day + i * 7) * 96)
#             data_main_rest(days)
#         days = range((after_abnormal_day - 1) * 96, after_abnormal_day * 96)
#         data_main_rest(days)
#         for i in range(1, 3):
#             days = range((after_abnormal_day + i * 7 - 1) * 96, (after_abnormal_day + i * 7) * 96)
#             data_main_rest(days)
#
#     data_diff(7, 69, range(1, 10))
#     data_diff(1, 70, range(1, 11))
#     data_diff(2, 71, range(1, 11))
#     data_diff(3, 72, range(1, 11))
#     data_diff(4, 73, range(1, 11))
#     data_diff(5, 74, range(1, 11))
#     data_diff(6, 75, range(1, 10))
#     day_90 = range(89 * 96, 90 * 96)
#     day_7 = range(6 * 96, 7 * 96)
#     flow_main[day_90, :] = flow_main[day_7, :]
#     flow_test_main_mean[day_90, :] = flow_test_main_mean[day_7, :]
#     flow_pca_rest[day_90, :] = flow_pca_rest[day_7, :]
#
#     train_x = []
#     train_y = []
#     test_x = []
#     test_y = []
#     train_y_main = []
#     test_y_main = []
#     test_y_main_mean = []
#
#     if data_type == 'normal':
#         flow_pca_rest_ns = Normal(flow_pca_rest)
#     else:
#         flow_pca_rest_ns = Standard(flow_pca_rest)
#
#     for i in range(train_num):
#         train_x.append(flow_pca_rest_ns.data[i:i + step, :])
#         train_y.append(flow_pca_rest_ns.data[i + step, :])
#         train_y_main.append(flow_main[i + step, :])
#
#     for i in range(test_num):
#         test_x.append(flow_pca_rest_ns.data[train_num + i:train_num + i + step, :])
#         test_y.append(flow_pca_rest_ns.data[train_num + i + step, :])
#         test_y_main.append(flow_main[train_num + i + step, :])
#         test_y_main_mean.append(flow_test_main_mean[train_num + i + step, :])
#
#     train_x_np = np.array(train_x)
#     train_y_np = np.array(train_y)
#     test_x_np = np.array(test_x)
#     test_y_np = np.array(test_y)
#
#     flow_pca_trend = TrendDiff(flow_pca_rest_ns, train_num, np.array(train_y_main), test_num, np.array(test_y_main),
#                                np.array(test_y_main_mean))
#
#     data = {'data_process': flow_pca_trend, 'train_x': train_x_np, 'train_y': train_y_np,
#             'test_x': test_x_np, 'test_y': test_y_np}
#     return data

def create_train_test_diff(step, train_num, test_num, data_type='normal'):
    ld = sio.loadmat('data/new147k1.mat')
    flow_data = abnormal_data_process_for_week(ld['data']) * 1956
    flow_data[flow_data <= 3] = 3
    flow_data = flow_data[0:90 * 96, :]
    station_num = flow_data.shape[1]
    flow_diff = np.zeros(shape=[90 * 96, station_num])
    flow_week_mean = np.zeros(shape=flow_diff.shape)

    def data_diff(first_day, after_abnormal_day, weeks, next_weeks=None):
        if next_weeks is None:
            next_weeks = weeks
        x = range((first_day - 1) * 96, first_day * 96)
        week_flow = flow_data[x, :][np.newaxis, :]
        for i in weeks:
            x = range(((first_day - 1) + i * 7) * 96, (first_day + i * 7) * 96)
            f = flow_data[x, :][np.newaxis, :]
            week_flow = np.concatenate((week_flow, f), axis=0)
        x = range((after_abnormal_day - 1) * 96, after_abnormal_day * 96)
        f = flow_data[x, :][np.newaxis, :]
        week_flow = np.concatenate((week_flow, f), axis=0)
        week_flow_mean = week_flow.mean(axis=0)

        x = range((first_day - 1) * 96, first_day * 96)
        flow_diff[x, :] = flow_data[x, :] - week_flow_mean
        flow_week_mean[x, :] = week_flow_mean
        for i in next_weeks:
            x = range((first_day - 1 + i * 7) * 96, (first_day + i * 7) * 96)
            flow_diff[x, :] = flow_data[x, :] - week_flow_mean
            flow_week_mean[x, :] = week_flow_mean
        for i in range(0, 3):
            x = range((after_abnormal_day + i * 7 - 1) * 96, (after_abnormal_day + i * 7) * 96)
            flow_diff[x, :] = flow_data[x, :] - week_flow_mean
            flow_week_mean[x, :] = week_flow_mean
        return week_flow_mean

    monday_aver = data_diff(7, 69, [1, 3, 4, 5, 7, 8, 9], range(1, 10))
    data_diff(8, 70, range(1, 10), range(-1, 10))
    data_diff(2, 71, range(1, 11))
    data_diff(3, 72, range(1, 11))
    data_diff(4, 73, range(1, 11))
    data_diff(5, 74, range(1, 11))
    data_diff(6, 75, range(1, 10))
    xx = range(89 * 96, 90 * 96)
    flow_diff[xx, :] = flow_data[xx, :] - monday_aver  # 为了产生train和test数据方便，最后加了一天
    flow_week_mean[xx, :] = monday_aver

    if data_type == 'normal':
        flow_diff_ns = Normal(flow_diff)
    else:
        flow_diff_ns = Standard(flow_diff)

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    train_y_aver = []
    test_y_aver = []

    for i in range(train_num):
        train_x.append(flow_diff_ns.data[i:i + step, :])
        train_y.append(flow_diff_ns.data[i + step, :])
        train_y_aver.append(flow_week_mean[i + step, :])

    for i in range(test_num):
        test_x.append(flow_diff_ns.data[train_num + i:train_num + i + step, :])
        test_y.append(flow_diff_ns.data[train_num + i + step, :])
        test_y_aver.append(flow_week_mean[train_num + i + step, :])

    train_x_np = np.array(train_x)
    train_y_np = np.array(train_y)
    test_x_np = np.array(test_x)
    test_y_np = np.array(test_y)
    train_y_aver_np = np.array(train_y_aver)
    test_y_aver_np = np.array(test_y_aver)

    flow_diff_trend = TrendDiff(flow_diff_ns, train_num, train_y_aver_np, test_num, test_y_aver_np)

    data = {'data_process': flow_diff_trend, 'train_x': train_x_np, 'train_y': train_y_np,
            'test_x': test_x_np, 'test_y': test_y_np}
    return data
