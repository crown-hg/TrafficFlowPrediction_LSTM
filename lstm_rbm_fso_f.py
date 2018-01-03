# -*- coding:utf-8 -*-
import tensorflow as tf
from get_data import *
from lstm_multicell import lstm_test
from tfrbm import GBRBM
from dbn_model import dbn

gpu_device = 1
time_step = 4
train_num = 71 * 96
test_num = 18 * 96
# lstm的hyper-parameters
# dbn_hidden_size = 400
data_type = 'normal'
lstm_layer_num = 1
dbn_hidden_num = 3
max_epoch = 20000
dropout_keep_rate = 0.9

# 取数据
if data_type == 'standard':
    x = np.hstack((flow_standardized, speed_standardized, occupancy_standardized))
    y = flow_standardized
else:
    x = np.hstack((flow_normalized, speed_normalized, occupancy_normalized))
    y = flow_normalized
train_x, train_y, test_x, test_y = create_train_test(x, y, time_step, train_num, test_num)
dbn_x = train_x[:, 0, :]

for dbn_hidden_size in [400]:
    with tf.device('/gpu:%d' % gpu_device):
        # 训练rbm
        _, _, dbn_input_size = train_x.shape
        weights, biases = dbn(dbn_hidden_num, dbn_input_size, dbn_hidden_size, dbn_x)

    # 存储运行结果的文件
    # date_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    file_name = 'fso_f_lstm_gbrbm'
    for hidden_size in [500]:
        test_mre, test_mae, test_rmse = lstm_test(hidden_size, lstm_layer_num, max_epoch, dropout_keep_rate,
                                                  train_x, train_y, test_x, test_y, file_name,
                                                  weights=weights, biases=biases, gpu_device=gpu_device)
