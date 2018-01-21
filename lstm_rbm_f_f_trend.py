# -*- coding:utf-8 -*-
import tensorflow as tf
import trend_data_diff_aver as tdda
from lstm_multicell import lstm_test
from dbn_model import dbn

gpu_device = 2
time_step = 3
train_num = 75 * 96
test_num = 14 * 96
# lstm的hyper-parameters
dbn_hidden_size = 400
data_type = 'normal'
lstm_layer_num = 1
dbn_hidden_num = 1
max_epoch = 30000
dropout_keep_rate = 0.9
hidden_size = 200

# 取数据
data = tdda.create_train_test_week(time_step, train_num, test_num)

dbn_x = data['train_x'][:, 0, :]
for dbn_hidden_size in [100]:
    with tf.device('/gpu:%d' % gpu_device):
        # 训练rbm
        _, _, dbn_input_size = data['train_x'].shape
        weights, biases = dbn(dbn_hidden_num, dbn_input_size, dbn_hidden_size, dbn_x)
    for hidden_size in [100, 200, 300, 400]:
        file_name = 'f_f_trend_lstm_rbm'
        test_mre, test_mae, test_rmse = lstm_test(hidden_size, lstm_layer_num, max_epoch, dropout_keep_rate,
                                                  data, file_name, weights=weights, biases=biases,
                                                  gpu_device=gpu_device)
