# -*- coding:utf-8 -*-
import tensorflow as tf
from get_data import *
from lstm_multicell import lstm_test
from sae_model import sae

gpu_device = 0

train_num = 71 * 96
test_num = 18 * 96
# lstm的hyper-parameters
time_step = 4
data_type = 'normal'
lstm_layer_num = 1
max_epoch = 30000
dropout_keep_rate = 0.9
# sae的参数
sae_hidden_num = 1
sae_epoches = 400
sae_lr = 0.01
hide_function = tf.nn.sigmoid
file_name = 'f_f_lstm_sae_sigmoid'

# 取数据
x = flow_normalized
y = flow_normalized

train_x, train_y, test_x, test_y = create_train_test(x, y, time_step, train_num, test_num)

for sae_hidden_size in [300, 400, 500]:
    with tf.device('/gpu:%d' % gpu_device):
        # 训练sae
        sae_x = train_x[:, 0, :]
        weights, biases = sae(sae_x, sae_hidden_num, sae_hidden_size, sae_epoches, lr=sae_lr,
                              hide_function=hide_function)

    # 存储运行结果的文件
    # date_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    for lstm_hidden_size in [300, 400, 500]:
        test_mre, test_mae, test_rmse = lstm_test(lstm_hidden_size, lstm_layer_num, max_epoch, dropout_keep_rate,
                                                  train_x, train_y, test_x, test_y, file_name,
                                                  weights=weights, biases=biases, gpu_device=gpu_device)
