# -*- coding:utf-8 -*-
import time
import datetime
import tensorflow as tf
import numpy as np
from lstm_model import lstm
from train_log import print_to_console
from nn_model import nn

start_time = datetime.datetime.now()


def lstm_test(hidden_size, layer_num, max_epoch, dropout_keep_rate, data, file_name, lr=1e-3, weights=None, biases=None,
              act_function=None, gpu_device=0):
    # ################参数的含义###################
    # 所用时间段的个数 timestep_size = 4
    # 每个隐含层的节点数hidden_size = 200
    # LSTM layer 的层数layer_num = 2
    # dropout_rate = 0.5
    # 输出的结点数output_size = 143
    # 训练次数max_epoch = 20000
    # 是否使用rbm use_rbm = False
    # rbm的权值rbm_w = None
    # rbm的bias rbm_b = None
    # #############################################

    train_x = data['train_x']
    train_y = data['train_y']

    # 根据输入数据来决定，train_num训练集大小,input_size输入维度
    train_num, time_step_size, input_size = train_x.shape
    # output_size输出的结点个数
    _, output_size = train_y.shape

    with tf.device('/gpu:%d' % gpu_device):
        # **步骤1：LSTM 的输入shape = (batch_size, time_step_size, input_size)，输出shape=(batch_size, output_size)
        x_input = tf.placeholder(tf.float32, [None, time_step_size, input_size])
        y_real = tf.placeholder(tf.float32, [None, output_size])

        # dropout的留下的神经元的比例
        keep_prob = tf.placeholder(tf.float32, [])

        # 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
        batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32

        if weights is not None:
            # 输入层赋初始值
            pre_layer_hidden_num = len(weights)
            pre_layer_hidden_size = len(biases[0])
            hide_output = nn(x_input, pre_layer_hidden_num, pre_layer_hidden_size, hide_act_function=act_function,
                             weights=weights, biases=biases)
        else:
            pre_layer_hidden_num = 0
            pre_layer_hidden_size = 0
            hide_output = x_input

    y_pred = lstm(layer_num, hidden_size, batch_size, output_size, hide_output, keep_prob)
    # 损失和评估函数
    mse = tf.losses.mean_squared_error(y_real, y_pred)
    train_op = tf.train.AdamOptimizer(lr).minimize(mse)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    # 设置 GPU 按需增长
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # 初始化变量
    sess.run(tf.global_variables_initializer())

    mre_result = []
    mae_result = []
    rmse_result = []

    for i in range(1, max_epoch + 1):
        feed_dict = {x_input: train_x, y_real: train_y, keep_prob: dropout_keep_rate, batch_size: train_num}
        sess.run(train_op, feed_dict=feed_dict)
        if i % 50 == 0:
            feed_dict = {x_input: train_x, y_real: train_y, keep_prob: 1.0, batch_size: train_num}
            train_y_pred = sess.run(y_pred, feed_dict=feed_dict)
            feed_dict = {x_input: data['test_x'], y_real: data['test_y'], keep_prob: 1.0,
                         batch_size: data['test_y'].shape[0]}
            test_y_pred = sess.run(y_pred, feed_dict=feed_dict)
            mre, mae, rmse = print_to_console(i, train_y_pred, test_y_pred, data)
            mre_result.append(mre)
            mae_result.append(mae)
            rmse_result.append(rmse)
            if i % 1000 == 0:
                global start_time
                current_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
                end_time = datetime.datetime.now()
                time_spend = (end_time - start_time).seconds
                start_time = datetime.datetime.now()
                test_result = '\n%s\t%d\t%d\t%d\t%d\t%.4f\t%d\t%d\t%.1f\t%.4f\t%.2f\t%.2f\t%s' % (
                    current_time, layer_num, pre_layer_hidden_num, pre_layer_hidden_size, hidden_size, lr,
                    time_step_size, i, dropout_keep_rate, mre, mae, rmse, time_spend)
                with open('result_log/%s.txt' % file_name, 'a') as fp:
                    fp.write(test_result)

    with open('result_log/%s.txt' % file_name, 'a') as fp:
        min_index = mre_result.index(np.min(mre_result))
        line = '\nepoch=%d min_mre %.4f %.2f %.2f' % (
            (min_index + 1) * 50, mre_result[min_index], mae_result[min_index], rmse_result[min_index])
        fp.write(line)
    print(line)
    sess.close()
    # 还原图，否则tensorflow再次运行会报错
    tf.reset_default_graph()
    return mre_result, mae_result, rmse_result
