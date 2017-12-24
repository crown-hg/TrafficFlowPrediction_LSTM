# -*- coding:utf-8 -*-
import time
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn


def lstm_test(hidden_size, layer_num, max_epoch, dropout_keep_rate, train_x, train_y, test_x, test_y,
              file_name, use_rbm=False, rbm_W=None, rbm_b=None):
    # 所用时间段的个数 timestep_size = 4
    # 每个隐含层的节点数hidden_size = 200
    # LSTM layer 的层数layer_num = 2
    # dropout_rate = 0.5
    # 输出的结点数output_size = 143
    # 训练次数max_epoch = 20000

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    # 设置 GPU 按需增长
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    ####################################

    # train_num训练集大小,input_size输入维度
    train_num, time_step_size, input_size = train_x.shape
    # output_size输出维度
    _, output_size = train_y.shape
    # train_num测试集大小
    test_num, _, _ = test_x.shape

    # 下面几个步骤是实现 RNN / LSTM 的关键
    ####################################################################
    # **步骤1：LSTM 的输入shape = (batch_size, timestep_size, input_size)，输出shape=(batch_size, output_size)
    _X = tf.placeholder(tf.float32, [None, time_step_size, input_size])
    _Y = tf.placeholder(tf.float32, [None, output_size])

    # dropout的留下的神经元的比例
    keep_prob = tf.placeholder(tf.float32, [])

    # 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
    batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32

    # 学习率
    lr = 1e-3

    if use_rbm:
        # rbm输入层
        _, rbm_output_size = rbm_W.shape
        lstm_x = tf.layers.dense(_X, rbm_output_size, kernel_initializer=tf.constant_initializer(rbm_W),
                                 bias_initializer=tf.constant_initializer(rbm_b))
    else:
        lstm_x = _X

    def multi_cells(cell_num):
        # 多cell的lstm必须多次建立cell保存在一个list当中
        multi_cell = []
        for _ in range(cell_num):
            # **步骤2：定义LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
            lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
            # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
            lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
            multi_cell.append(lstm_cell)
        return multi_cell

    # **步骤4：调用 MultiRNNCell 来实现多层 LSTM
    mlstm_cell = rnn.MultiRNNCell(multi_cells(layer_num), state_is_tuple=True)

    # **步骤5：用全零来初始化state
    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

    # **步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
    # ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
    # ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
    # ** state.shape = [layer_num, 2, batch_size, hidden_size]（中间的‘2’指的是每个cell中有两层分别是c和h）,
    # ** 或者，可以取 h_state = state[-1][1] 作为最后输出
    # ** 最后输出维度是 [batch_size, hidden_size]
    outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=lstm_x, initial_state=init_state, time_major=False)
    h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]

    # 输出层
    # W_o = tf.Variable(tf.truncated_normal([hidden_size, output_size], stddev=0.1), dtype=tf.float32)
    # b_o = tf.Variable(tf.constant(0.1, shape=[output_size]), dtype=tf.float32)
    # y_pre = tf.add(tf.matmul(h_state, W_o), b_o)
    y_pre = tf.layers.dense(h_state, output_size)

    # 损失和评估函数
    mse = tf.losses.mean_squared_error(_Y, y_pre)
    train_op = tf.train.AdamOptimizer(lr).minimize(mse)

    mre = tf.reduce_mean(tf.div(tf.abs(tf.subtract(_Y, y_pre)), _Y))
    mae = tf.reduce_mean(tf.abs(tf.subtract(_Y, y_pre))) * 1956
    rmse = tf.sqrt(mse) * 1956

    # 初始化变量
    sess.run(tf.global_variables_initializer())

    def model_run(x, y, kp, bs):
        t_mre, t_mae, t_rmse = sess.run([mre, mae, rmse],
                                        feed_dict={_X: x, _Y: y, keep_prob: kp,
                                                   batch_size: bs})
        return round(t_mre, 4), round(t_mae, 2), round(t_rmse, 2)

    train_mre_reult = []
    test_mre_reult = []
    # 训练和测试
    start_time = datetime.datetime.now()
    for i in range(1, max_epoch + 1):
        if i % 5 == 0:
            train_mre, train_mae, train_rmse = model_run(train_x, train_y, 1.0, train_num)
            test_mre, test_mae, test_rmse = model_run(test_x, test_y, 1.0, test_num)
            train_mre_reult.append(train_mre)
            test_mre_reult.append(test_mre)
            if i % 10 == 0:
                current_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
                end_time = datetime.datetime.now()
                time_spend = (end_time - start_time).seconds
                start_time = datetime.datetime.now()
                test_result = '\n%s\t%d\t%d\t\t%d\t%d\t%.1f\t%.4f\t%.2f\t%.2f\t%s' % (
                    current_time, layer_num, hidden_size, time_step_size, i, dropout_keep_rate, test_mre, test_mae,
                    test_rmse, time_spend)
                with open(file_name, 'a') as fp:
                    fp.write(test_result)
            print('epoch ', i, 'train', train_mre, train_mae, train_rmse, 'test', test_mre, test_mae, test_rmse)
        sess.run(train_op, feed_dict={_X: train_x, _Y: train_y, keep_prob: dropout_keep_rate, batch_size: train_num})
    print('min_mre=%.4f\n' % np.min(test_mre_reult))

    # 画图
    plt.plot(train_mre_reult, 'b-', label='train')
    plt.plot(test_mre_reult, 'r-', label='test')
    plt.legend()
    plt.savefig('result_picture/hn%dts%ddp%.f.png' % (hidden_size, time_step_size, dropout_keep_rate))
    plt.show()
    sess.close()
    tf.reset_default_graph()
    return train_mre_reult, test_mre_reult

# def draw_picture():
#     re = tf.reduce_mean(tf.div(tf.abs(tf.subtract(_Y, y_pre)), _Y), 0)
#     rr = sess.run(re, feed_dict={_X: test_x, _Y: test_y, keep_prob: 1.0, batch_size: test_num})
#     plt.plot(range(0, 147), rr, '*')
#     plt.savefig('re.png')
#     pred = sess.run(y_pre, feed_dict={_X: test_x, _Y: test_y, keep_prob: 1.0, batch_size: test_num})
#     plt.plot(range(test_num), test_y[:, 95], 'b-', label='real')
#     plt.plot(range(test_num), pred[:, 95], 'r-', label='pred')
#     plt.legend()
#     plt.savefig('95.png')
#     plt.show()
