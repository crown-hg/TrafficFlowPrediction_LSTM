# -*- coding:utf-8 -*-
import time
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
import get_data as gd


def lstm_test(hidden_size, layer_num, dropout_rate):
    start_time = datetime.datetime.now()

    # 设置 GPU 按需增长
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ####################################
    #  设置超参数

    lr = 1e-3
    # 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
    batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32
    # 在 1.0 版本以后请使用 ：
    # keep_prob = tf.placeholder(tf.float32, [])
    # batch_size = tf.placeholder(tf.int32, [])

    # 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
    input_size = 143
    # 所用时间段的个数
    timestep_size = 4
    # 每个隐含层的节点数
    # hidden_size = 200
    # LSTM layer 的层数
    # layer_num = 2
    # dropout
    # dropout_rate = 0.5
    # 输出的结点数
    output_size = 143
    # 训练次数
    max_epoch = 20000
    # 训练集大小
    train_num = 71 * 96
    # 测试集大小
    test_num = 18 * 96
    sess = tf.Session(config=config)
    _X = tf.placeholder(tf.float32, [None, timestep_size, input_size])
    _Y = tf.placeholder(tf.float32, [None, output_size])
    # dropout的留下的神经元的比例
    keep_prob = tf.placeholder(tf.float32, [])

    ####################################################################
    # 根据timestep_size获取数据
    train_x, train_y, test_x, test_y = gd.train_f_test_f(timestep_size)

    # 下面几个步骤是实现 RNN / LSTM 的关键
    ####################################################################
    # **步骤1：RNN 的输入shape = (batch_size, timestep_size, input_size)
    X = _X

    def lstm_multi_cell(cell_num):
        # **步骤2：定义LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
        lstm_multi_cell = []
        for _ in range(cell_num):
            lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
            # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
            lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
            lstm_multi_cell.append(lstm_cell)
        return lstm_multi_cell

    # **步骤4：调用 MultiRNNCell 来实现多层 LSTM
    mlstm_cell = rnn.MultiRNNCell(lstm_multi_cell(layer_num), state_is_tuple=True)

    # **步骤5：用全零来初始化state
    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

    # **步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
    # ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
    # ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
    # ** state.shape = [layer_num, 2, batch_size, hidden_size]（中间的‘2’指的是每个cell中有两层分别是c和h）,
    # ** 或者，可以取 h_state = state[-1][1] 作为最后输出
    # ** 最后输出维度是 [batch_size, hidden_size]
    # outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
    # h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]

    # *************** 为了更好的理解 LSTM 工作原理，我们把上面 步骤6 中的函数自己来实现 ***************
    # 通过查看文档你会发现， RNNCell 都提供了一个 __call__()函数（见最后附），我们可以用它来展开实现LSTM按时间步迭代。
    # **步骤6：方法二，按时间步展开计算
    # outputs = list()
    # state = init_state
    # with tf.variable_scope('RNN'):
    #     for timestep in range(timestep_size):
    #         if timestep > 0:
    #             tf.get_variable_scope().reuse_variables()
    #         # 这里的state保存了每一层 LSTM 的状态
    #         (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
    #         outputs.append(cell_output)
    # h_state = outputs[-1]
    outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
    h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]

    # 输出层
    W_o = tf.Variable(tf.truncated_normal([hidden_size, output_size], stddev=0.1), dtype=tf.float32)
    b_o = tf.Variable(tf.constant(0.1, shape=[output_size]), dtype=tf.float32)
    y_pre = tf.add(tf.matmul(h_state, W_o), b_o)

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
    for i in range(1, max_epoch + 1):
        if i % 50 == 0:
            train_mre, train_mae, train_rmse = model_run(train_x, train_y, 1.0, train_num)
            test_mre, test_mae, test_rmse = model_run(test_x, test_y, 1.0, test_num)
            train_mre_reult.append(train_mre)
            test_mre_reult.append(test_mre)
            if i % 1000 == 0:
                current_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
                test_result = '\n%s\t%d\t%d\t\t%d\t%d\t%.1f\t%.4f\t%.2f\t%.2f' % (
                    current_time, layer_num, hidden_size, timestep_size, i, dropout_rate, test_mre, test_mae, test_rmse)
                with open('Result.txt', 'a') as fp:
                    fp.write(test_result)
                end_time = datetime.datetime.now()
                print(end_time - start_time)
            print('epoch ', i, 'train', train_mre, train_mae, train_rmse, 'test', test_mre, test_mae, test_rmse)
        sess.run(train_op, feed_dict={_X: train_x, _Y: train_y, keep_prob: dropout_rate, batch_size: train_num})
    plt.plot(train_mre_reult, 'b-', label='train')
    plt.plot(test_mre_reult, 'r-', label='test')
    plt.legend()
    plt.savefig('result_picture/hn%dts%ddp%.f.png' % (hidden_size, timestep_size, dropout_rate))
    plt.show()
    sess.close()
    tf.reset_default_graph()
    return train_mre_reult, test_mre_reult

# test_mre, test_mae, test_rmse = model_run(train_x, train_y, 1.0, train_num)
# current_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
# test_result = '\n%s\t%d\t%d\t\t%d\t%d\t%.4f\t%.2f\t%.2f' % (
#     current_time, layer_num, hidden_size, timestep_size, max_epoch, test_mre, test_mae, test_rmse)
# with open('./Result.txt', 'a') as fp:
#     fp.write(test_result)


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
